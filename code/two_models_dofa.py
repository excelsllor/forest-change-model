"""One DOFA encoder, two separate models on the same block family.

Mask: 768-d DOFA features + 3x3 context + CtxAttn + ONE CfCBiBlock + AttnReadout.
Type: 768-d DOFA features + 3x3 context + CtxAttn + TWO CfCBiBlocks + AttnReadout
      + texture9/spec9 per date.
The encoder is the one already used by TypeNet (CnnDofaLike, 768-d).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
TEX = os.path.join(ROOT, "outputs", "vozh", "tex_exp", "vozh_feats.npz")
OUT = os.path.join(ROOT, "outputs", "vozh", "two_models_dofa")
sys.path.insert(0, SRC)

from cfc import CfCBiBlock  # noqa: E402
from exp_ctx_mask import CtxAttn  # noqa: E402
from typenet_big import AttnReadout  # noqa: E402

WIDTH = 256
HIDDEN = 128
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiCfCCore(nn.Module):
    def __init__(self, d_in, width=WIDTH, hidden=HIDDEN, blocks=2, n_queries=2, ctx_dim=768):
        super().__init__()
        self.proj = nn.Linear(d_in + 3, width)
        self.ctx_proj = nn.Linear(ctx_dim, width)
        self.ctx = CtxAttn(3, width)
        self.blocks = nn.ModuleList([
            CfCBiBlock(width if i == 0 else 2 * hidden, hidden) for i in range(blocks)])
        self.pool = AttnReadout(2 * hidden, width=width, n_queries=n_queries)

    def forward(self, feats, ctx, valid, dt, doy, extra):
        parts = [feats]
        if extra.shape[-1] > 0:
            parts.append(extra)
        parts.append(doy)
        x = self.proj(torch.cat(parts, dim=-1))
        ctx = self.ctx_proj(ctx)
        b, t, _ = x.shape
        x = self.ctx(x.reshape(b * t, -1), ctx.reshape(b * t, 3, 3, -1),
                     valid.reshape(b * t, 3, 3)).reshape(b, t, -1)
        for blk in self.blocks:
            x = blk(x, dt)
        L = (dt > 0).sum(dim=1) + 1
        vmask = torch.arange(x.shape[1], device=x.device)[None] < L[:, None]
        return self.pool(x, vmask)


class MaskModel(nn.Module):
    def __init__(self, d_in=768, blocks=1):
        super().__init__()
        self.core = BiCfCCore(d_in, blocks=blocks)
        self.head = nn.Sequential(nn.LayerNorm(2 * WIDTH), nn.Linear(2 * WIDTH, 1))

    def forward(self, feats, ctx, valid, dt, doy):
        extra = feats.new_zeros((feats.shape[0], feats.shape[1], 0))
        return self.head(self.core(feats, ctx, valid, dt, doy, extra)).reshape(-1)


class TypeModel(nn.Module):
    def __init__(self, d_in=768, blocks=2, nc=3):
        super().__init__()
        self.core = BiCfCCore(d_in + 18, blocks=blocks)
        self.head = nn.Sequential(nn.LayerNorm(2 * WIDTH), nn.Linear(2 * WIDTH, 64),
                                  nn.GELU(), nn.Dropout(0.2), nn.Linear(64, nc))

    def forward(self, feats, ctx, valid, dt, doy, tex, spec):
        extra = torch.cat([tex, spec], dim=-1)
        return self.head(self.core(feats, ctx, valid, dt, doy, extra))


def load_mask():
    z = np.load(os.path.join(EVAL, "vozh_ctx_mask_big.npz"), allow_pickle=True)
    dt = z["dt"].astype(np.float32)
    L = (dt > 0).sum(1) + 1
    cv = np.asarray(z["center_valid"], bool)
    ok = np.array([cv[i, :L[i]].all() for i in range(len(L))])
    return dict(
        feats=torch.from_numpy(z["center"].astype(np.float32)),
        ctx=torch.from_numpy(z["ctx3"].astype(np.float32)),
        valid=torch.from_numpy(np.asarray(z["valid3"])),
        dt=torch.from_numpy(dt),
        doy=torch.from_numpy(z["doy"].astype(np.float32)),
        y=torch.from_numpy(z["y"].astype(np.int64)),
        split=np.array([str(x) for x in z["split"]]),
        per=np.array([str(x) for x in z["period"]]), ok=ok)


def load_type():
    z = np.load(os.path.join(EVAL, "vozh_ctx_mask_big.npz"), allow_pickle=True)
    t = np.load(os.path.join(EVAL, "vozh_ctx_type.npz"), allow_pickle=True)
    tx = np.load(TEX)
    key = {(str(p), int(i), int(j)): k for k, (p, i, j)
           in enumerate(zip(z["period"], z["gi"], z["gj"]))}
    idx = np.array([key[(str(p), int(i), int(j))] for p, i, j
                    in zip(t["period"], t["gi"], t["gj"])])
    dt = z["dt"][idx].astype(np.float32)
    L = (dt > 0).sum(1) + 1
    cv = np.asarray(z["center_valid"], bool)[idx]
    ok = np.array([cv[i, :L[i]].all() for i in range(len(L))])
    ok &= np.asarray(tx["y"] == t["y"])
    return dict(
        feats=torch.from_numpy(z["center"][idx].astype(np.float32)),
        ctx=torch.from_numpy(z["ctx3"][idx].astype(np.float32)),
        valid=torch.from_numpy(np.asarray(z["valid3"])[idx]),
        dt=torch.from_numpy(dt),
        doy=torch.from_numpy(z["doy"][idx].astype(np.float32)),
        spec=torch.from_numpy(t["spec"].astype(np.float32)),
        tex=torch.from_numpy(tx["tex"].astype(np.float32)),
        y=torch.from_numpy(t["y"].astype(np.int64)),
        split=np.array([str(x) for x in t["split"]]),
        per=np.array([str(x) for x in t["period"]]), ok=ok)


def batches(pools, bs, steps, rng):
    for _ in range(steps):
        yield np.concatenate([rng.choice(pools[c], bs // len(pools), replace=True)
                              for c in pools])


def infer(model, d, idx, bs=512, kind="mask"):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            args = [d["feats"][j].to(dev), d["ctx"][j].to(dev), d["valid"][j].to(dev),
                    d["dt"][j].to(dev), d["doy"][j].to(dev)]
            if kind == "type":
                args += [d["tex"][j].to(dev), d["spec"][j].to(dev)]
            outs.append(model(*args).cpu())
    return torch.cat(outs).numpy()


def mask_metrics(logits, y, thr):
    pred = (logits > thr).astype(int)
    return (f1_score(y, pred, zero_division=0), precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))


def train_mask(d, seed=0, epochs=60, steps=125, bs=64, lr=2e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr = np.where((d["split"] == "train") & d["ok"])[0]
    va = np.where((d["split"] == "val") & d["ok"])[0]
    y = d["y"].numpy()
    pools = {c: tr[y[tr] == c] for c in (0, 1)}
    model = MaskModel(768, blocks=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    best = (-1.0, None)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for idx in batches(pools, bs, steps, rng):
            idx = np.asarray(idx)
            logits = model(d["feats"][idx].to(dev), d["ctx"][idx].to(dev),
                           d["valid"][idx].to(dev), d["dt"][idx].to(dev),
                           d["doy"][idx].to(dev))
            loss = F.binary_cross_entropy_with_logits(logits, d["y"][idx].to(dev).float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        if ep % 4 == 0 or ep == epochs - 1:
            lg = infer(model, d, va, kind="mask")
            f1, p, r = mask_metrics(lg, y[va], 0.5)
            if f1 > best[0]:
                best = (f1, {k: v.clone() for k, v in model.state_dict().items()})
            print(f"[mask] ep{ep}: loss={tot / steps:.4f} val F1={f1:.4f} "
                  f"P={p:.4f} R={r:.4f} best={best[0]:.4f}", flush=True)
    model.load_state_dict(best[1])
    lg = infer(model, d, va, kind="mask")
    scan = [(mask_metrics(lg, y[va], t)[0], t) for t in np.arange(0.10, 0.91, 0.01)]
    f1, thr = max(scan)
    res = {"val_best_f1": round(float(f1), 4), "val_thr": round(float(thr), 2)}
    for name, m in (("val", va), ("val_p1p2", va[np.isin(d["per"][va], ["p1", "p2"])]),
                    ("val_p3", va[d["per"][va] == "p3"])):
        lg = infer(model, d, m, kind="mask")
        f, p, r = mask_metrics(lg, y[m], thr)
        res[name] = {"f1": round(float(f), 4), "precision": round(float(p), 4),
                     "recall": round(float(r), 4), "n": int(len(m))}
        print(f"[mask] {name:8s}: F1={f:.4f} P={p:.4f} R={r:.4f} n={len(m)}", flush=True)
    return model, res


def train_type(d, seed=0, epochs=60, steps=125, bs=64, lr=2e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr = np.where((d["split"] == "train") & d["ok"])[0]
    va = np.where((d["split"] == "val") & d["ok"])[0]
    y = d["y"].numpy()
    pools = {c: tr[y[tr] == c] for c in range(3)}
    pools = {c: v for c, v in pools.items() if len(v)}
    model = TypeModel(768, blocks=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    best = (-1.0, None)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for idx in batches(pools, bs, steps, rng):
            idx = np.asarray(idx)
            logits = model(d["feats"][idx].to(dev), d["ctx"][idx].to(dev),
                           d["valid"][idx].to(dev), d["dt"][idx].to(dev),
                           d["doy"][idx].to(dev), d["tex"][idx].to(dev),
                           d["spec"][idx].to(dev))
            loss = F.cross_entropy(logits, d["y"][idx].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        if ep % 4 == 0 or ep == epochs - 1:
            lg = infer(model, d, va, kind="type")
            pred = lg.argmax(1)
            macro = f1_score(y[va], pred, average="macro", zero_division=0)
            acc = float((pred == y[va]).mean())
            if macro > best[0]:
                best = (macro, {k: v.clone() for k, v in model.state_dict().items()})
            print(f"[type] ep{ep}: loss={tot / steps:.4f} val macro={macro:.4f} "
                  f"acc={acc:.4f} best={best[0]:.4f}", flush=True)
    model.load_state_dict(best[1])
    res = {"val_macro": round(float(best[0]), 4)}
    for name, m in (("val", va), ("val_p1p2", va[np.isin(d["per"][va], ["p1", "p2"])]),
                    ("val_p3", va[d["per"][va] == "p3"])):
        lg = infer(model, d, m, kind="type")
        pred = lg.argmax(1)
        yy = y[m]
        res[name] = {"acc": round(float((pred == yy).mean()), 4),
                     "macro": round(float(f1_score(yy, pred, average="macro",
                                                   zero_division=0)), 4),
                     "macro01": round(float(f1_score(yy, pred, labels=[0, 1],
                                                     average="macro", zero_division=0)), 4)}
        res[name]["per_class"] = {
            str(c): {"f1": round(float(f1_score(yy == c, pred == c, zero_division=0)), 4),
                     "precision": round(float(precision_score(yy == c, pred == c,
                                                              zero_division=0)), 4),
                     "recall": round(float(recall_score(yy == c, pred == c,
                                                        zero_division=0)), 4),
                     "n": int((yy == c).sum())} for c in (0, 1, 2)}
        res[name]["confusion"] = confusion_matrix(yy, pred, labels=[0, 1, 2]).tolist()
        print(f"[type] {name:8s}: acc={res[name]['acc']:.4f} macro={res[name]['macro']:.4f} "
              f"macro01={res[name]['macro01']:.4f} per={res[name]['per_class']}", flush=True)
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=125)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    dm = load_mask()
    print(f"[mask] n={len(dm['y'])} train={int(((dm['split'] == 'train') & dm['ok']).sum())} "
          f"val={int(((dm['split'] == 'val') & dm['ok']).sum())}", flush=True)
    mm, mr = train_mask(dm, args.seed, args.epochs, args.steps, args.bs)
    torch.save({"state": mm.state_dict(), "config": {"d_in": 768, "blocks": 1}},
               os.path.join(OUT, "mask.pth"))
    dt = load_type()
    print(f"[type] n={len(dt['y'])} train={int(((dt['split'] == 'train') & dt['ok']).sum())} "
          f"val={int(((dt['split'] == 'val') & dt['ok']).sum())}", flush=True)
    tm, tr = train_type(dt, args.seed, args.epochs, args.steps, args.bs)
    torch.save({"state": tm.state_dict(), "config": {"d_in": 768, "blocks": 2}},
               os.path.join(OUT, "type.pth"))
    res = {"seed": args.seed, "seconds": round(time.time() - t0, 1), "mask": mr, "type": tr,
           "encoder": "CnnDofaLike 768-distilled (typenet_encoder.onnx)",
           "blocks": {"ctx": "CtxAttn 3x3", "mask_cfc": 1, "type_cfc": 2,
                      "readout": "AttnReadout", "width": WIDTH, "hidden": HIDDEN}}
    with open(os.path.join(OUT, f"result_seed{args.seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"[two_models_dofa] saved {OUT} in {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
