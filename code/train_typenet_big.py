"""TypeNet training on cached features of the DOFA-distilled big CNN encoder (12x768, dilated RF).

Data: tn_{vozh,big4,cal,pairs,long}.npz (feats (n,T,768) f16, y in {0,1,2}, dt, doy, split)
Model: TypeNetFeats - CfCBiStack(blocks=2) + head (3 classes: logging/windfall/fire).
Balanced sampler over classes, source-stratified. 60 epochs, best by val macro-F1.
Run: python train_typenet_big.py --width 512
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
sys.path.insert(0, os.path.join(ROOT, "src"))

from typenet_big import TypeNetFeats  # noqa: E402
from train_masknet_v2 import sample_subset, valid_len  # noqa: E402

NAMES = ("vozh", "big4", "cal", "pairs", "long")
W_CLS = {
    0: np.array([0.45, 0.35, 0.00, 0.10, 0.10]),   # logging
    1: np.array([0.60, 0.20, 0.00, 0.10, 0.10]),   # windfall (vozh boosted: only 291 train)
    2: np.array([0.05, 0.45, 0.35, 0.05, 0.10]),   # fire
}
FIXED_T = {"big4": 2, "cal": 2, "pairs": 2}


def load_all():
    srcs = []
    for name in NAMES:
        z = np.load(os.path.join(EVAL, f"tn_{name}.npz"))
        srcs.append({"name": name, "feats": z["feats"], "spec": z["spec"], "y": z["y"],
                     "dt": z["dt"].astype(np.float32), "doy": z["doy"].astype(np.float32),
                     "split": np.array([str(x) for x in z["split"]])})
        print(f"[tnt] {name}: n={len(z['y'])} train={int((srcs[-1]['split'] == 'train').sum())} "
              f"val={int((srcs[-1]['split'] == 'val').sum())} "
              f"classes={np.bincount(z['y'], minlength=3).tolist()}", flush=True)
    return srcs


def build_batch(srcs, chunk, rng):
    subs = []
    for s_i, i in chunk:
        s = srcs[s_i]
        if s["name"] in FIXED_T:
            subs.append([0, 1])
        else:
            subs.append(sample_subset(valid_len(s["dt"][i]), rng))
    T = max(len(x) for x in subs)
    Fb = np.zeros((len(chunk), T, 768), np.float32)
    Db = np.zeros((len(chunk), T), np.float32)
    Yb = np.zeros((len(chunk), T, 3), np.float32)
    Sb = np.zeros((len(chunk), T, 9), np.float32)
    for j, ((s_i, i), sub) in enumerate(zip(chunk, subs)):
        s = srcs[s_i]
        Fb[j, :len(sub)] = s["feats"][i][sub].astype(np.float32)
        Sb[j, :len(sub)] = s["spec"][i][sub].astype(np.float32)
        Yb[j, :len(sub)] = s["doy"][i][sub]
        d = np.zeros(len(sub), np.float32)
        for k in range(1, len(sub)):
            d[k] = s["dt"][i, sub[k - 1] + 1:sub[k] + 1].sum()
        Db[j, :len(sub)] = d
    return Fb, Db, Yb, Sb


def predict(model, srcs, pairs, dev):
    model.eval()
    out = np.zeros(len(pairs), np.int64)
    with torch.no_grad():
        for i0 in range(0, len(pairs), 128):
            chunk = pairs[i0:i0 + 128]
            Fb, Db, Yb, Sb = build_batch(srcs, chunk, np.random.default_rng(0))
            logits = model(torch.from_numpy(Fb).to(dev), torch.from_numpy(Db).to(dev),
                           torch.from_numpy(Yb).to(dev), torch.from_numpy(Sb).to(dev))
            out[i0:i0 + len(chunk)] = logits.argmax(1).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--head-width", type=int, default=64)
    ap.add_argument("--head-mode", default="plain", choices=("plain", "summary"))
    ap.add_argument("--readout", default="cfc", choices=("cfc", "attn"))
    ap.add_argument("--spec", action="store_true", help="feed pooled spectral indices (9/date)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epoch-sub", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
        args.epoch_sub = 192
        args.val_every = 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    srcs = load_all()
    pools = {}
    for s_i, s in enumerate(srcs):
        for c in range(3):
            pools[(s_i, c)] = np.where((s["split"] == "train") & (s["y"] == c))[0]
    print("[tnt] train pools: " + " | ".join(
        f"{s['name']}c{c}={len(pools[(i, c)])}" for i, s in enumerate(srcs) for c in range(3)),
        flush=True)

    model = TypeNetFeats(cfc_width=args.width, blocks=args.blocks,
                         head_width=args.head_width, head_mode=args.head_mode,
                         use_spec=args.spec, readout=args.readout).to(dev)
    n_seq = sum(p.numel() for p in model.parameters())
    print(f"[tnt] width={args.width} blocks={args.blocks} readout={args.readout} "
          f"head={args.head_mode}/{args.head_width} | seq+head {n_seq / 1e6:.2f}M | "
          f"encoder (frozen, cached) 7.58M | total {7.58 + n_seq / 1e6:.2f}M", flush=True)

    val_pairs = [(s, int(i)) for s in range(len(srcs))
                 for i in np.where(srcs[s]["split"] == "val")[0]]
    vy = np.array([srcs[s]["y"][i] for s, i in val_pairs], np.int64)
    print(f"[tnt] val={len(val_pairs)} classes={np.bincount(vy, minlength=3).tolist()}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, args.epoch_sub // args.bs)
    warm = min(args.warmup, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps - warm))],
        milestones=[warm])

    def draw(n, c):
        w = W_CLS[c].copy()
        w = w / w.sum()
        out = []
        for _ in range(n):
            s_i = int(rng.choice(len(srcs), p=w))
            pool = pools[(s_i, c)]
            if len(pool) == 0:
                s_i = next(ss for ss in range(len(srcs)) if len(pools[(ss, c)]) > 0)
                pool = pools[(s_i, c)]
            out.append((s_i, int(pool[rng.integers(len(pool))])))
        return out

    best = {"f1": -1.0, "state": None, "epoch": -1}
    t_train = time.time()
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for _ in range(args.epoch_sub // args.bs):
            chunk = draw(args.bs // 3 + 1, 0) + draw(args.bs // 3, 1) + draw(args.bs // 3, 2)
            chunk = chunk[:args.bs]
            Fb, Db, Yb, Sb = build_batch(srcs, chunk, rng)
            yb = np.array([srcs[s]["y"][i] for s, i in chunk], np.int64)
            logits = model(torch.from_numpy(Fb).to(dev), torch.from_numpy(Db).to(dev),
                           torch.from_numpy(Yb).to(dev), torch.from_numpy(Sb).to(dev))
            loss = F.cross_entropy(logits, torch.from_numpy(yb).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gnorm):
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if torch.isfinite(loss):
                tot += float(loss.detach())
            nb += 1
        ep_time = time.time() - t0
        if ep % args.val_every == 0 or ep == args.epochs - 1:
            preds = predict(model, srcs, val_pairs, dev)
            present = np.unique(vy)
            f1 = float(f1_score(vy, preds, average="macro", labels=present, zero_division=0))
            if f1 > best["f1"]:
                best = {"f1": f1, "epoch": ep,
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}
                torch.save(best["state"], os.path.join(
                    EVAL, f"typenet_big_w{args.width}_b{args.blocks}{args.tag}.pth"))
            per = {srcs[s]["name"]: round(float(f1_score(
                vy[m], preds[m], average="macro", labels=np.unique(vy[m]), zero_division=0)), 3)
                for s in range(len(srcs))
                for m in [np.array([p[0] == s for p in val_pairs])] if m.sum()}
            per_cls = [round(float(f1_score(vy, preds, average="macro", labels=[c],
                                           zero_division=0)), 3) for c in range(3)]
            print(f"[tnt] ep{ep}: {ep_time:.0f}s loss={tot / max(nb, 1):.3f} val_mF1={f1:.3f} "
                  f"best={best['f1']:.3f} classes={per_cls} {per}", flush=True)
        else:
            print(f"[tnt] ep{ep}: {ep_time:.0f}s loss={tot / max(nb, 1):.3f}", flush=True)
    print(f"[tnt] width {args.width}: training {((time.time() - t_train) / 60):.1f} min", flush=True)

    model.load_state_dict(best["state"])
    preds = predict(model, srcs, val_pairs, dev)
    res = {"width": args.width, "blocks": args.blocks, "best_val_macro_f1": best["f1"],
           "best_epoch": best["epoch"], "seq_head_params": int(n_seq),
           "encoder_params": 7580000, "epochs": args.epochs,
           "head_width": args.head_width, "head_mode": args.head_mode, "spec": args.spec,
           "readout": args.readout}
    for s in range(len(srcs)):
        m = np.array([p[0] == s for p in val_pairs])
        if m.sum():
            res[srcs[s]["name"]] = {"n": int(m.sum()),
                                    "macro_f1": float(f1_score(vy[m], preds[m], average="macro",
                                                               labels=np.unique(vy[m]),
                                                               zero_division=0))}
    with open(os.path.join(EVAL, f"typenet_big_w{args.width}_b{args.blocks}{args.tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"[tnt] FINAL width={args.width} blocks={args.blocks}: val macro-F1="
          f"{best['f1']:.3f} (seq+head {n_seq / 1e6:.2f}M, encoder 7.58M)", flush=True)


if __name__ == "__main__":
    main()
