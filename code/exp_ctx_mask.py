"""Context experiments for the MASK net on vozh (conv 3x3/5x5 vs raw 3x3/5x5).

Center chip: small-encoder features (256-d, cached grid). Context: small-encoder features
of the k x k neighbourhood per date. Train/eval on vozh only (held-out quarters).
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
from sklearn.metrics import f1_score

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
OUT = os.path.join(ROOT, "outputs", "vozh", "ctx_exp")
sys.path.insert(0, os.path.join(ROOT, "src"))

from cfc import CfCBiEncoderPN  # noqa: E402

DIM = 256


class CtxConv(nn.Module):
    def __init__(self, k, dim=256):
        super().__init__()
        self.k = k
        pad = 0 if k % 2 == 0 else k // 2
        self.conv = nn.Sequential(nn.Conv2d(DIM + 1, dim, k, padding=pad),
                                  nn.GroupNorm(8, dim), nn.GELU())

    def forward(self, ctx, valid):
        ctx = ctx * valid.unsqueeze(-1)                  # ignore empty neighbour patches
        x = torch.cat([ctx, valid.unsqueeze(-1)], dim=-1).permute(0, 3, 1, 2)
        y = self.conv(x)
        return y[:, :, 0, 0] if self.k % 2 == 0 else y[:, :, self.k // 2, self.k // 2]


class CtxRaw(nn.Module):
    def __init__(self, k, dim=256):
        super().__init__()
        self.k = k
        self.slot = nn.Parameter(torch.zeros(k * k, DIM))
        nn.init.normal_(self.slot, std=0.02)
        c = (k - 1) / 2.0
        rr, cc = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        rel = np.stack([(rr - c) / max(c, 1e-6), (cc - c) / max(c, 1e-6)], -1)
        self.register_buffer("rel", torch.tensor(rel.reshape(-1, 2), dtype=torch.float32))
        self.proj = nn.Sequential(nn.Linear(k * k * (DIM + 2), dim), nn.GELU(),
                                  nn.Linear(dim, dim))

    def forward(self, ctx, valid):
        b = ctx.shape[0]
        x = ctx.reshape(b, self.k * self.k, 256) + self.slot.unsqueeze(0)
        x = x * valid.reshape(b, self.k * self.k, 1)
        x = torch.cat([x, self.rel.unsqueeze(0).expand(b, -1, -1)], dim=-1)
        return self.proj(x.reshape(b, -1))


class FeatMaskNet(nn.Module):
    def __init__(self, ctx_mode="none", k=3, cfc_hidden=128, head_width=256, ctx_width=256,
                 ctx_drop=0.2, use_spec=False):
        super().__init__()
        self.ctx_mode = ctx_mode
        self.k = k
        if ctx_mode == "conv":
            self.ctx_net = CtxConv(k, ctx_width)
            d_ctx = ctx_width
        elif ctx_mode == "raw":
            self.ctx_net = CtxRaw(k, ctx_width)
            d_ctx = ctx_width
        else:
            self.ctx_net = None
            d_ctx = 0
        self.ctx_drop = nn.Dropout(ctx_drop) if d_ctx else None
        self.d_ctx = d_ctx
        self.use_spec = use_spec
        self.seq = CfCBiEncoderPN(DIM + d_ctx + 4 + (9 if use_spec else 0), cfc_hidden,
                                  backbone_units=cfc_hidden, backbone_layers=2,
                                  dt_mode="seq")
        h_in = 2 * cfc_hidden + 2 * DIM + (2 * d_ctx if d_ctx else 0) + (18 if use_spec else 0)
        self.head = nn.Sequential(nn.LayerNorm(h_in), nn.Linear(h_in, head_width), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(head_width, 2))
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([1.0, -1.0]))

    def forward(self, center, ctx, valid, dt, doy, lengths=None, spec=None):
        b, t = center.shape[:2]
        if self.d_ctx:
            ctx_vec = self.ctx_net(ctx.reshape(b * t, self.k, self.k, 256),
                                   valid.reshape(b * t, self.k, self.k)).reshape(b, t, -1)
            ctx_vec = self.ctx_drop(ctx_vec)
        else:
            ctx_vec = None
        parts = [center] + ([ctx_vec] if ctx_vec is not None else [])
        if self.use_spec:
            parts.append(spec)
        parts += [doy, dt.unsqueeze(-1)]
        x = torch.cat(parts, dim=-1)
        h = self.seq(x, dt, lengths)
        idx = (dt > 0).sum(dim=1)
        rows = torch.arange(b, device=center.device)
        n = (idx + 1).unsqueeze(1).clamp(min=1)
        outs = [h, center.sum(dim=1) / n, center[rows, idx] - center[:, 0]]
        if ctx_vec is not None:
            outs += [ctx_vec.sum(dim=1) / n, ctx_vec[rows, idx] - ctx_vec[:, 0]]
        if self.use_spec:
            outs += [spec.sum(dim=1) / n, spec[rows, idx] - spec[:, 0]]
        return self.head(torch.cat(outs, dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="none", choices=("none", "conv", "raw"))
    ap.add_argument("--k", type=int, default=3, choices=(3, 5, 6))
    ap.add_argument("--ctx-width", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epoch-sub", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--overfit-gap", type=float, default=0.08,
                    help="stop only on obvious overfit: train_f1 - val_f1 above this")
    ap.add_argument("--min-epochs", type=int, default=30,
                    help="no early stop before this epoch")
    ap.add_argument("--ctx-drop", type=float, default=0.3,
                    help="probability of blanking the whole context (fallback training)")
    ap.add_argument("--spec", action="store_true", help="feed pooled spectral indices (9/date)")
    ap.add_argument("--clean-split", action="store_true",
                    help="drop train/val chips whose k-window crosses the split")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")

    z = np.load(os.path.join(EVAL, "vozh_ctx_mask.npz"))
    center = z["center"].astype(np.float32)
    ctx = z[f"ctx{args.k}"].astype(np.float32)
    valid = z[f"valid{args.k}"]
    dt = z["dt"].astype(np.float32)
    doy = z["doy"].astype(np.float32)
    spec = z["spec"].astype(np.float32)
    y = z["y"]; split = np.array([str(x) for x in z["split"]])
    period = np.array([str(x) for x in z["period"]])
    L_np = ((dt > 0).sum(axis=1) + 1).astype(np.int64)
    tr = split == "train"
    va = split == "val"
    # ignore samples whose central patch is empty (all-zero / no data) on any valid date
    cv = np.asarray(z["center_valid"])
    L = ((dt > 0).sum(axis=1) + 1).astype(np.int64)
    ok = np.ones(len(y), bool)
    for i in range(len(y)):
        if not cv[i, :L[i]].all():
            ok[i] = False
    print(f"[ctx] empty-center chips dropped: train={int((tr & ~ok).sum())} "
          f"val={int((va & ~ok).sum())}", flush=True)
    tr = tr & ok
    va = va & ok
    if args.clean_split:
        import rasterio
        with rasterio.open(os.path.join(ROOT, "outputs", "vozh", "labels",
                                        "quarters.tif")) as ds:
            q = ds.read(1)
        hg, wg = q.shape[0] // 32, q.shape[1] // 32
        qg = q[16::32, 16::32][:hg, :wg].astype(np.int64)
        gi_, gj_ = z["gi"], z["gj"]
        train_q = set(np.unique(qg[gi_[tr], gj_[tr]]).tolist())
        val_q = set(np.unique(qg[gi_[va], gj_[va]]).tolist())
        p_ = args.k // 2
        qp = np.zeros((hg + 2 * p_, wg + 2 * p_), np.int64) - 1
        qp[p_:p_ + hg, p_:p_ + wg] = qg

        def clean(mask, other):
            out = np.zeros(len(mask), bool)
            for i in np.where(mask)[0]:
                r0, c0 = gi_[i] + p_ - p_, gj_[i] + p_ - p_
                cells = np.unique(qp[r0:r0 + args.k, c0:c0 + args.k])
                if not any(int(v) in other for v in cells):
                    out[i] = True
            return out
        tr = tr & clean(tr, val_q)
        va = va & clean(va, train_q)
        print(f"[ctx] clean split: train={int(tr.sum())} val={int(va.sum())}", flush=True)
    pools = {c: np.where(tr & (y == c))[0] for c in (0, 1)}
    print(f"[ctx] {args.ctx}{args.k}: n={len(y)} train neg/pos={len(pools[0])}/{len(pools[1])} "
          f"val={int(va.sum())}", flush=True)

    model = FeatMaskNet(args.ctx, args.k, ctx_width=args.ctx_width,
                        ctx_drop=args.dropout, use_spec=args.spec).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[ctx] params {n_par / 1e6:.2f}M (ctx {model.d_ctx}d, mode {args.ctx}{args.k})",
          flush=True)

    # ---- data-flow audit: shapes + empirical sensitivity to the context
    model.eval()
    with torch.no_grad():
        b, k = 4, args.k
        c0 = torch.randn(b, 4, DIM, device=dev) * 0.3
        x0 = torch.randn(b, 4, k, k, DIM, device=dev) * 0.1
        v0 = torch.ones(b, 4, k, k, device=dev, dtype=torch.bool)
        d0 = torch.zeros(b, 4, device=dev); d0[:, 1] = 0.5
        y0 = torch.zeros(b, 4, 3, device=dev)
        s0 = torch.zeros(b, 4, 9, device=dev)
        o0 = model(c0, x0, v0, d0, y0, None, s0)
        x1 = x0.clone(); x1[:, 0, k // 2 if k % 2 else 0, k // 2 if k % 2 else 0] += 1.0
        o1 = model(c0, x1, v0, d0, y0, None, s0)
        o2 = model(c0, torch.zeros_like(x0), v0, d0, y0, None, s0)
        head_in = 2 * 128 + 2 * DIM + 2 * model.d_ctx
        print(f"[audit] shapes: center(B,4,{DIM}) ctx(B,4,{k},{k},{DIM}) -> ctx_vec(B,4,"
              f"{model.d_ctx}) -> cfcin(B,4,{DIM + model.d_ctx + 4}) -> head_in {head_in} "
              f"-> logits {tuple(o0.shape)}", flush=True)
        print(f"[audit] context sensitivity: center-cell +1 -> |d|={float((o1 - o0).abs().max()):.4f}; "
              f"ctx=0 -> |d|={float((o2 - o0).abs().max()):.4f}", flush=True)
        if args.ctx != "none":
            assert float((o1 - o0).abs().max()) > 1e-6, "context does not reach the output!"
            assert float((o2 - o0).abs().max()) > 1e-6, "zeroing context does not change output!"
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, args.epoch_sub // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

    Ct = torch.from_numpy(center).to(dev)
    Cx = torch.from_numpy(ctx).to(dev)
    Vd = torch.from_numpy(valid).to(dev)
    Dt = torch.from_numpy(dt).to(dev)
    Dy = torch.from_numpy(doy).to(dev)
    Sp = torch.from_numpy(spec).to(dev)

    @torch.no_grad()
    def evaluate():
        model.eval()
        sc = []
        vidx = np.where(va)[0]
        for i0 in range(0, len(vidx), 128):
            j = vidx[i0:i0 + 128]
            Lb = L_np[j]
            Tb = int(Lb.max())
            ii = torch.from_numpy(j).to(dev)
            logits = model(Ct[ii, :Tb], Cx[ii, :Tb], Vd[ii, :Tb], Dt[ii, :Tb], Dy[ii, :Tb],
                           torch.from_numpy(Lb).to(dev), Sp[ii, :Tb])
            sc.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        sc = np.concatenate(sc)
        yv = y[vidx]
        p = (sc > 0).astype(np.int64)
        f1 = f1_score(yv, p, average="binary", zero_division=0)
        prec = float(((p == 1) & (yv == 1)).sum() / max(int(p.sum()), 1))
        rec = float(((p == 1) & (yv == 1)).sum() / max(int(yv.sum()), 1))
        per = {}
        for pid in ("p1", "p2", "p3"):
            m = period[vidx] == pid
            if m.sum():
                per[pid] = round(float(f1_score(yv[m], p[m], average="binary",
                                                zero_division=0)), 3)
        return float(f1), prec, rec, per

    @torch.no_grad()
    def train_f1(n=1024):
        model.eval()
        j = np.random.default_rng(123).choice(np.where(tr)[0], min(n, int(tr.sum())),
                                              replace=False)
        Lb = L_np[j]
        Tb = int(Lb.max())
        ii = torch.from_numpy(j).to(dev)
        lg = model(Ct[ii, :Tb], Cx[ii, :Tb], Vd[ii, :Tb], Dt[ii, :Tb], Dy[ii, :Tb],
                   torch.from_numpy(Lb).to(dev), Sp[ii, :Tb])
        p = (lg[:, 1] > lg[:, 0]).cpu().numpy()
        return float(f1_score(y[ii.cpu().numpy()], p, average="binary", zero_division=0))

    best = {"f1": -1.0, "state": None, "epoch": -1}
    over = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for _ in range(args.epoch_sub // args.bs):
            idx = np.concatenate([pools[0][rng.integers(0, len(pools[0]), args.bs // 2)],
                                  pools[1][rng.integers(0, len(pools[1]), args.bs // 2)]])
            lb = y[idx].astype(np.float32)
            if args.label_smooth > 0:
                lb = lb * (1 - args.label_smooth) + 0.5 * args.label_smooth
            Lb = L_np[idx]
            Tb = int(Lb.max())
            ii = torch.from_numpy(idx).to(dev)
            cx, vd = Cx[ii, :Tb], Vd[ii, :Tb]
            if args.ctx_drop > 0 and model.d_ctx:
                m_ = (torch.rand(len(ii), device=dev) < args.ctx_drop)
                cx = cx * (~m_).reshape(-1, 1, 1, 1, 1)
                vd = vd & (~m_).reshape(-1, 1, 1, 1)
            logits = model(Ct[ii, :Tb], cx, vd, Dt[ii, :Tb], Dy[ii, :Tb],
                           torch.from_numpy(Lb).to(dev), Sp[ii, :Tb])
            loss = F.binary_cross_entropy_with_logits(
                logits[:, 1] - logits[:, 0], torch.from_numpy(lb).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(g):
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot += float(loss.detach()); nb += 1
        if ep % 4 == 0 or ep == args.epochs - 1:
            f1, prec, rec, per = evaluate()
            tf1 = train_f1()
            if f1 > best["f1"]:
                best = {"f1": f1, "epoch": ep,
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}
                torch.save(best["state"], os.path.join(
                    OUT, f"mask_ctx_{args.ctx}{args.k}_{args.tag or 'seed0'}.pth"))
            print(f"[ctx] ep{ep}: {time.time() - t0:.0f}s loss={tot / max(nb, 1):.3f} "
                  f"train_f1={tf1:.3f} val_f1={f1:.3f} P={prec:.3f} R={rec:.3f} "
                  f"best={best['f1']:.3f} {per}", flush=True)
            over = over + 1 if (tf1 - f1) > args.overfit_gap else 0
            if ep >= args.min_epochs and over >= 2:
                print(f"[ctx] early stop: obvious overfit (train_f1={tf1:.3f} vs "
                      f"val_f1={f1:.3f})", flush=True)
                break

    model.load_state_dict(best["state"])
    f1, prec, rec, per = evaluate()
    print(f"\n[ctx] FINAL mask {args.ctx}{args.k}: F1={f1:.3f} P={prec:.3f} R={rec:.3f} "
          f"per={per} (best ep{best['epoch']})", flush=True)
    res = {"ctx": args.ctx, "k": args.k, "val_f1": f1, "precision": prec, "recall": rec,
           "per_period": per, "params": int(n_par), "n_val": int(va.sum()),
           "best_epoch": best["epoch"], "ctx_width": args.ctx_width, "spec": args.spec}
    with open(os.path.join(OUT, f"mask_ctx_{args.ctx}{args.k}_{args.tag or 'seed0'}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
