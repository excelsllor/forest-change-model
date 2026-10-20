"""Stage 1: distill DOFA features into the CnnEncoder on the MAX dataset (60 epochs).

Sources (raw reflectance, teacher = DOFA-B features):
  vozh      6030   T<=4  (type_vozh_raw + type_feats)
  big4     48000   T=2   (DISFOR pairs, 4 classes, G:/datasets/unified/big4_*)
  california 12115 T=2   (burned-area pre/post, G:/datasets/unified/california.npz)
  long      2307   T=8   (DISFOR long series + type_feats)
  eo4w      7853   T=1   (single S2 scenes, G:/datasets/unified/eo4w.npz)

Loss: MSE + cosine on L2-normalised features through a 256->768 projection head.
Channel dropout (p per sample, 1-2 bands zeroed) transfers missing-band robustness.
Saves: outputs/vozh/mask_eval/encoder_dofa_distilled.pth
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
UNI = r"G:\datasets\unified"
sys.path.insert(0, os.path.join(ROOT, "src"))

from masknet_v2 import CnnEncoder, WL6, band_weights  # noqa: E402
from train_masknet_v2 import valid_len  # noqa: E402


def sanitize(v, hi=1.5):
    """Clip saturated/nodata reflectance (DISFOR rasters contain DN up to 65535 = 6.55)."""
    return np.clip(v.astype(np.float32), 0.0, hi).astype(np.float16)


def load_max():
    srcs = []
    zr = np.load(os.path.join(EVAL, "type_vozh_raw.npz"))
    zt = np.load(os.path.join(EVAL, "type_feats.npz"))
    n0 = len(zr["vol"])
    srcs.append({"name": "vozh", "vol": sanitize(zr["vol"]), "feats": zt["feats"][:n0],
                 "dt": zr["dt"].astype(np.float32),
                 "split": np.array([str(x) for x in zr["split"]]), "w": 0.25})
    zb = np.load(os.path.join(UNI, "big4_chips.npz"))
    zf = np.load(os.path.join(UNI, "big4_feats.npz"))
    vol_b = np.stack([zb["X1_tr"], zb["X2_tr"]], 1).astype(np.float32) / 10000.0
    vol_bv = np.stack([zb["X1_va"], zb["X2_va"]], 1).astype(np.float32) / 10000.0
    vol = sanitize(np.concatenate([vol_b, vol_bv]))
    feats = np.concatenate([zf["feats_tr"], zf["feats_va"]])
    split = np.array(["train"] * len(vol_b) + ["val"] * len(vol_bv))
    srcs.append({"name": "big4", "vol": vol, "feats": feats, "dt": None, "split": split,
                 "w": 0.30})
    zc = np.load(os.path.join(UNI, "california.npz"))
    srcs.append({"name": "california", "vol": sanitize(zc["chips"]), "feats": zc["feats"],
                 "dt": None, "split": zc["split"], "w": 0.20})
    zl = np.load(os.path.join(EVAL, "binary_long.npz"))
    n1 = len(np.load(os.path.join(EVAL, "binary_ds.npz"))["dvol"])
    srcs.append({"name": "long", "vol": sanitize(zl["vol"]), "feats": zt["feats"][n0 + n1:],
                 "dt": zl["dt"].astype(np.float32),
                 "split": np.array([str(x) for x in zl["split"]]), "w": 0.10})
    ze = np.load(os.path.join(UNI, "eo4w.npz"))
    srcs.append({"name": "eo4w", "vol": sanitize(ze["chips"])[:, None], "feats": ze["feats"][:, None],
                 "dt": None, "split": np.full(len(ze["chips"]), "train"), "w": 0.15})
    for s in srcs:
        print(f"[distill] {s['name']}: n={len(s['vol'])} train={int((s['split'] == 'train').sum())}",
              flush=True)
    return srcs


def dates_of(s, i):
    """Date indices: long uses 8 evenly spaced (teacher constraint), others full valid."""
    if s["name"] == "long":
        T = s["vol"].shape[1]
        return np.linspace(0, T - 1, min(8, T)).round().astype(int)
    if s["dt"] is None:
        return np.arange(s["vol"].shape[1])
    return np.arange(valid_len(s["dt"][i]))


def make_batch(s, idxs, rng, chdrop_p):
    didx = [dates_of(s, i) for i in idxs]
    Ls = [len(x) for x in didx]
    T, B = max(Ls), len(idxs)
    Vb = np.zeros((B, T, 6, 32, 32), np.float32)
    Fb = np.zeros((B, T, 768), np.float32)
    Mb = np.zeros((B, T), bool)
    for j, (i, di) in enumerate(zip(idxs, didx)):
        L = Ls[j]
        v = s["vol"][i][di].astype(np.float32)
        if chdrop_p and rng.random() < chdrop_p:
            k = 1 if rng.random() < 0.7 else 2
            for c in rng.choice(6, size=k, replace=False):
                v[:, c] = 0.0
        Vb[j, :L] = v
        Fb[j, :L] = s["feats"][i, :L].astype(np.float32)
        Mb[j, :L] = True
    return Vb, Fb, Mb


def mean_cos(enc, proj, srcs, dev, wl, cap=256, chdrop_p=0.0, seed=0):
    enc.eval()
    rng = np.random.default_rng(seed)
    per = {}
    with torch.no_grad():
        for s in srcs:
            vi = np.where(s["split"] == "val")[0]
            if len(vi) == 0:
                continue
            if len(vi) > cap:
                vi = rng.choice(vi, cap, replace=False)
            tot, n = 0.0, 0
            for i0 in range(0, len(vi), 64):
                Vb, Fb, Mb = make_batch(s, [int(x) for x in vi[i0:i0 + 64]], rng, chdrop_p)
                mb = torch.from_numpy(Mb).to(dev)
                out = proj(enc(torch.from_numpy(Vb).to(dev), wl))
                a = F.normalize(out[mb], dim=-1)
                b = F.normalize(torch.from_numpy(Fb).to(dev)[mb], dim=-1)
                tot += float((a * b).sum(-1).sum())
                n += int(Mb.sum())
            per[s["name"]] = tot / max(n, 1)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epoch-sub", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chdrop-p", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="small", choices=["small", "big"],
                    help="small = CnnEncoder (4 stages, dilations); big = CnnDofaLike "
                         "(12 layers x 768, dilations) - RF covers the whole 32x32 chip")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
        args.epoch_sub = 256

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    srcs = load_max()
    pools = []
    for s in srcs:
        pools.append(np.where(s["split"] == "train")[0])
    w = np.array([s["w"] for s in srcs])
    w = w / w.sum()

    enc = CnnEncoder(256).to(dev)
    proj = nn.Sequential(nn.Linear(256, 768)).to(dev)
    out_name = "encoder_dofa_distilled.pth"
    if args.arch == "big":
        from typenet_v3 import CnnDofaLike
        enc = CnnDofaLike(768, layers=12).to(dev)
        proj = nn.Identity().to(dev)
        out_name = "encoder_dofa_distilled_big.pth"
    n_par = sum(p.numel() for p in enc.parameters())
    print(f"[distill] encoder {args.arch} (CNN, dilated RF): {n_par / 1e6:.2f}M params",
          flush=True)
    wl = band_weights(WL6).to(dev)
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, args.epoch_sub // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

    t0 = time.time()
    for ep in range(args.epochs):
        enc.train()
        tot, nb = 0.0, 0
        for _ in range(args.epoch_sub // args.bs):
            si = int(rng.choice(len(srcs), p=w))
            s, pool = srcs[si], pools[si]
            idxs = [int(pool[rng.integers(len(pool))]) for _ in range(args.bs)]
            Vb, Fb, Mb = make_batch(s, idxs, rng, args.chdrop_p)
            out = proj(enc(torch.from_numpy(Vb).to(dev), wl))
            mb = torch.from_numpy(Mb).to(dev)
            a = F.normalize(out[mb], dim=-1)
            b = F.normalize(torch.from_numpy(Fb).to(dev)[mb], dim=-1)
            loss = F.mse_loss(a, b) + (1.0 - (a * b).sum(-1).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        if ep % 5 == 4 or ep == args.epochs - 1 or ep == 0:
            full = mean_cos(enc, proj, srcs, dev, wl, chdrop_p=0.0)
            drop = mean_cos(enc, proj, srcs, dev, wl, chdrop_p=1.0, seed=1)
            favg = np.mean(list(full.values()))
            davg = np.mean(list(drop.values()))
            print(f"[distill] ep{ep}: {time.time() - t0:.0f}s loss={tot / max(nb, 1):.4f} "
                  f"cos(full)={favg:.3f} cos(drop)={davg:.3f} | " +
                  " ".join(f"{k}:{v:.2f}" for k, v in full.items()), flush=True)
        else:
            print(f"[distill] ep{ep}: {time.time() - t0:.0f}s loss={tot / max(nb, 1):.4f}",
                  flush=True)
    enc.eval()
    torch.save(enc.state_dict(), os.path.join(EVAL, out_name))
    print(f"[distill] saved {out_name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
