"""Extract per-date features of the DOFA-distilled big CNN encoder (12x768) for the TYPE task.

Sources (changed samples only, 3 classes: logging=0, windfall=1, fire=2):
  vozh   cls in 1..3          T<=4   (type_vozh_raw)
  big4   y in 0..2 (drop deg) T=2    (G:/datasets/unified/big4_chips.npz + dates)
  cal    y == 1 -> fire       T=2    (G:/datasets/unified/california.npz)
  pairs  dcls in 1..3         T=2    (binary_ds.npz)
  long   lcls in 1..3         T<=36  (binary_long.npz; val excludes big4-train territories)

Saves outputs/vozh/mask_eval/tn_{vozh,big4,cal,pairs,long}.npz
  feats (n,T,768) f16, y (n,), dt (n,T), doy (n,T,3), split (n,)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
UNI = r"G:\datasets\unified"
sys.path.insert(0, os.path.join(ROOT, "src"))

from typenet_v3 import CnnDofaLike, WL6, band_weights  # noqa: E402

CANON = {"2023-09": 258, "2024-05": 135, "2025-07": 196, "2026-06": 166}
SEQD = ("2023-09", "2024-05", "2025-07", "2026-06")
PLEN = {"p1": 2, "p2": 3, "p3": 4}


def sanitize(v, hi=1.5):
    return np.clip(v.astype(np.float32), 0.0, hi)


def check_reflectance(name, v, hi=7.0):
    """Guard against a missing /10000: raw DN (0..65535) would blow up here.

    All volumes must already be reflectance/10000 (max ~6.55 for saturated DISFOR).
    """
    mx = float(np.max(v))
    if mx > hi:
        raise ValueError(f"[tnf] {name}: reflectance max {mx:.1f} > {hi} - missing /10000?")
    if mx > 2.0:
        print(f"[tnf] {name}: saturated pixels present (max {mx:.2f})", flush=True)


def doy3(d):
    d = pd.Timestamp(str(d))
    t = d.dayofyear
    return np.array([np.sin(2 * np.pi * t / 365.0), np.cos(2 * np.pi * t / 365.0),
                     d.year - 2015.0], np.float32)


def feats9(v):
    """Pooled 9 spectral features from a (6,32,32) reflectance chip (bands + indices)."""
    b02, b03, b04, b08, b11, b12 = v
    ndvi = (b08 - b04) / np.maximum(b08 + b04, 1e-6)
    nbr = (b08 - b12) / np.maximum(b08 + b12, 1e-6)
    ndmi = (b08 - b11) / np.maximum(b08 + b11, 1e-6)
    return np.array([b02.mean(), b03.mean(), b04.mean(), b08.mean(), b11.mean(), b12.mean(),
                     ndvi.mean(), nbr.mean(), ndmi.mean()], np.float32)


def build_sources():
    srcs = []
    t0 = time.time()
    print("[tnf] building sources...", flush=True)
    zr = np.load(os.path.join(EVAL, "type_vozh_raw.npz"))
    print(f"[tnf] loaded type_vozh_raw ({time.time() - t0:.0f}s)", flush=True)
    cls = zr["cls"]
    vol_all = zr["vol"]                    # single decompression (NpzFile does not cache!)
    dt_all = zr["dt"]
    period_all = zr["period"]
    split_all = zr["split"]
    keep = np.where((cls >= 1) & (cls <= 3))[0]
    check_reflectance("vozh", vol_all[keep])
    vol = np.zeros((len(keep), 4, 6, 32, 32), np.float32)
    dt = np.zeros((len(keep), 4), np.float32)
    doy = np.zeros((len(keep), 4, 3), np.float32)
    for j, i in enumerate(keep):
        L = PLEN[period_all[i]]
        vol[j, :L] = sanitize(vol_all[i, :L])
        dt[j, :L] = dt_all[i, :L]
        for t in range(L):
            a = 2 * np.pi * CANON[SEQD[t]] / 365.0
            doy[j, t] = [np.sin(a), np.cos(a), 0.0]
    srcs.append({"name": "vozh", "vol": vol, "dt": dt, "doy": doy,
                 "y": (cls[keep] - 1).astype(np.int64),
                 "split": np.array([str(split_all[i]) for i in keep])})
    print(f"[tnf] vozh built ({time.time() - t0:.0f}s)", flush=True)

    zb = np.load(os.path.join(UNI, "big4_chips.npz"))
    print(f"[tnf] loaded big4_chips ({time.time() - t0:.0f}s)", flush=True)
    y_all = np.concatenate([zb["y_tr"], zb["y_va"]])
    keep = np.where(y_all < 3)[0]
    vol = np.concatenate([np.stack([zb["X1_tr"], zb["X2_tr"]], 1),
                          np.stack([zb["X1_va"], zb["X2_va"]], 1)])[keep] / 10000.0
    check_reflectance("big4", vol)
    pre = np.concatenate([zb["pre_tr"], zb["pre_va"]])[keep]
    post = np.concatenate([zb["post_tr"], zb["post_va"]])[keep]
    print(f"[tnf] big4: {len(pre)} pairs, computing dates...", flush=True)
    pre_d = pd.to_datetime([str(x) for x in pre])
    post_d = pd.to_datetime([str(x) for x in post])
    gaps = np.maximum(np.array([(b - a).days for a, b in zip(post_d, pre_d)]), 1) / 365.25
    dt = np.stack([np.zeros_like(gaps, dtype=np.float32), gaps.astype(np.float32)], 1)
    uniq = {str(d) for d in list(pre) + list(post)}
    enc = {d: doy3(d) for d in sorted(uniq)}
    doy = np.array([[enc[str(a)], enc[str(b)]] for a, b in zip(pre, post)], np.float32)
    split = np.array(["train"] * len(zb["X1_tr"]) + ["val"] * len(zb["X1_va"]))[keep]
    srcs.append({"name": "big4", "vol": sanitize(vol), "dt": dt, "doy": doy,
                 "y": y_all[keep].astype(np.int64), "split": split})
    print(f"[tnf] big4 built ({time.time() - t0:.0f}s)", flush=True)

    zc = np.load(os.path.join(UNI, "california.npz"))
    print(f"[tnf] loaded california ({time.time() - t0:.0f}s)", flush=True)
    keep = np.where(zc["y"] == 1)[0]
    check_reflectance("cal", zc["chips"][keep])
    srcs.append({"name": "cal", "vol": sanitize(zc["chips"][keep]), "dt": zc["dt"][keep],
                 "doy": zc["doy"][keep], "y": np.full(len(keep), 2, np.int64),
                 "split": zc["split"][keep]})
    print(f"[tnf] cal built ({time.time() - t0:.0f}s)", flush=True)

    z = np.load(os.path.join(EVAL, "binary_ds.npz"))
    print(f"[tnf] loaded binary_ds ({time.time() - t0:.0f}s)", flush=True)
    keep = np.where((z["dcls"] >= 1) & (z["dcls"] <= 3))[0]
    check_reflectance("pairs", z["dvol"][keep])
    srcs.append({"name": "pairs", "vol": sanitize(z["dvol"][keep]), "dt": z["ddt"][keep],
                 "doy": z["ddoy"][keep], "y": (z["dcls"][keep] - 1).astype(np.int64),
                 "split": np.array([str(x) for x in z["dsplit"]])[keep]})
    print(f"[tnf] pairs built ({time.time() - t0:.0f}s)", flush=True)

    zl = np.load(os.path.join(EVAL, "binary_long.npz"))
    print(f"[tnf] loaded binary_long ({time.time() - t0:.0f}s)", flush=True)
    keep = np.where((zl["cls"] >= 1) & (zl["cls"] <= 3))[0]
    check_reflectance("long", zl["vol"][keep])
    l_split = np.array([str(x) for x in zl["split"]])
    b4_train = set(int(x) for x in zb["sid_tr"])
    l_ok = np.array([(s == "val") and (int(sid) not in b4_train)
                     for s, sid in zip(l_split, zl["sid"])])
    l2 = np.where(l_split == "train", "train", np.where(l_ok, "val", "skip"))
    srcs.append({"name": "long", "vol": sanitize(zl["vol"][keep]), "dt": zl["dt"][keep],
                 "doy": zl["doy"][keep], "y": (zl["cls"][keep] - 1).astype(np.int64),
                 "split": l2[keep]})
    print(f"[tnf] long built ({time.time() - t0:.0f}s)", flush=True)
    for s in srcs:
        print(f"[tnf] {s['name']}: n={len(s['y'])} train={int((s['split'] == 'train').sum())} "
              f"val={int((s['split'] == 'val').sum())} classes={np.bincount(s['y'], minlength=3).tolist()}",
              flush=True)
    return srcs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of sources (vozh,big4,cal,pairs,long)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = CnnDofaLike(768, layers=12).to(dev)
    enc.load_state_dict(torch.load(os.path.join(EVAL, "encoder_dofa_distilled_big.pth"),
                                   map_location="cpu"))
    enc.eval()
    print(f"[tnf] encoder: {sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M params",
          flush=True)
    wl = band_weights(WL6).to(dev)

    srcs = build_sources()
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        srcs = [s for s in srcs if s["name"] in want]
        print(f"[tnf] extracting only: {[s['name'] for s in srcs]}", flush=True)
    t0 = time.time()
    aug = os.environ.get("AUG", "0") == "1"
    variants = [("id", lambda v: v), ("r90", lambda v: np.rot90(v, 1, (3, 4))),
                ("r180", lambda v: np.rot90(v, 2, (3, 4))),
                ("r270", lambda v: np.rot90(v, 3, (3, 4))),
                ("hf", lambda v: v[:, :, :, ::-1, :]), ("vf", lambda v: v[:, :, :, :, ::-1])]
    AUG_SRC = {"vozh", "pairs", "long"}
    for s in srcs:
        use_v = variants if (aug and s["name"] in AUG_SRC) else variants[:1]
        print(f"[tnf] extracting {s['name']}: {len(s['y'])} samples x {len(use_v)} variants...",
              flush=True)
        Fs, Ss, Ys, Ds, DOs, SPs, Vs = [], [], [], [], [], [], []
        for vi, (vname, tr) in enumerate(use_v):
            vol = np.ascontiguousarray(tr(s["vol"]))
            n, T = vol.shape[0], vol.shape[1]
            feats = np.zeros((n, T, 768), np.float16)
            chips = []
            pos = []
            for i in range(n):
                L = T if s["name"] == "long" else int(np.nonzero(s["dt"][i])[0][-1]) + 1
                for t in range(L):
                    chips.append(vol[i, t])
                    pos.append((i, t))
            chips = np.stack(chips).astype(np.float32)
            spec_all = np.stack([feats9(c) for c in chips]).astype(np.float16)
            spec = np.zeros((n, T, 9), np.float16)
            for k, (ii, tt) in enumerate(pos):
                spec[ii, tt] = spec_all[k]
            for i0 in range(0, len(chips), 128):
                x = torch.from_numpy(chips[i0:i0 + 128]).to(dev)
                with torch.no_grad():
                    f = enc(x[:, None], wl)[:, 0]                  # (B,768)
                for k in range(f.shape[0]):
                    ii, tt = pos[i0 + k]
                    feats[ii, tt] = f[k].cpu().numpy().astype(np.float16)
                if (i0 // 128) % 40 == 0:
                    el = time.time() - t0
                    frac = (i0 + 128) / max(len(chips), 1)
                    eta = el / max(frac, 1e-6) * (1 - frac) / 60
                    print(f"[tnf]   {s['name']}/{vname} {i0}/{len(chips)} chips | {el:.0f}s | "
                          f"ETA {eta:.0f} min", flush=True)
            Fs.append(feats)
            Ss.append(spec)
            Ys.append(s["y"])
            Ds.append(s["dt"])
            DOs.append(s["doy"])
            SPs.append(s["split"])
            Vs.append(np.full(len(s["y"]), vi))
        feats = np.concatenate(Fs)
        spec = np.concatenate(Ss)
        y = np.concatenate(Ys)
        dt = np.concatenate(Ds)
        doy = np.concatenate(DOs)
        split = np.concatenate(SPs)
        vid = np.concatenate(Vs)
        keep = (split == "train") | (vid == 0)          # val stays un-augmented
        np.savez_compressed(os.path.join(EVAL, f"tn_{s['name']}.npz"), feats=feats[keep],
                            spec=spec[keep], y=y[keep], dt=dt[keep], doy=doy[keep],
                            split=split[keep])
        print(f"[tnf] {s['name']}: feats {feats[keep].shape} saved ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
