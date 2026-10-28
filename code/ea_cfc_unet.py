"""Evolutionary post-processing for the trained CfC U-Net pixel mask.

Tunes threshold + 3x3 opening/closing + minimum component size on val p1+p2;
reports p3 test. Reuses the trained checkpoint and cached probabilities (computed
once here). No network weights are evolved.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "outputs", "vozh", "cfc_unet")
sys.path.insert(0, SRC)
from cfc_unet import CfCUNet, dev, load_cache  # noqa: E402
from ea_postprocess import evolve  # noqa: E402


def infer(model, X, dt, idx, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            xb = torch.from_numpy(X[j].astype(np.float32)).to(dev)
            db = torch.from_numpy(dt[j]).to(dev)
            lb = torch.full((len(j),), X.shape[1], dtype=torch.int64, device=dev)
            out.append(torch.sigmoid(model(xb, db, lb)).cpu().numpy())
    return np.concatenate(out)


def post(p, t, op, cl, min_px):
    pred = p > t
    if op:
        pred = np.asarray([ndimage.binary_opening(x, np.ones((3, 3)), iterations=int(op))
                           for x in pred])
    if cl:
        pred = np.asarray([ndimage.binary_closing(x, np.ones((3, 3)), iterations=int(cl))
                           for x in pred])
    if min_px:
        out = np.zeros_like(pred)
        for i, x in enumerate(pred):
            lb, n = ndimage.label(x)
            if n:
                sz = np.bincount(lb.ravel())
                out[i] = np.isin(lb, np.where(sz >= int(min_px))[0]) & (lb > 0)
        pred = out
    return pred


def metrics(pred, y):
    yy = y.reshape(-1).astype(np.uint8)
    pp = pred.reshape(-1).astype(np.uint8)
    f = f1_score(yy, pp, zero_division=0)
    return {"f1": float(f),
            "precision": float(precision_score(yy, pp, zero_division=0)),
            "recall": float(recall_score(yy, pp, zero_division=0)),
            "iou": float(f / (2 - f) if f > 0 else 0.0),
            "area_ratio": float(pp.sum() / max(yy.sum(), 1))}


def main():
    X, Y, per, split, dt = load_cache()
    va = np.where((split == "val") & np.isin(per, ["p1", "p2"]))[0]
    te = np.where((split == "val") & (per == "p3"))[0]
    model = CfCUNet().to(dev)
    ck = torch.load(os.path.join(OUT, "cfc_unet_seed0.pth"), map_location="cpu")
    model.load_state_dict(ck["state"])
    pv = infer(model, X, dt, va)
    pt = infer(model, X, dt, te)
    yv, yt = Y[va], Y[te]
    np.savez_compressed(os.path.join(OUT, "probs_val_test.npz"),
                        pv=pv, pt=pt, va=va, te=te)
    print(f"[ea] val={len(va)} p3={len(te)}", flush=True)

    def fit(x):
        m = metrics(post(pv, x[0], x[1], x[2], x[3]), yv)
        return m["f1"] - 0.35 * abs(m["area_ratio"] - 1.0)

    best, score = evolve(fit, [0.10, 0, 0, 0], [0.90, 2, 2, 80],
                         pop=20, gens=16, seed=7, ints=(1, 2, 3))
    x = best
    print(f"[ea] best thr={x[0]:.3f} open={int(x[1])} close={int(x[2])} "
          f"min_px={int(x[3])} score={score:.4f}", flush=True)
    res = {"params": {"thr": round(float(x[0]), 3), "open": int(x[1]),
                      "close": int(x[2]), "min_px": int(x[3])},
           "fitness": round(float(score), 4)}
    for name, p, y in (("val", pv, yv), ("p3", pt, yt)):
        m = metrics(post(p, x[0], x[1], x[2], x[3]), y)
        res[name] = m
        print(f"[ea] {name}: {m}", flush=True)
    # fixed comparisons
    for name, p, y in (("val", pv, yv), ("p3", pt, yt)):
        res[f"{name}_base_0.45"] = metrics(post(p, 0.45, 0, 0, 0), y)
    with open(os.path.join(OUT, "ea_result.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
