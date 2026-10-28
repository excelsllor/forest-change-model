"""EA + fusion for the CfC U-Net pixel mask (p3-trained checkpoint).

EA tunes threshold/opening/closing/min_px on p1+p2 val, reports p3.
Fusion on p3: union / intersection with spectral dNDVI/dNBR, chip-mask gate, weighted.
"""
import json
import os
import sys

import numpy as np
import rasterio
import torch
from scipy import ndimage
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "outputs", "vozh", "cfc_unet")
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
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
    return {"f1": round(float(f), 4),
            "precision": round(float(precision_score(yy, pp, zero_division=0)), 4),
            "recall": round(float(recall_score(yy, pp, zero_division=0)), 4),
            "iou": round(float(f / (2 - f) if f > 0 else 0.0), 4),
            "area_ha": round(float(pp.sum() * 0.01), 1),
            "area_ratio": round(float(pp.sum() / max(yy.sum(), 1)), 3)}


def main():
    X, Y, per, split, dt = load_cache()
    va = np.where((split == "val") & np.isin(per, ["p1", "p2"]))[0]
    te = np.where((split == "val") & (per == "p3"))[0]
    model = CfCUNet().to(dev)
    ck = torch.load(os.path.join(OUT, "cfc_unet_seed0_p3.pth"), map_location="cpu")
    model.load_state_dict(ck["state"])
    pv = infer(model, X, dt, va)
    pt = infer(model, X, dt, te)
    yv, yt = Y[va], Y[te]
    print(f"[ea] val={len(va)} p3={len(te)} gt_p3={yt.sum()*0.01:.1f} ha", flush=True)

    def fit(x):
        m = metrics(post(pv, x[0], x[1], x[2], x[3]), yv)
        return m["f1"] - 0.35 * abs(m["area_ratio"] - 1.0)

    x, score = evolve(fit, [0.10, 0, 0, 0], [0.90, 2, 2, 80],
                      pop=20, gens=16, seed=7, ints=(1, 2, 3))
    print(f"[ea] best thr={x[0]:.3f} op={int(x[1])} cl={int(x[2])} "
          f"min_px={int(x[3])} score={score:.4f}", flush=True)
    res = {"params": {"thr": round(float(x[0]), 3), "open": int(x[1]),
                      "close": int(x[2]), "min_px": int(x[3])},
           "fitness": round(float(score), 4)}
    u_val = post(pv, x[0], x[1], x[2], x[3])
    u_p3 = post(pt, x[0], x[1], x[2], x[3])
    res["val"] = metrics(u_val, yv)
    res["p3"] = metrics(u_p3, yt)

    # spectral + fusion on p3
    dndvi = X[te, -1, 6] - X[te, 0, 6]
    dnbr = X[te, 0, 7] - X[te, -1, 7]
    spec = (dndvi < -0.10) | (dnbr > 0.05)
    res["p3_spectral"] = metrics(spec, yt)
    res["p3_union"] = metrics(u_p3 | spec, yt)
    res["p3_intersection"] = metrics(u_p3 & spec, yt)
    # chip mask gate on the same p3 test chips
    z = np.load(os.path.join(EVAL, "vozh_ctx_mask.npz"), allow_pickle=True)
    gi = z["gi"].astype(int)
    gj = z["gj"].astype(int)
    with rasterio.open(os.path.join(ROOT, "outputs", "vozh", "full_test_mask", "mask.tif")) as ds:
        cm = ds.read(1)
    gate = np.empty_like(u_p3)
    for k, i in enumerate(te):
        r, c = gi[i] * 32, gj[i] * 32
        gate[k] = cm[r:r + 32, c:c + 32] == 1
    res["p3_gate"] = metrics(u_p3 & gate, yt)
    res["p3_union_gate"] = metrics((u_p3 | spec) & gate, yt)
    # weighted probability + spectral
    for w in (0.3, 0.5, 0.7):
        res[f"p3_weighted_{w}"] = metrics((pt + w * spec) > 0.5, yt)
    with open(os.path.join(OUT, "ea_fusion_result.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
