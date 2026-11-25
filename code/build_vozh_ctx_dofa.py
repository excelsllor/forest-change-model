"""Build the mask context dataset on the ORIGINAL DOFA-B teacher features (768-d).

Center + 3x3 neighbourhood features are extracted from the same mosaics with the
frozen DOFA-B teacher (CLS token, 768-d); validity/labels/splits come from the
small-encoder dataset (identical chips).
"""
import os
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
MOS = os.path.join(ROOT, "outputs", "vozh", "mosaic")
DATES = ("2023-09", "2024-05", "2025-07", "2026-06")
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
CHIP = 32
dev = torch.device("cuda")

z = np.load(os.path.join(EVAL, "vozh_ctx_mask.npz"))
gi, gj = z["gi"], z["gj"]
valid3 = z["valid3"].copy()
n = len(gi)
keys = set()
for i in range(n):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            keys.add((int(gi[i]) + di, int(gj[i]) + dj))
keys = sorted(keys)
idx_of = {k: j for j, k in enumerate(keys)}
U = len(keys)
print("samples", n, "unique chips (with halo)", U, flush=True)

bb = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE).to(dev).eval()
for p in bb.parameters():
    p.requires_grad_(False)
WL = torch.tensor([0.4966, 0.5600, 0.6645, 0.8351, 1.6137, 2.2024], device=dev)
feats = np.zeros((U, len(DATES), 768), np.float16)

for ti, date in enumerate(DATES):
    buf = np.zeros((U, 6, CHIP, CHIP), np.float32)
    for bi, b in enumerate(BANDS):
        with rasterio.open(os.path.join(MOS, f"{date}_{b}.tif")) as ds:
            arr = ds.read(1)
        for j, (r, c) in enumerate(keys):
            r0, c0 = r * CHIP, c * CHIP
            if 0 <= r and 0 <= c and r0 + CHIP <= arr.shape[0] and c0 + CHIP <= arr.shape[1]:
                buf[j, bi] = arr[r0:r0 + CHIP, c0:c0 + CHIP]
    buf = np.clip(buf / 10000.0, 0.0, 1.5)
    with torch.no_grad():
        for i0 in range(0, U, 64):
            x = torch.from_numpy(buf[i0:i0 + 64]).to(dev)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            feats[i0:i0 + 64, ti] = bb.forward_features(x, WL).float().cpu().numpy()
    print(f"[dofa] {date} done", flush=True)

center = np.zeros((n, 4, 768), np.float16)
ctx = np.zeros((n, 4, 3, 3, 768), np.float16)
dt = z["dt"]
L = (dt > 0).sum(1) + 1
for i in range(n):
    li = int(L[i])
    for t in range(li):
        center[i, t] = feats[idx_of[(int(gi[i]), int(gj[i]))], t]
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                key = (int(gi[i]) + dj, int(gj[i]) + di)
                if key in idx_of and valid3[i, t, dj + 1, di + 1]:
                    ctx[i, t, dj + 1, di + 1] = feats[idx_of[key], t]
    if li < 4:
        center[i, li:] = 0
        ctx[i, li:] = 0
        valid3[i, li:] = False

out = {"center": center, "ctx3": ctx, "valid3": valid3,
       "center_valid": z["center_valid"], "dt": z["dt"], "doy": z["doy"],
       "y": z["y"], "split": z["split"], "period": z["period"], "cls": z["cls"],
       "gi": z["gi"], "gj": z["gj"]}
dst = os.path.join(EVAL, "vozh_ctx_mask_dofa.npz")
np.savez(dst, **out)
print(f"[dofa] saved {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)", flush=True)
