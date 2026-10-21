"""Per-chip pixel-class fractions for vozh val windfall: correct vs missed."""
import os
import sys

import numpy as np
import rasterio
import torch

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
sys.path.insert(0, os.path.join(ROOT, "src"))

from typenet_v3 import CnnDofaLike, WL6, band_weights  # noqa: E402
from typenet_big import TypeNetFeats  # noqa: E402

CANON = {"2023-09": 258, "2024-05": 135, "2025-07": 196, "2026-06": 166}
SEQ = ("2023-09", "2024-05", "2025-07", "2026-06")
dev = torch.device("cuda")

z = np.load(os.path.join(EVAL, "type_vozh_raw.npz"))
gi = np.load(os.path.join(EVAL, "vozh_ctx_mask.npz"))["gi"]
gj = np.load(os.path.join(EVAL, "vozh_ctx_mask.npz"))["gj"]
cls = z["cls"].astype(np.int64)
split = np.array([str(x) for x in z["split"]])
period = np.array([str(x) for x in z["period"]])
dt = z["dt"].astype(np.float32)
vol = z["vol"].astype(np.float32)
idx = np.where(split == "val")[0]
L = (dt[idx] > 0).sum(1) + 1

enc = CnnDofaLike(768, layers=12).to(dev)
enc.load_state_dict(torch.load(os.path.join(EVAL, "encoder_dofa_distilled_big.pth"),
                               map_location="cpu"))
enc.eval()
wl = band_weights(WL6).to(dev)
feats = np.zeros((len(idx), 4, 768), np.float32)
with torch.no_grad():
    for i in range(len(idx)):
        v = torch.from_numpy(vol[idx[i], :L[i]]).to(dev)
        feats[i, :L[i]] = enc(v[None], wl)[0].cpu().numpy()
doy = np.zeros((len(idx), 4, 3), np.float32)
for i in range(len(idx)):
    for t in range(L[i]):
        a = 2 * np.pi * CANON[SEQ[t]] / 365.0
        doy[i, t] = [np.sin(a), np.cos(a), 0.0]
model = TypeNetFeats(cfc_width=256, blocks=2, head_width=256, head_mode="summary",
                     readout="attn").to(dev)
model.load_state_dict(torch.load(os.path.join(EVAL, "typenet_big_w256_b2_attn.pth"),
                                 map_location="cpu"))
model.eval()
with torch.no_grad():
    pred = model(torch.from_numpy(feats).to(dev), torch.from_numpy(dt[idx]).to(dev),
                 torch.from_numpy(doy).to(dev)).argmax(1).cpu().numpy()
gt = cls[idx] - 1                      # 0 logging, 1 windfall, 2 fire
keep = cls[idx] >= 1                   # changed chips only

# pixel-class fractions per chip from the period label rasters
rasters = {}
for p in ("p1", "p2", "p3"):
    with rasterio.open(os.path.join(ROOT, "outputs", "vozh", "labels", f"classes_{p}.tif")) as ds:
        rasters[p] = ds.read(1)

frac = np.zeros((len(idx), 4), np.float32)  # logging, windfall, fire, total changed
for j, i in enumerate(idx):
    r0, c0 = int(gi[i]) * 32, int(gj[i]) * 32
    a = rasters[str(period[i])][r0:r0 + 32, c0:c0 + 32]
    for k in (1, 2, 3):
        frac[j, k - 1] = float((a == k).mean())
    frac[j, 3] = float((a > 0).mean())

wf = gt == 1
miss = wf & (pred != 1)
ok = wf & (pred == 1)
print("windfall val n", int(wf.sum()), "ok", int(ok.sum()), "missed", int(miss.sum()))
for name, m in (("all windfall", wf), ("correct", ok), ("missed", miss)):
    print(name, "frac logging/windfall/fire/changed:",
          np.round(frac[m].mean(0), 3).tolist(),
          "median windfall", round(float(np.median(frac[m, 1])), 3))
# how many windfall chips are "pure" (>=50% windfall pixels)?
for thr in (0.1, 0.25, 0.5):
    pure = wf & (frac[:, 1] >= thr)
    pp = pred[pure]
    tp = int(((pp == 1) & pure[pure]).sum())
    f1 = 2 * tp / max(2 * tp + int(((pp == 1) & ~pure[pure]).sum()) + int((pure[pure] & (pp != 1)).sum()), 1)
    print(f"windfall frac>={thr}: n={int(pure.sum())} model recall={float((pp == 1).mean()):.3f}")
