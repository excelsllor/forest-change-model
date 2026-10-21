"""Visualise vozh windfall errors (pre/post RGB + false colour) for the attn TypeNet."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
sys.path.insert(0, os.path.join(ROOT, "src"))

from typenet_v3 import CnnDofaLike, WL6, band_weights  # noqa: E402
from typenet_big import TypeNetFeats  # noqa: E402

CANON = {"2023-09": 258, "2024-05": 135, "2025-07": 196, "2026-06": 166}
SEQ = ("2023-09", "2024-05", "2025-07", "2026-06")
PLEN = {"p1": 2, "p2": 3, "p3": 4}
dev = torch.device("cuda")


def stretch(img, lo=2, hi=98):
    a = img.astype(np.float32)
    p1, p2 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p1) / max(p2 - p1, 1e-6), 0, 1)


z = np.load(os.path.join(EVAL, "type_vozh_raw.npz"))
cls = z["cls"].astype(np.int64)
split = np.array([str(x) for x in z["split"]])
period = np.array([str(x) for x in z["period"]])
dt = z["dt"].astype(np.float32)
vol = z["vol"].astype(np.float32)   # (n,4,6,32,32) raw/10000
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
    lg = model(torch.from_numpy(feats).to(dev), torch.from_numpy(dt[idx]).to(dev),
               torch.from_numpy(doy).to(dev))
pred = lg.argmax(1).cpu().numpy()
gt = cls[idx] - 1                      # 0 logging, 1 windfall, 2 fire (changed chips)

miss = idx[(gt == 1) & (pred == 0)][:8]
ok = idx[(gt == 1) & (pred == 1)][:4]
log = idx[(gt == 0) & (pred == 0)][:4]
rows = [("missed windfall", i) for i in miss] + [("windfall ok", i) for i in ok] + \
       [("logging", i) for i in log]

fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.1 * len(rows)), squeeze=False)
for r, (tag, i) in enumerate(rows):
    li = int((dt[i] > 0).sum() + 1)
    pre, post = vol[i, 0], vol[i, li - 1]
    rgb_pre = np.stack([pre[2], pre[1], pre[0]], -1)
    rgb_post = np.stack([post[2], post[1], post[0]], -1)
    fc_post = np.stack([post[3], post[2], post[1]], -1)
    for c, img in enumerate((rgb_pre, rgb_post, fc_post)):
        axes[r, c].imshow(stretch(img), interpolation="nearest")
        axes[r, c].axis("off")
    axes[r, 0].set_title(f"{tag} (cls {cls[i]}, pred {pred[np.where(idx == i)[0][0]] + 1})",
                         fontsize=8, loc="left")
fig.tight_layout()
out = os.path.join(ROOT, "outputs", "vozh", "windfall_montage.png")
fig.savefig(out, dpi=110)
print("saved", out, "rows", len(rows), flush=True)
