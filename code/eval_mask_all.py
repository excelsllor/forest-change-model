"""Evaluate the final context mask (conv3, length-aware) on all datasets.

vozh      - with 3x3 context (the standard val, exact lengths)
big4/cal/pairs/long - without context (zero-shot; the model's ctx-drop fallback path)
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
OUT = os.path.join(ROOT, "outputs", "vozh", "ctx_exp")
UNI = r"G:\datasets\unified"
sys.path.insert(0, os.path.join(ROOT, "src"))

from exp_ctx_mask import FeatMaskNet  # noqa: E402
from masknet_v2 import CnnEncoder, WL6, band_weights  # noqa: E402

dev = torch.device("cuda")
wl = band_weights(WL6).to(dev)
enc = CnnEncoder().to(dev)
enc.load_state_dict(torch.load(os.path.join(EVAL, "encoder_dofa_distilled.pth"),
                               map_location="cpu"))
enc.eval()
model = FeatMaskNet("conv", 3, ctx_width=256).to(dev)
model.load_state_dict(torch.load(os.path.join(OUT, "mask_ctx_conv3_len.pth"),
                                 map_location="cpu"))
model.eval()


def feats_of(vol):
    """vol (n,T,6,32,32) raw reflectance -> (n,T,256) features."""
    n, T = vol.shape[:2]
    v = np.clip(vol.astype(np.float32), 0, 1.5).reshape(n * T, 6, 32, 32)
    f = np.empty((n * T, 256), np.float32)
    with torch.no_grad():
        for i0 in range(0, n * T, 256):
            f[i0:i0 + 256] = enc(torch.from_numpy(v[i0:i0 + 256]).to(dev)[:, None],
                                 wl)[:, 0].cpu().numpy()
    return f.reshape(n, T, 256)


def run_batches(feats, ctx, valid, dt, doy, lengths, bs=256):
    out = np.zeros(len(feats), np.int64)
    with torch.no_grad():
        for i0 in range(0, len(feats), bs):
            sl = slice(i0, i0 + bs)
            lg = model(torch.from_numpy(feats[sl]).to(dev),
                       torch.from_numpy(ctx[sl]).to(dev),
                       torch.from_numpy(valid[sl]).to(dev),
                       torch.from_numpy(dt[sl]).to(dev),
                       torch.from_numpy(doy[sl]).to(dev),
                       torch.from_numpy(lengths[sl]).to(dev))
            out[sl] = (lg[:, 1] > lg[:, 0]).cpu().numpy()
    return out


def report(name, y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"{name:10s} n={len(y):6d} pos={int(y.sum()):5d} | P={prec:.3f} R={rec:.3f} "
          f"F1={2 * prec * rec / max(prec + rec, 1e-9):.3f}", flush=True)


def lengths_of(dt):
    return ((dt > 0).sum(axis=1) + 1).astype(np.int64)


# ---------------- vozh (with context)
z = np.load(os.path.join(EVAL, "vozh_ctx_mask.npz"))
sp = np.array([str(x) for x in z["split"]])
va = sp == "val"
cen = z["center"][va].astype(np.float32)
ctx = z["ctx3"][va].astype(np.float32)
val = np.asarray(z["valid3"][va])
dt = z["dt"][va].astype(np.float32)
doy = z["doy"][va].astype(np.float32)
y = z["y"][va]
ln = lengths_of(dt)
p_ctx = run_batches(cen, ctx, val, dt, doy, ln)
report("vozh(ctx)", y, p_ctx)
# the same vozh val with the context blanked (fallback check)
p_no = run_batches(cen, np.zeros_like(ctx), np.zeros_like(val), dt, doy, ln)
report("vozh(noctx)", y, p_no)

# ---------------- big4 (DISFOR pairs, no context)
zb = np.load(os.path.join(UNI, "big4_chips.npz"))
pre = np.concatenate([zb["pre_tr"], zb["pre_va"]])
post = np.concatenate([zb["post_tr"], zb["post_va"]])
x1 = np.concatenate([zb["X1_tr"], zb["X1_va"]]).astype(np.float32) / 10000.0
x2 = np.concatenate([zb["X2_tr"], zb["X2_va"]]).astype(np.float32) / 10000.0
vol = np.stack([x1, x2], 1)
b_split = np.array(["train"] * len(zb["X1_tr"]) + ["val"] * len(zb["X1_va"]))
va = b_split == "val"
gaps = np.array([max((pd.Timestamp(str(b)) - pd.Timestamp(str(a))).days, 1) / 365.25
                 for a, b in zip(pre, post)], np.float32)
b_dt = np.stack([np.zeros_like(gaps), gaps], 1)[va]

def doy3(d):
    d = pd.Timestamp(str(d))
    t = d.dayofyear
    return np.array([np.sin(2 * np.pi * t / 365.0), np.cos(2 * np.pi * t / 365.0),
                     d.year - 2015.0], np.float32)

b_doy = np.stack([[doy3(a), doy3(b)] for a, b in zip(pre, post)])[va]
f = feats_of(vol[va])
T = 2
ctx0 = np.zeros((f.shape[0], T, 3, 3, 256), np.float32)
val0 = np.zeros((f.shape[0], T, 3, 3), bool)
yy = np.ones(f.shape[0], np.int64)
report("big4", yy, run_batches(f, ctx0, val0, b_dt, b_doy, np.full(len(yy), 2)))

# ---------------- california
zc = np.load(os.path.join(UNI, "california.npz"))
c_split = np.array([str(x) for x in zc["split"]])
va = c_split == "val"
f = feats_of(zc["chips"][va])
dt = zc["dt"][va].astype(np.float32)
doy = zc["doy"][va].astype(np.float32)
yy = zc["y"][va].astype(np.int64)
ln = lengths_of(dt)
report("california", yy, run_batches(f, np.zeros((len(f), 2, 3, 3, 256), np.float32),
                                     np.zeros((len(f), 2, 3, 3), bool), dt, doy, ln))

# ---------------- pairs (DISFOR pairs, binary)
zp = np.load(os.path.join(EVAL, "binary_ds.npz"))
p_split = np.array([str(x) for x in zp["dsplit"]])
va = p_split == "val"
f = feats_of(zp["dvol"][va])
Tp = f.shape[1]
dt = zp["ddt"][va][:, :Tp].astype(np.float32)
doy = zp["ddoy"][va][:, :Tp].astype(np.float32)
yy = (zp["dcls"][va] > 0).astype(np.int64)
ln = lengths_of(dt)
report("pairs", yy, run_batches(f, np.zeros((len(f), Tp, 3, 3, 256), np.float32),
                                np.zeros((len(f), Tp, 3, 3), bool), dt, doy, ln))

# ---------------- long (DISFOR long series, T<=8 evenly spaced)
zl = np.load(os.path.join(EVAL, "binary_long.npz"))
l_split = np.array([str(x) for x in zl["split"]])
va = l_split == "val"
vol = zl["vol"][va].astype(np.float32)
ldt = zl["dt"][va].astype(np.float32)
ldoy = zl["doy"][va].astype(np.float32)
yy = (zl["cls"][va] > 0).astype(np.int64)
T = vol.shape[1]
idx = np.linspace(0, T - 1, min(8, T)).round().astype(int)
vol8 = vol[:, idx]
dt8 = np.zeros((len(vol8), len(idx)), np.float32)
for j, i in enumerate(idx):
    if j > 0:
        dt8[:, j] = ldt[:, idx[j - 1] + 1:idx[j] + 1].sum(1)
doy8 = ldoy[:, idx]
f = feats_of(vol8)
ln = lengths_of(dt8)
report("long", yy, run_batches(f, np.zeros((len(f), len(idx), 3, 3, 256), np.float32),
                               np.zeros((len(f), len(idx), 3, 3), bool), dt8, doy8, ln))
