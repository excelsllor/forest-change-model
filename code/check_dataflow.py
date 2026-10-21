"""Data-flow checks requested in review: dt/L, ctx-drop zeroing (torch vs ONNX), radiometry."""
import os
import sys

import numpy as np
import torch

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
OUT = os.path.join(ROOT, "outputs", "vozh", "ctx_exp")
PLUGIN_MODEL = os.path.join(ROOT, "qgis_plugin", "forest_change", "model")
sys.path.insert(0, os.path.join(ROOT, "src"))

from exp_ctx_mask import FeatMaskNet  # noqa: E402

print("=== dt / lengths ===")
z = np.load(os.path.join(EVAL, "type_vozh_raw.npz"))
dt = z["dt"]
L = (dt > 0).sum(1) + 1
print("type dt unique rows:", np.unique(dt, axis=0).tolist())
print("type L:", np.bincount(L).tolist(), "min positive gap:",
      float(dt[dt > 0].min()), "zero gaps among k<L:",
      int(sum(((dt[i, 1:L[i]] == 0)).sum() for i in range(len(dt)))))

zm = np.load(os.path.join(EVAL, "vozh_ctx_mask.npz"))
dtm = zm["dt"]
Lm = (dtm > 0).sum(1) + 1
cv = zm["center_valid"]
print("mask dt unique rows:", np.unique(dtm, axis=0).tolist())
print("mask L:", np.bincount(Lm).tolist())
bad = int(sum((dtm[i, 1:Lm[i]] == 0).sum() for i in range(len(dtm))))
print("mask zero gaps among k<L:", bad)
mism = int(((cv.sum(1) != Lm) & (cv.sum(1) > 0)).sum())
print("center_valid.sum != L (non-empty):", mism)
ok = np.array([bool(cv[i, :Lm[i]].all()) for i in range(len(Lm))])
print("trainer drops:", int((~ok).sum()), "(train", int(((~ok) & (np.array([str(x) for x in zm["split"]]) == "train")).sum()),
      "val", int(((~ok) & (np.array([str(x) for x in zm["split"]]) == "val")).sum()), ")",
      "| empty-center:", int((cv.sum(1) == 0).sum()), "| partial:", mism,
      "| sum:", int((cv.sum(1) == 0).sum()) + mism)

print("\n=== radiometry (mosaic 2025-07) ===")
import rasterio  # noqa: E402
MOS = os.path.join(ROOT, "outputs", "vozh", "mosaic")
for b in ("B02", "B04", "B08", "B11"):
    with rasterio.open(os.path.join(MOS, f"2025-07_{b}.tif")) as ds:
        a = ds.read(1, window=((3000, 3400), (6000, 6400)))
    v = a.astype(np.int64)
    vv = v[v > 0]
    print(b, "min", int(v.min()), "p1/p50/p99", np.percentile(vv, [1, 50, 99]).astype(int).tolist())

print("\n=== ctx-drop / empty neighbours: torch vs ONNX ===")
m = FeatMaskNet("attn", 3, ctx_width=256)
m.load_state_dict(torch.load(os.path.join(OUT, "mask_ctx_attn3_len.pth"), map_location="cpu"))
m.eval()
torch.manual_seed(0)
B, T = 4, 4
feats = torch.randn(B, T, 256) * 0.3
ctx = torch.randn(B, T, 3, 3, 256) * 0.1
valid = torch.ones(B, T, 3, 3, dtype=torch.bool)
valid[1] = False                      # whole context dropped for sample 1
valid[2, 2:] = False                  # dates 2,3 empty for sample 2
dtb = torch.zeros(B, T); dtb[:, 1] = 0.72; dtb[:, 2] = 1.15; dtb[:, 3] = 1.0
doy = torch.randn(B, T, 3) * 0.1
lens = torch.tensor([4, 4, 2, 3], dtype=torch.int64)
with torch.no_grad():
    ref = m(feats, ctx, valid, dtb, doy, lens).numpy()
print("torch logits:", np.round(ref, 4).tolist())

import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(os.path.join(PLUGIN_MODEL, "masknet_cfc.onnx"),
                            providers=["CPUExecutionProvider"])
out = sess.run(None, {"feats": feats.numpy(), "ctx": ctx.numpy(),
                      "valid": valid.float().numpy(), "dt": dtb.numpy(),
                      "doy": doy.numpy(), "lengths": lens.numpy()})[0]
print("onnx  logits:", np.round(out, 4).tolist())
print("max|torch-onnx|:", float(np.abs(out - ref).max()))
print("inputs:", [i.name + str(i.shape) for i in sess.get_inputs()])
