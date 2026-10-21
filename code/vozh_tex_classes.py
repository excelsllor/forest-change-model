"""Per-class P/R/F1 for the Vozh-tuned texture TypeNet on the vozh val split."""
import os
import sys

import numpy as np
import torch

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
TEX = os.path.join(ROOT, "outputs", "vozh", "tex_exp")
sys.path.insert(0, os.path.join(ROOT, "src"))

from typenet_big import TypeNetFeats  # noqa: E402

dev = torch.device("cuda")
z = np.load(os.path.join(TEX, "vozh_feats.npz"))
feats = z["feats"].astype(np.float32)
tex = z["tex"].astype(np.float32)
dt = z["dt"].astype(np.float32)
doy = z["doy"].astype(np.float32)
y = z["y"].astype(np.int64)
split = np.array([str(x) for x in z["split"]])
model = TypeNetFeats(cfc_width=256, blocks=2, head_width=256, head_mode="summary",
                     readout="attn", use_spec=True).to(dev)
model.load_state_dict(torch.load(os.path.join(EVAL, "typenet_vozh_attn_tex.pth"),
                                 map_location="cpu"))
model.eval()
va = np.where(split == "val")[0]
pred = np.zeros(len(va), np.int64)
with torch.no_grad():
    for i0 in range(0, len(va), 128):
        j = va[i0:i0 + 128]
        lg = model(torch.from_numpy(feats[j]).to(dev), torch.from_numpy(dt[j]).to(dev),
                   torch.from_numpy(doy[j]).to(dev), torch.from_numpy(tex[j]).to(dev))
        pred[i0:i0 + len(j)] = lg.argmax(1).cpu().numpy()
yy = y[va]
print(f"vozh-tuned val n={len(yy)} acc={float((yy == pred).mean()):.3f}")
for k, name in ((0, "logging"), (1, "windfall"), (2, "fire")):
    nk = int((yy == k).sum())
    tp = int(((pred == k) & (yy == k)).sum())
    fp = int(((pred == k) & (yy != k)).sum())
    fn = int(((yy == k) & (pred != k)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    print(f"  {name:8s} n={nk:4d} P={P:.3f} R={R:.3f} F1={2*P*R/max(P+R,1e-9):.3f}")
