"""Vozh windfall diagnosis: confusion, confidence, per-class metrics for the attn TypeNet."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
sys.path.insert(0, os.path.join(ROOT, "src"))

from typenet_big import TypeNetFeats  # noqa: E402
from train_typenet_big import load_all, predict, build_batch  # noqa: E402

dev = torch.device("cuda")
srcs = load_all()
s = 0  # vozh
model = TypeNetFeats(cfc_width=256, blocks=2, head_width=256, head_mode="summary",
                     readout="attn").to(dev)
model.load_state_dict(torch.load(os.path.join(EVAL, "typenet_big_w256_b2_attn.pth"),
                                 map_location="cpu"))
model.eval()
pairs = [(s, int(i)) for i in np.where(srcs[s]["split"] == "val")[0]]
vy = np.array([srcs[s]["y"][i] for s, i in pairs], np.int64)
pp = predict(model, srcs, pairs, dev)
cm = confusion_matrix(vy, pp, labels=[0, 1, 2])
print("confusion (rows=gt logging/windfall/fire, cols=pred):")
print(cm.tolist())
for k, name in ((0, "logging"), (1, "windfall"), (2, "fire")):
    nk = int((vy == k).sum())
    tp = int(((pp == k) & (vy == k)).sum())
    fp = int(((pp == k) & (vy != k)).sum())
    fn = int(((vy == k) & (pp != k)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    print(f"{name:8s} n={nk:4d} P={P:.3f} R={R:.3f} F1={2*P*R/max(P+R,1e-9):.3f}")

# windfall confidence: softmax of the true windfall samples
probs = np.zeros((len(pairs), 3), np.float32)
with torch.no_grad():
    for i0 in range(0, len(pairs), 128):
        chunk = pairs[i0:i0 + 128]
        Fb, Db, Yb, Sb = build_batch(srcs, chunk, np.random.default_rng(0))
        lg = model(torch.from_numpy(Fb).to(dev), torch.from_numpy(Db).to(dev),
                   torch.from_numpy(Yb).to(dev), torch.from_numpy(Sb).to(dev))
        probs[i0:i0 + len(chunk)] = F.softmax(lg, dim=1).cpu().numpy()
wf = vy == 1
print("windfall prob pred: mean", probs[wf].mean(0).round(3).tolist(),
      "| correct windfall mean p(wind)=",
      float(probs[wf & (pp == 1), 1].mean()) if (wf & (pp == 1)).any() else 0,
      "| missed windfall mean p(wind)=",
      float(probs[wf & (pp != 1), 1].mean()) if (wf & (pp != 1)).any() else 0)
print("missed windfall -> pred counts:", np.bincount(pp[wf & (pp != 1)], minlength=3).tolist())
# threshold sweep for windfall (argmax replaced by wind prob > thr)
for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
    other = probs[:, [0, 2]]
    q = np.where(probs[:, 1] > thr, 1, np.where(other.argmax(1) == 0, 0, 2))
    print(f"wind thr={thr:.1f}: acc={float((q == vy).mean()):.3f} "
          f"wind F1={f1_score(vy, q, average='macro', labels=[1], zero_division=0):.3f}")
