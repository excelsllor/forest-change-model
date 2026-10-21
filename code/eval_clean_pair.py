"""Compare mask checkpoints on train/all/clean val (context windows)."""
import argparse
import os
import sys

import numpy as np
import rasterio
import torch
from sklearn.metrics import f1_score

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
OUT = os.path.join(ROOT, "outputs", "vozh", "ctx_exp")
sys.path.insert(0, os.path.join(ROOT, "src"))
from exp_ctx_mask import FeatMaskNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="mask_ctx_attn3_len.pth")
ap.add_argument("--mode", default="attn", choices=("none", "conv", "attn", "raw"))
ap.add_argument("--cont", default="inline", choices=("inline", "sep"))
args = ap.parse_args()

dev = torch.device("cuda")
z = np.load(os.path.join(EVAL, "vozh_ctx_mask.npz"), mmap_mode="r")
gi, gj, y = z["gi"], z["gj"], z["y"]
split = np.array([str(x) for x in z["split"]])
dt = z["dt"]
with rasterio.open(os.path.join(ROOT, "outputs", "vozh", "labels", "quarters.tif")) as ds:
    q = ds.read(1)
hg, wg = q.shape[0] // 32, q.shape[1] // 32
qg = q[16::32, 16::32][:hg, :wg].astype(np.int64)
trq = set(np.unique(qg[gi[split == "train"], gj[split == "train"]]).tolist())
p = 3
qp = np.zeros((hg + 2 * p, wg + 2 * p), np.int64) - 1
qp[p:p + hg, p:p + wg] = qg


def scores(model, idx, k=3):
    out = np.zeros(len(idx), np.float32)
    for i0 in range(0, len(idx), 128):
        j = idx[i0:i0 + 128]
        c = torch.from_numpy(np.asarray(z["center"][j], dtype=np.float32)).to(dev)
        x = torch.from_numpy(np.asarray(z[f"ctx{k}"][j], dtype=np.float32)).to(dev)
        v = torch.from_numpy(np.asarray(z[f"valid{k}"][j])).to(dev)
        d = torch.from_numpy(np.asarray(dt[j], dtype=np.float32)).to(dev)
        do = torch.from_numpy(np.asarray(z["doy"][j], dtype=np.float32)).to(dev)
        L = ((d > 0).sum(1) + 1).long()
        with torch.no_grad():
            lg = model(c, x, v, d, do, L)
        out[i0:i0 + len(j)] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
    return out


def mets(sc, yy):
    pr = (sc > 0).astype(np.int64)
    f1 = f1_score(yy, pr, average="binary", zero_division=0)
    P = float(((pr == 1) & (yy == 1)).sum() / max(int(pr.sum()), 1))
    R = float(((pr == 1) & (yy == 1)).sum() / max(int(yy.sum()), 1))
    return f1, P, R


model = FeatMaskNet(args.mode, 3, ctx_width=256, cont_mode=args.cont).to(dev)
model.load_state_dict(torch.load(os.path.join(OUT, args.ckpt), map_location="cpu"))
model.eval()
tr = np.where(split == "train")[0]
va = np.where(split == "val")[0]
clean = []
for i in va:
    r0, c0 = gi[i] + p - 1, gj[i] + p - 1
    cells = qp[r0:r0 + 3, c0:c0 + 3]
    if not any(int(v) in trq for v in np.unique(cells)):
        clean.append(i)
clean = np.array(clean)
a = mets(scores(model, tr), y[tr])
b = mets(scores(model, va), y[va])
c = mets(scores(model, clean), y[clean])
print(f"{args.ckpt} ({args.mode}/{args.cont}): train F1={a[0]:.3f} | "
      f"val(all) F1={b[0]:.3f} P={b[1]:.3f} R={b[2]:.3f} | "
      f"clean(n={len(clean)}) F1={c[0]:.3f} P={c[1]:.3f} R={c[2]:.3f}", flush=True)
