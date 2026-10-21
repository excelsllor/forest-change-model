"""Vozh-only windfall fine-tune of the attn TypeNet (low LR, windfall-heavy sampling)."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
sys.path.insert(0, os.path.join(ROOT, "src"))

from typenet_big import TypeNetFeats  # noqa: E402
from train_typenet_big import load_all, build_batch, predict  # noqa: E402

dev = torch.device("cuda")
srcs = load_all()
s = 0
model = TypeNetFeats(cfc_width=256, blocks=2, head_width=256, head_mode="summary",
                     readout="attn").to(dev)
model.load_state_dict(torch.load(os.path.join(EVAL, "typenet_big_w256_b2_attn.pth"),
                                 map_location="cpu"))
pairs_tr = [(s, int(i)) for i in np.where(srcs[s]["split"] == "train")[0]]
pairs_va = [(s, int(i)) for i in np.where(srcs[s]["split"] == "val")[0]]
ytr = np.array([srcs[s]["y"][i] for _, i in pairs_tr], np.int64)
yva = np.array([srcs[s]["y"][i] for _, i in pairs_va], np.int64)
pools = {c: [p for p, y in zip(pairs_tr, ytr) if y == c] for c in range(3)}
print("train pools", {c: len(v) for c, v in pools.items()}, "val", np.bincount(yva).tolist())
rng = np.random.default_rng(0)
w = np.array([0.30, 0.55, 0.15])          # heavily weight windfall in the batch
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
best = (-1.0, None, -1)
for ep in range(12):
    model.train()
    tot = 0.0
    for _ in range(120):
        cs = rng.choice(3, size=32, p=w)
        chunk = [pools[int(c)][rng.integers(0, len(pools[int(c)]))] for c in cs]
        Fb, Db, Yb, Sb = build_batch(srcs, chunk, rng)
        yb = np.array([srcs[s]["y"][i] for _, i in chunk], np.int64)
        lg = model(torch.from_numpy(Fb).to(dev), torch.from_numpy(Db).to(dev),
                   torch.from_numpy(Yb).to(dev), torch.from_numpy(Sb).to(dev))
        loss = F.cross_entropy(lg, torch.from_numpy(yb).to(dev), label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += float(loss.detach())
    pred = predict(model, srcs, pairs_va, dev)
    m = f1_score(yva, pred, average="macro", labels=np.unique(yva), zero_division=0)
    wf = f1_score(yva, pred, average="macro", labels=[1], zero_division=0)
    print(f"ep{ep}: loss={tot / 120:.3f} vozh macro={m:.3f} windfall F1={wf:.3f}", flush=True)
    if m > best[0]:
        best = (m, {k: v.clone() for k, v in model.state_dict().items()}, ep)
model.load_state_dict(best[1])
pred = predict(model, srcs, pairs_va, dev)
print("BEST ep", best[2], "macro", round(best[0], 3),
      "per-class", [round(float(f1_score(yva, pred, average="macro", labels=[c],
                                         zero_division=0)), 3) for c in range(3)])
torch.save(best[1], os.path.join(EVAL, "typenet_big_w256_b2_attn_vozhft.pth"))
print("saved typenet_big_w256_b2_attn_vozhft.pth")
