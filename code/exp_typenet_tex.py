"""Vozh-only TypeNet with texture features (windfall streaks) vs feats-only.

Texture per date (9 dims): 4-bin gradient-orientation histogram, entropy,
anisotropy, edge density, local std, gradient mean (B08).  Feeds the existing
`spec` slot of TypeNetFeats (use_spec=True) - no change to the encoder.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
OUT = os.path.join(ROOT, "outputs", "vozh", "tex_exp")
sys.path.insert(0, os.path.join(ROOT, "src"))

from typenet_v3 import CnnDofaLike, WL6, band_weights  # noqa: E402
from typenet_big import TypeNetFeats  # noqa: E402

CANON = {"2023-09": 258, "2024-05": 135, "2025-07": 196, "2026-06": 166}
SEQ = ("2023-09", "2024-05", "2025-07", "2026-06")
PLEN = {"p1": 2, "p2": 3, "p3": 4}
dev = torch.device("cuda")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(0)


def tex9(chip):
    g = chip[3].astype(np.float32)                  # B08
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi
    hist = np.zeros(4, np.float32)
    for b in range(4):
        hist[b] = mag[(ang >= b / 4) & (ang < (b + 1) / 4)].sum()
    hist /= max(float(hist.sum()), 1e-6)
    ent = float(-(hist * np.log(hist + 1e-6)).sum() / np.log(4))
    aniso = float((hist.max() - hist.min()) / (hist.mean() + 1e-6))
    edge = float((mag > mag.mean() + mag.std()).mean())
    return np.array([*hist, ent, aniso, edge, float(g.std()), float(mag.mean())],
                    np.float32)


def main():
    z = np.load(os.path.join(EVAL, "type_vozh_raw.npz"))
    cls = z["cls"].astype(np.int64)
    split = np.array([str(x) for x in z["split"]])
    period = np.array([str(x) for x in z["period"]])
    dt = z["dt"].astype(np.float32)
    vol = z["vol"].astype(np.float32)
    keep = np.where(cls >= 1)[0]
    L = (dt[keep] > 0).sum(1) + 1
    y = cls[keep] - 1

    enc = CnnDofaLike(768, layers=12).to(dev)
    enc.load_state_dict(torch.load(os.path.join(EVAL, "encoder_dofa_distilled_big.pth"),
                                   map_location="cpu"))
    enc.eval()
    wl = band_weights(WL6).to(dev)
    n = len(keep)
    feats = np.zeros((n, 4, 768), np.float16)
    tex = np.zeros((n, 4, 9), np.float16)
    doy = np.zeros((n, 4, 3), np.float32)
    with torch.no_grad():
        for i in range(n):
            li = int(L[i])
            v = np.clip(vol[keep[i], :li], 0.0, 1.5)
            feats[i, :li] = enc(torch.from_numpy(v).to(dev)[None], wl)[0].cpu().numpy()
            for t in range(li):
                tex[i, t] = tex9(v[t])
                a = 2 * np.pi * CANON[SEQ[t]] / 365.0
                doy[i, t] = [np.sin(a), np.cos(a), 0.0]
    np.savez_compressed(os.path.join(OUT, "vozh_feats.npz"), feats=feats.astype(np.float32),
                        tex=tex.astype(np.float32), dt=dt[keep], doy=doy,
                        y=y, split=split[keep], period=period[keep], L=L)
    print(f"[tex] saved feats/tex n={n} (wind {int((y == 1).sum())})", flush=True)

    def run(use_tex, tag):
        tr = split[keep] == "train"
        va = split[keep] == "val"
        pools = {c: np.where(tr & (y == c))[0] for c in range(3)}
        model = TypeNetFeats(cfc_width=256, blocks=2, head_width=256, head_mode="summary",
                             readout="attn", use_spec=use_tex).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        steps = 60 * 125
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        best = (-1.0, None, -1)

        def batch(idx):
            T = int(L[idx].max())
            Fb = np.zeros((len(idx), T, 768), np.float32)
            Db = np.zeros((len(idx), T), np.float32)
            Yb = np.zeros((len(idx), T, 3), np.float32)
            Sb = np.zeros((len(idx), T, 9), np.float32)
            for j, i in enumerate(idx):
                li = int(L[i])
                Fb[j, :li] = feats[i, :li]
                Db[j, :li] = dt[keep[i], :li]
                Yb[j, :li] = doy[i, :li]
                Sb[j, :li] = tex[i, :li]
            return Fb, Db, Yb, Sb

        def evaluate():
            model.eval()
            idx = np.where(va)[0]
            pp = np.zeros(len(idx), np.int64)
            with torch.no_grad():
                for i0 in range(0, len(idx), 128):
                    j = idx[i0:i0 + 128]
                    Fb, Db, Yb, Sb = batch(j)
                    lg = model(torch.from_numpy(Fb).to(dev), torch.from_numpy(Db).to(dev),
                               torch.from_numpy(Yb).to(dev),
                               torch.from_numpy(Sb).to(dev))
                    pp[i0:i0 + len(j)] = lg.argmax(1).cpu().numpy()
            m = f1_score(y[va], pp, average="macro", labels=np.unique(y[va]),
                         zero_division=0)
            per = [round(float(f1_score(y[va], pp, average="macro", labels=[c],
                                       zero_division=0)), 3) for c in range(3)]
            return float(m), per

        for ep in range(60):
            model.train()
            for _ in range(125):
                cs = rng.choice(3, size=32, p=[0.45, 0.45, 0.10])
                chunk = [pools[int(c)][rng.integers(0, len(pools[int(c)]))] for c in cs]
                Fb, Db, Yb, Sb = batch(chunk)
                yb = torch.from_numpy(y[chunk]).to(dev)
                lg = model(torch.from_numpy(Fb).to(dev), torch.from_numpy(Db).to(dev),
                           torch.from_numpy(Yb).to(dev), torch.from_numpy(Sb).to(dev))
                loss = F.cross_entropy(lg, yb, label_smoothing=0.05)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            if ep % 4 == 0 or ep == 59:
                m, per = evaluate()
                if m > best[0]:
                    best = (m, {k: v.clone() for k, v in model.state_dict().items()}, ep)
                print(f"[{tag}] ep{ep}: val macro={m:.3f} per={per} best={best[0]:.3f}",
                      flush=True)
        model.load_state_dict(best[1])
        m, per = evaluate()
        torch.save(best[1], os.path.join(OUT, f"typenet_vozh_{tag}.pth"))
        with open(os.path.join(OUT, f"typenet_vozh_{tag}.json"), "w", encoding="utf-8") as fh:
            json.dump({"tag": tag, "macro": m, "per_class": per, "best_epoch": best[2],
                       "use_tex": use_tex}, fh, indent=2)
        print(f"[{tag}] FINAL macro={m:.3f} per={per}", flush=True)

    run(False, "noftex")
    run(True, "tex")


if __name__ == "__main__":
    main()
