"""Evaluate a saved TypeNet checkpoint per source: per-class F1 + macro over PRESENT classes.

The earlier per-source macro-F1 was computed over all 3 labels, so single-class sources
(california = fire only, vozh = almost no fire) were deflated ~3x. This reports honest numbers.
"""

import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
sys.path.insert(0, os.path.join(ROOT, "src"))

from typenet_big import TypeNetFeats  # noqa: E402
from train_typenet_big import NAMES, load_all, predict  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="typenet_big_w512.pth")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--head-width", type=int, default=64)
    ap.add_argument("--head-mode", default="plain", choices=("plain", "summary"))
    ap.add_argument("--spec", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srcs = load_all()
    model = TypeNetFeats(cfc_width=args.width, blocks=args.blocks,
                         head_width=args.head_width, head_mode=args.head_mode,
                         use_spec=args.spec).to(dev)
    model.load_state_dict(torch.load(os.path.join(EVAL, args.ckpt), map_location="cpu"))
    val_pairs = [(s, int(i)) for s in range(len(srcs))
                 for i in np.where(srcs[s]["split"] == "val")[0]]
    vy = np.array([srcs[s]["y"][i] for s, i in val_pairs], np.int64)
    preds = predict(model, srcs, val_pairs, dev)

    print(f"[eval] {args.ckpt}: n={len(vy)} classes={np.bincount(vy, minlength=3).tolist()}",
          flush=True)
    present = np.unique(vy)
    m_all = float(f1_score(vy, preds, average="macro", labels=[0, 1, 2], zero_division=0))
    m_pres = float(f1_score(vy, preds, average="macro", labels=present, zero_division=0))
    print(f"[eval] ALL: macro(3 labels)={m_all:.3f} | macro(present {present.tolist()})="
          f"{m_pres:.3f}", flush=True)
    for c in range(3):
        mc = vy == c
        if mc.sum():
            f1c = float(f1_score(vy[mc], preds[mc], average="macro", labels=[c],
                                 zero_division=0))
            print(f"[eval]   class {c}: n={int(mc.sum())} F1={f1c:.3f}", flush=True)
    print("[eval] per source:", flush=True)
    for s in range(len(srcs)):
        m = np.array([p[0] == s for p in val_pairs])
        if not m.sum():
            continue
        yy, pp = vy[m], preds[m]
        pres = np.unique(yy)
        f1_pres = float(f1_score(yy, pp, average="macro", labels=pres, zero_division=0))
        f1_3 = float(f1_score(yy, pp, average="macro", labels=[0, 1, 2], zero_division=0))
        per_cls = {int(c): round(float(f1_score(yy, pp, average="macro", labels=[c],
                                                zero_division=0)), 3)
                   for c in pres}
        print(f"[eval]   {srcs[s]['name']}: n={int(m.sum())} classes={np.bincount(yy, minlength=3).tolist()} "
              f"macro(present)={f1_pres:.3f} | macro(3)={f1_3:.3f} | per-class {per_cls}",
              flush=True)


if __name__ == "__main__":
    main()
