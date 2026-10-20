"""Stage 2: train the mask model (bicfcpn + FROZEN distilled encoder) on the MAX balanced dataset.

Sources (raw reflectance):
  vozh       6030   T<=4   both classes (organizer labels)
  big4      48000   T=2    DISFOR pairs, 4 classes -> all positive
  california 12115  T=2    burned-area pre/post -> positives AND negatives
  long        2307  T<=36  DISFOR long series (val excludes big4-train territories)

Sampler: 50/50 positives/negatives, source-stratified; encoder frozen (from
encoder_dofa_distilled.pth); 60 epochs, cosine, best checkpoint by val F1.
Saves: outputs/vozh/mask_eval/masknet_v2_bicfcpn_big.pth / .json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
UNI = r"G:\datasets\unified"
sys.path.insert(0, os.path.join(ROOT, "src"))

from masknet_v2 import MaskNetV2, WL6, band_weights  # noqa: E402
from train_masknet_v2 import sample_subset, valid_len  # noqa: E402

CANON = {"2023-09": 258, "2024-05": 135, "2025-07": 196, "2026-06": 166}
SEQD = ("2023-09", "2024-05", "2025-07", "2026-06")


def doy3(d):
    d = pd.Timestamp(str(d))
    t = d.dayofyear
    return np.array([np.sin(2 * np.pi * t / 365.0), np.cos(2 * np.pi * t / 365.0),
                     d.year - 2015.0], np.float32)


def sanitize(v, hi=1.5):
    """Clip saturated/nodata reflectance (DISFOR rasters contain DN up to 65535 = 6.55)."""
    return np.clip(v.astype(np.float32), 0.0, hi).astype(np.float16)


def load_all():
    zr = np.load(os.path.join(EVAL, "type_vozh_raw.npz"))
    n_period = {"p1": 2, "p2": 3, "p3": 4}
    vdoy = np.zeros((len(zr["cls"]), 4, 3), np.float32)
    for i, pid in enumerate(zr["period"]):
        for t in range(n_period[pid]):
            a = 2 * np.pi * CANON[SEQD[t]] / 365.0
            vdoy[i, t] = [np.sin(a), np.cos(a), 0.0]
    srcs = [{"name": "vozh", "vol": sanitize(zr["vol"]), "dt": zr["dt"].astype(np.float32),
             "doy": vdoy, "y": (zr["cls"] > 0).astype(np.int64),
             "split": np.array([str(x) for x in zr["split"]])}]

    zb = np.load(os.path.join(UNI, "big4_chips.npz"))
    pre = np.concatenate([zb["pre_tr"], zb["pre_va"]])
    post = np.concatenate([zb["post_tr"], zb["post_va"]])
    gaps = np.array([max((pd.Timestamp(str(b)) - pd.Timestamp(str(a))).days, 1) / 365.25
                     for a, b in zip(pre, post)], np.float32)
    b_dt = np.stack([np.zeros_like(gaps), gaps], 1)
    b_doy = np.stack([np.stack([doy3(a), doy3(b)]) for a, b in zip(pre, post)])
    b_vol = np.concatenate([np.stack([zb["X1_tr"], zb["X2_tr"]], 1),
                            np.stack([zb["X1_va"], zb["X2_va"]], 1)]).astype(np.float32) / 10000.0
    b_split = np.array(["train"] * len(zb["X1_tr"]) + ["val"] * len(zb["X1_va"]))
    srcs.append({"name": "big4", "vol": sanitize(b_vol), "dt": b_dt, "doy": b_doy,
                 "y": np.ones(len(b_vol), np.int64), "split": b_split})

    zc = np.load(os.path.join(UNI, "california.npz"))
    srcs.append({"name": "california", "vol": sanitize(zc["chips"]), "dt": zc["dt"].astype(np.float32),
                 "doy": zc["doy"], "y": zc["y"], "split": zc["split"]})

    zl = np.load(os.path.join(EVAL, "binary_long.npz"))
    l_split = np.array([str(x) for x in zl["split"]])
    b4_train = set(int(x) for x in zb["sid_tr"])
    l_val_ok = np.array([(s == "val") and (int(sid) not in b4_train)
                         for s, sid in zip(l_split, zl["sid"])])
    l_split2 = np.where(l_split == "train", "train", np.where(l_val_ok, "val", "skip"))
    srcs.append({"name": "long", "vol": sanitize(zl["vol"]), "dt": zl["dt"].astype(np.float32),
                 "doy": zl["doy"], "y": (zl["cls"] > 0).astype(np.int64), "split": l_split2})
    for s in srcs:
        print(f"[big] {s['name']}: n={len(s['vol'])} train={int((s['split'] == 'train').sum())} "
              f"val={int((s['split'] == 'val').sum())} pos={int(s['y'].sum())}", flush=True)
    return srcs


W_POS = np.array([0.30, 0.35, 0.20, 0.15])
W_NEG = np.array([0.50, 0.00, 0.35, 0.15])


def build_batch(srcs, chunk, rng):
    subs = []
    for s_i, i in chunk:
        s = srcs[s_i]
        if s["name"] in ("big4", "california"):
            subs.append([0, 1])
        else:
            subs.append(sample_subset(valid_len(s["dt"][i]), rng))
    T = max(len(x) for x in subs)
    Vb = np.zeros((len(chunk), T, 6, 32, 32), np.float32)
    Db = np.zeros((len(chunk), T), np.float32)
    Yb = np.zeros((len(chunk), T, 3), np.float32)
    for j, ((s_i, i), sub) in enumerate(zip(chunk, subs)):
        s = srcs[s_i]
        Vb[j, :len(sub)] = s["vol"][i][sub].astype(np.float32)
        Yb[j, :len(sub)] = s["doy"][i][sub]
        d = np.zeros(len(sub), np.float32)
        for k in range(1, len(sub)):
            d[k] = s["dt"][i, sub[k - 1] + 1:sub[k] + 1].sum()
        Db[j, :len(sub)] = d
    return Vb, Db, Yb


def predict(model, srcs, pairs, dev, wl):
    model.eval()
    out = np.zeros(len(pairs), np.float32)
    with torch.no_grad():
        for i0 in range(0, len(pairs), 128):
            chunk = pairs[i0:i0 + 128]
            Vb, Db, Yb = build_batch(srcs, chunk, np.random.default_rng(0))
            logits = model(torch.from_numpy(Vb).to(dev), wl, torch.from_numpy(Db).to(dev),
                           torch.from_numpy(Yb).to(dev))
            out[i0:i0 + len(chunk)] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epoch-sub", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--head-width", type=int, default=64)
    ap.add_argument("--head-mode", default="plain", choices=("plain", "summary"))
    ap.add_argument("--tv-drop", type=float, default=0.0,
                    help="per-sample prob of zeroing the temporal vector (dt+doy) in training")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
        args.epoch_sub = 256
        args.val_every = 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    srcs = load_all()
    pools = {}
    for s_i, s in enumerate(srcs):
        for cls in (0, 1):
            pools[(s_i, cls)] = np.where((s["split"] == "train") & (s["y"] == cls))[0]
    print("[big] train pools: " + " | ".join(
        f"{srcs[s]['name']}{'pos' if c else 'neg'}={len(pools[(s, c)])}"
        for s in range(len(srcs)) for c in (0, 1)), flush=True)

    model = MaskNetV2(seq="bicfcpn", enc="cnn", head_width=args.head_width,
                      head_mode=args.head_mode).to(dev)
    model.encoder.load_state_dict(torch.load(os.path.join(EVAL, "encoder_dofa_distilled.pth"),
                                             map_location="cpu"))
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[big] encoder frozen (distilled); trainable params {n_tr/1e6:.2f}M", flush=True)
    wl = band_weights(WL6).to(dev)

    val_pairs = [(s, int(i)) for s in range(len(srcs))
                 for i in np.where(srcs[s]["split"] == "val")[0]]
    vy = np.array([srcs[s]["y"][i] for s, i in val_pairs], np.int64)
    print(f"[big] val={len(val_pairs)} (pos {int(vy.sum())})", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=1e-4)
    steps = args.epochs * max(1, args.epoch_sub // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

    def draw(n, cls):
        w = (W_POS if cls == 1 else W_NEG).copy()
        w = w / w.sum()
        out = []
        for _ in range(n):
            s_i = int(rng.choice(len(srcs), p=w))
            pool = pools[(s_i, cls)]
            if len(pool) == 0:
                s_i = next(ss for ss in range(len(srcs)) if len(pools[(ss, cls)]) > 0)
                pool = pools[(s_i, cls)]
            out.append((s_i, int(pool[rng.integers(len(pool))])))
        return out

    best = {"f1": -1.0, "state": None, "epoch": -1}
    t_train = time.time()
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for _ in range(args.epoch_sub // args.bs):
            n_pos = args.bs // 2
            chunk = draw(n_pos, 1) + draw(args.bs - n_pos, 0)
            Vb, Db, Yb = build_batch(srcs, chunk, rng)
            if args.tv_drop > 0:
                drop = rng.random(len(chunk)) < args.tv_drop
                Db[drop] = 0.0
                Yb[drop] = 0.0
            yb = np.array([srcs[s]["y"][i] for s, i in chunk], np.float32)
            logits = model(torch.from_numpy(Vb).to(dev), wl, torch.from_numpy(Db).to(dev),
                           torch.from_numpy(Yb).to(dev))
            loss = F.binary_cross_entropy_with_logits(logits[:, 1] - logits[:, 0],
                                                      torch.from_numpy(yb).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            if torch.isfinite(gnorm):
                opt.step()
            else:
                print("[big] non-finite grad, step skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            sched.step()
            if torch.isfinite(loss):
                tot += float(loss.detach())
            nb += 1
        ep_time = time.time() - t0
        if ep % args.val_every == 0 or ep == args.epochs - 1:
            scores = predict(model, srcs, val_pairs, dev, wl)
            if np.isfinite(scores).all():
                pr_auc = float(average_precision_score(vy, scores))
                f1 = float(f1_score(vy, (scores > 0).astype(np.int64), average="binary",
                                    zero_division=0))
                if f1 > best["f1"]:
                    best = {"f1": f1, "epoch": ep,
                            "state": {k: v.clone() for k, v in model.state_dict().items()}}
                    torch.save(best["state"],
                               os.path.join(EVAL, f"masknet_v2_bicfcpn_big{args.tag}.pth"))
                print(f"[big] ep{ep}: {ep_time:.0f}s loss={tot / max(nb, 1):.3f} val_f1={f1:.3f} "
                      f"val_pr_auc={pr_auc:.3f} best={best['f1']:.3f}", flush=True)
            else:
                print(f"[big] ep{ep}: {ep_time:.0f}s non-finite val scores, skipped", flush=True)
        else:
            print(f"[big] ep{ep}: {ep_time:.0f}s loss={tot / max(nb, 1):.3f}", flush=True)
    print(f"[big] training time: {(time.time() - t_train) / 60:.1f} min", flush=True)

    model.load_state_dict(best["state"])
    scores = predict(model, srcs, val_pairs, dev, wl)
    res = {"best_val_f1": best["f1"], "best_epoch": best["epoch"], "epochs": args.epochs,
           "tv_drop": args.tv_drop, "head_width": args.head_width,
           "head_mode": args.head_mode}
    for s_i, s in enumerate(srcs):
        m = np.array([p[0] == s_i for p in val_pairs])
        if m.sum() == 0:
            continue
        yy = vy[m]
        ss = scores[m]
        res[s["name"]] = {"n": int(m.sum()), "f1": float(f1_score(
            yy, (ss > 0).astype(np.int64), average="binary", zero_division=0)),
            "pr_auc": float(average_precision_score(yy, ss))}
        print(f"[big]   {s['name']}: n={res[s['name']]['n']} f1={res[s['name']]['f1']:.3f} "
              f"pr_auc={res[s['name']]['pr_auc']:.3f}", flush=True)
    torch.save(best["state"], os.path.join(EVAL, f"masknet_v2_bicfcpn_big{args.tag}.pth"))
    with open(os.path.join(EVAL, f"masknet_v2_bicfcpn_big{args.tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"[big] saved masknet_v2_bicfcpn_big{args.tag}.pth/.json", flush=True)


if __name__ == "__main__":
    main()
