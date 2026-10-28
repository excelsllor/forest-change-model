"""CfC U-Net: pixel change mask at 10 m, arbitrary T (2..8).

Vozh chips 32x32, 9 features/date (6 bands + NDVI/NBR/NDMI). Per-date shared U-Net
encoder -> CfCBiEncoderPN over the time axis at the bottleneck (per spatial location,
length-aware) -> U-Net decoder -> pixel logits. Train p1+p2, test p3.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
MOS = os.path.join(ROOT, "outputs", "vozh", "mosaic")
LAB = os.path.join(ROOT, "outputs", "vozh", "labels")
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
OUT = os.path.join(ROOT, "outputs", "vozh", "cfc_unet")
CACHE = os.path.join(EVAL, "cfc_unet_data.npz")
DATES = ("2023-09", "2024-05", "2025-07", "2026-06")
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
PIDS = ("p1", "p2", "p3")
sys.path.insert(0, SRC)
from cfc import CfCBiBlock  # noqa: E402
from typenet_big import AttnReadout  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 10000.0


def build_cache():
    z = np.load(os.path.join(EVAL, "vozh_ctx_mask.npz"), allow_pickle=True)
    gi = z["gi"].astype(int)
    gj = z["gj"].astype(int)
    per = np.array([str(x) for x in z["period"]])
    split = np.array([str(x) for x in z["split"]])
    dt = z["dt"].astype(np.float32)
    N = len(gi)
    X = np.empty((N, len(DATES), 9, 32, 32), np.float16)
    for ti, date in enumerate(DATES):
        arrs = []
        for b in BANDS:
            with rasterio.open(os.path.join(MOS, f"{date}_{b}.tif")) as ds:
                arrs.append(ds.read(1))
        for i in range(N):
            r, c = int(gi[i]) * 32, int(gj[i]) * 32
            b = np.stack([a[r:r + 32, c:c + 32] for a in arrs]).astype(np.float32) / SCALE
            ndvi = (b[3] - b[2]) / np.maximum(b[3] + b[2], 1e-6)
            nbr = (b[3] - b[5]) / np.maximum(b[3] + b[5], 1e-6)
            ndmi = (b[3] - b[4]) / np.maximum(b[3] + b[4], 1e-6)
            X[i, ti] = np.stack([*b, ndvi, nbr, ndmi]).astype(np.float16)
        del arrs
        print(f"[data] {date} done", flush=True)
    Y = np.zeros((N, 32, 32), np.uint8)
    for pid in PIDS:
        with rasterio.open(os.path.join(LAB, f"classes_{pid}.tif")) as ds:
            lab = ds.read(1)
        idx = np.where(per == pid)[0]
        for i in idx:
            r, c = int(gi[i]) * 32, int(gj[i]) * 32
            w = lab[r:r + 32, c:c + 32]
            Y[i] = ((w > 0) & (w != 255)).astype(np.uint8)
        print(f"[data] {pid}: {len(idx)} labels", flush=True)
    np.savez_compressed(CACHE, X=X, Y=Y, per=per, split=split, dt=dt)
    print(f"[data] saved {CACHE} X={X.shape} pos={Y.mean():.4f}", flush=True)


def load_cache():
    if not os.path.exists(CACHE):
        build_cache()
    z = np.load(CACHE)
    return z["X"], z["Y"], np.array([str(x) for x in z["per"]]), \
        np.array([str(x) for x in z["split"]]), z["dt"]


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class CfCUNet(nn.Module):
    def __init__(self, in_ch=9, base=32, hidden=128):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.cfc = nn.ModuleList([CfCBiBlock(base * 4 if i == 0 else 2 * hidden, hidden) for i in range(2)])
        self.pool = AttnReadout(2 * hidden, width=2 * hidden)
        self.up3 = nn.ConvTranspose2d(4 * hidden, base * 2, 2, 2)
        self.d3 = ConvBlock(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d2 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x, dt, lengths):
        b, t, c, h, w = x.shape
        f1 = self.e1(x.reshape(b * t, c, h, w))
        f2 = self.e2(F.avg_pool2d(f1, 2))
        f3 = self.e3(F.avg_pool2d(f2, 2))
        d, h4, w4 = f3.shape[1:]
        seq = f3.reshape(b, t, d, h4, w4).permute(0, 3, 4, 1, 2).reshape(b * h4 * w4, t, d)
        dtx = dt.unsqueeze(1).expand(b, h4 * w4, t).reshape(-1, t)
        lx = lengths.unsqueeze(1).expand(b, h4 * w4).reshape(-1)
        for blk in self.cfc:
            seq = blk(seq, dtx)
        valid = torch.arange(t, device=x.device)[None] < lx[:, None]
        z = self.pool(seq, valid)
        z = z.reshape(b, h4, w4, z.shape[-1]).permute(0, 3, 1, 2)
        s2 = f2.reshape(b, t, 2 * 32, h // 2, w // 2).mean(1)
        s1 = f1.reshape(b, t, 32, h, w).mean(1)
        y = self.d3(torch.cat([self.up3(z), s2], 1))
        y = self.d2(torch.cat([self.up2(y), s1], 1))
        return self.head(y).squeeze(1)


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    num = (2 * (p * y).sum((1, 2)) + 1.0)
    den = (p.sum((1, 2)) + y.sum((1, 2)) + 1.0)
    return 1.0 - (num / den).mean()


def evaluate(model, X, Y, idx, dt, thr=0.5):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            j = idx[i:i + 64]
            xb = torch.from_numpy(X[j].astype(np.float32)).to(dev)
            db = torch.from_numpy(dt[j]).to(dev)
            lb = torch.full((len(j),), X.shape[1], dtype=torch.int64, device=dev)
            probs.append(torch.sigmoid(model(xb, db, lb)).cpu().numpy())
    p = np.concatenate(probs)
    yy = Y[idx].astype(np.uint8)
    pred = (p > thr).astype(np.uint8)
    f = f1_score(yy.ravel(), pred.ravel(), zero_division=0)
    pr = precision_score(yy.ravel(), pred.ravel(), zero_division=0)
    rc = recall_score(yy.ravel(), pred.ravel(), zero_division=0)
    iou = f / (2 - f) if f > 0 else 0.0
    area_ratio = pred.sum() / max(yy.sum(), 1)
    chip = (pred.reshape(len(idx), -1).any(1) == yy.reshape(len(idx), -1).any(1)).mean()
    return {"f1": float(f), "precision": float(pr), "recall": float(rc),
            "iou": float(iou), "area_ratio": float(area_ratio),
            "chip_acc": float(chip), "n": int(len(idx))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-p3", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(OUT, exist_ok=True)
    X, Y, per, split, dt = load_cache()
    if args.include_p3:
        tr = np.where(split == "train")[0]
        va = np.where(split == "val")[0]
    else:
        tr = np.where((split == "train") & np.isin(per, ["p1", "p2"]))[0]
        va = np.where((split == "val") & np.isin(per, ["p1", "p2"]))[0]
    te = np.where((split == "val") & (per == "p3"))[0]
    print(f"[data] train={len(tr)} val={len(va)} test_p3={len(te)} pos={Y.mean():.4f}",
          flush=True)
    pos = Y[tr].sum() / max((1 - Y[tr]).sum(), 1)
    pw = torch.tensor(max(pos, 1.0), device=dev)
    model = CfCUNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(tr) // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
    best = (-1.0, None)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order) - args.bs + 1, args.bs):
            j = order[i:i + args.bs]
            xb = torch.from_numpy(X[j].astype(np.float32)).to(dev)
            yb = torch.from_numpy(Y[j].astype(np.float32)).to(dev)
            db = torch.from_numpy(dt[j]).to(dev)
            lb = torch.full((len(j),), X.shape[1], dtype=torch.int64, device=dev)
            logits = model(xb, db, lb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw) + dice_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        if ep % 2 == 0 or ep == args.epochs - 1:
            v = evaluate(model, X, Y, va, dt)
            print(f"[ep{ep}] loss={tot / max(steps, 1):.4f} val F1={v['f1']:.4f} "
                  f"IoU={v['iou']:.4f} P={v['precision']:.4f} R={v['recall']:.4f} "
                  f"area={v['area_ratio']:.3f}", flush=True)
            if v["f1"] > best[0]:
                best = (v["f1"], {k: v_.clone() for k, v_ in model.state_dict().items()})
    model.load_state_dict(best[1])
    res = {"train": {"n": len(tr)}, "val": evaluate(model, X, Y, va, dt),
           "test_p3": evaluate(model, X, Y, te, dt), "seconds": round(time.time() - t0, 1)}
    # threshold scan on val
    scan = []
    for t in np.arange(0.1, 0.91, 0.05):
        v = evaluate(model, X, Y, va, dt, float(t))
        scan.append((v["f1"], float(t)))
    res["best_thr"] = max(scan)[1]
    res["val_at_best"] = evaluate(model, X, Y, va, dt, res["best_thr"])
    res["test_p3_at_best"] = evaluate(model, X, Y, te, dt, res["best_thr"])
    # spectral baseline dNDVI/dNBR on the same chips
    def spectral(d, thr_d=0.10, thr_b=0.05):
        dndvi = d[:, -1, 6] - d[:, 0, 6]
        dnbr = d[:, 0, 7] - d[:, -1, 7]
        return ((dndvi < -thr_d) | (dnbr > thr_b))
    def base_metrics(idx):
        pred = spectral(X[idx].astype(np.float32))
        yy = Y[idx]
        f = f1_score(yy.ravel(), pred.ravel(), zero_division=0)
        return {"f1": float(f), "iou": float(f / (2 - f) if f > 0 else 0),
                "area_ratio": float(pred.sum() / max(yy.sum(), 1))}
    res["spectral_val"] = base_metrics(va)
    res["spectral_p3"] = base_metrics(te)
    torch.save({"state": model.state_dict()}, os.path.join(OUT, f"cfc_unet_seed{args.seed}{args.tag}.pth"))
    with open(os.path.join(OUT, f"result_seed{args.seed}{args.tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    print(f"[cfc_unet] saved {OUT} in {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()





