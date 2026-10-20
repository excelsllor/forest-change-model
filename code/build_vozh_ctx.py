"""Build vozh context datasets for the MASK (and later the TYPE) model.

Grids : per date, all chips of the mosaic: CnnEncoder (256-d, distilled) -> (4, Hg, Wg, 256) f16
Mask  : all 6030 vozh chips -> center = grid, ctx windows 3x3/5x5 + validity, y (change),
        split, dt, doy. Saves outputs/vozh/mask_eval/vozh_ctx_mask.npz
Type  : optional (--with-big-center): changed chips, big encoder 768-d center -> vozh_ctx_type.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
MOS = os.path.join(ROOT, "outputs", "vozh", "mosaic")
sys.path.insert(0, os.path.join(ROOT, "src"))

from masknet_v2 import CnnEncoder, WL6, band_weights  # noqa: E402
from typenet_v3 import CnnDofaLike  # noqa: E402

DATES = ("2023-09", "2024-05", "2025-07", "2026-06")
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
CHIP = 32
CANON = {"2023-09": 258, "2024-05": 135, "2025-07": 196, "2026-06": 166}
PLEN = {"p1": 2, "p2": 3, "p3": 4}


def grid_file():
    return os.path.join(EVAL, "vozh_ctx_grids.npz")


def build_grids(dev, wl):
    enc = CnnEncoder().to(dev)
    enc.load_state_dict(torch.load(os.path.join(EVAL, "encoder_dofa_distilled.pth"),
                                   map_location="cpu"))
    enc.eval()
    with rasterio.open(os.path.join(MOS, f"{DATES[0]}_B02.tif")) as ds0:
        W, H = ds0.width, ds0.height
        count = ds0.count
    wg, hg = W // CHIP, H // CHIP
    grid = np.zeros((len(DATES), hg, wg, 256), np.float16)
    valid = np.zeros((len(DATES), hg, wg), bool)
    print(f"[ctx] grids {len(DATES)}x{hg}x{wg}, bands={count}", flush=True)
    t0 = time.time()
    for ti, date in enumerate(DATES):
        dss = [rasterio.open(os.path.join(MOS, f"{date}_{b}.tif")) for b in BANDS]
        for gi in range(hg):
            arr = np.stack([d.read(1, window=Window(0, gi * CHIP, wg * CHIP, CHIP))
                            for d in dss]).astype(np.float32) / 10000.0
            arr = np.clip(arr, 0.0, 1.5).reshape(6, CHIP, wg, CHIP).transpose(2, 0, 1, 3)
            chips = np.ascontiguousarray(arr)                     # (wg,6,32,32)
            valid[ti, gi] = chips.sum(axis=(1, 2, 3)) > 0
            with torch.no_grad():
                x = torch.from_numpy(chips).to(dev)[:, None]
                grid[ti, gi] = enc(x, wl)[:, 0].cpu().numpy().astype(np.float16)
            if gi % 40 == 0:
                print(f"[ctx] {date} {gi}/{hg} ({time.time() - t0:.0f}s)", flush=True)
        for d in dss:
            d.close()
        print(f"[ctx] {date} done ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(grid_file(), grid=grid, valid=valid)
    print(f"[ctx] grids saved ({os.path.getsize(grid_file()) / 1e6:.0f} MB)", flush=True)
    return grid, valid


def load_or_build_grids(dev, wl):
    if os.path.exists(grid_file()):
        z = np.load(grid_file(), mmap_mode="r")
        print(f"[ctx] grids loaded {z['grid'].shape}", flush=True)
        return z["grid"], z["valid"]
    return build_grids(dev, wl)


def windows(grid, valid, gi, gj, k):
    _, hg, wg, D = grid.shape
    p = max(2, k // 2)
    gp = np.zeros((len(DATES), hg + 2 * p, wg + 2 * p, D), np.float16)
    vp = np.zeros((len(DATES), hg + 2 * p, wg + 2 * p), bool)
    gp[:, p:p + hg, p:p + wg] = grid
    vp[:, p:p + hg, p:p + wg] = valid
    ctx = np.zeros((len(gi), len(DATES), k, k, D), np.float16)
    val = np.zeros((len(gi), len(DATES), k, k), bool)
    off = k // 2
    for i in range(len(gi)):
        r0, c0 = gi[i] + p - off, gj[i] + p - off
        ctx[i] = gp[:, r0:r0 + k, c0:c0 + k]
        val[i] = vp[:, r0:r0 + k, c0:c0 + k]
    return ctx, val


def feats9(chip):
    b02, b03, b04, b08, b11, b12 = chip
    ndvi = (b08 - b04) / np.maximum(b08 + b04, 1e-6)
    nbr = (b08 - b12) / np.maximum(b08 + b12, 1e-6)
    ndmi = (b08 - b11) / np.maximum(b08 + b11, 1e-6)
    return np.array([b02.mean(), b03.mean(), b04.mean(), b08.mean(), b11.mean(),
                     b12.mean(), ndvi.mean(), nbr.mean(), ndmi.mean()], np.float32)


def doy_per_period(period):
    doy = np.zeros((len(period), 4, 3), np.float32)
    for i, pid in enumerate(period):
        for t in range(PLEN[pid]):
            a = 2 * np.pi * CANON[DATES[t]] / 365.0
            doy[i, t] = [np.sin(a), np.cos(a), 0.0]
    return doy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-big-center", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    wl = band_weights(WL6).to(dev)

    zr = np.load(os.path.join(EVAL, "type_vozh_raw.npz"))
    df = pd.read_csv(os.path.join(EVAL, "dataset.csv"))
    grid, valid = load_or_build_grids(dev, wl)

    gi_all = (df["r"].to_numpy() // CHIP).astype(np.int64)
    gj_all = (df["c"].to_numpy() // CHIP).astype(np.int64)
    dt = zr["dt"].astype(np.float32)
    doy = doy_per_period(np.array([str(x) for x in zr["period"]]))
    split = np.array([str(x) for x in zr["split"]])
    center = grid[:, gi_all, gj_all].transpose(1, 0, 2).astype(np.float16)   # (n,4,256)
    center_valid = valid[:, gi_all, gj_all].T.copy()        # (n,4) bool
    # zero out dates beyond the period (padding must match training: vol was zero there)
    L = ((dt > 0).sum(axis=1) + 1).astype(np.int64)
    for i in range(len(L)):
        center[i, L[i]:] = 0
        center_valid[i, L[i]:] = False
    vol_all = zr["vol"].astype(np.float32)
    spec = np.zeros((len(gi_all), 4, 9), np.float32)
    for i in range(len(gi_all)):
        L_i = int(L[i])
        for t in range(L_i):
            spec[i, t] = feats9(vol_all[i, t])
    out = {"center": center, "center_valid": center_valid, "dt": dt, "doy": doy,
           "spec": spec,
           "y": (zr["cls"] > 0).astype(np.int64), "split": split,
           "gi": gi_all, "gj": gj_all, "cls": zr["cls"],
           "period": np.array([str(x) for x in zr["period"]])}
    for k in (3, 5, 6):
        ctx, val = windows(grid, valid, gi_all, gj_all, k)
        for i in range(len(L)):
            ctx[i, L[i]:] = 0
            val[i, L[i]:] = 0
        out[f"ctx{k}"] = ctx
        out[f"valid{k}"] = val
        print(f"[ctx] mask windows {k}x{k}: {ctx.shape} {ctx.nbytes / 1e6:.0f} MB", flush=True)
    dst = os.path.join(EVAL, "vozh_ctx_mask.npz")
    np.savez(dst, **out)
    print(f"[ctx] saved {dst} ({os.path.getsize(dst) / 1e6:.0f} MB) "
          f"n={len(gi_all)} train={int((split == 'train').sum())} "
          f"val={int((split == 'val').sum())}", flush=True)

    if args.with_big_center:
        keep = np.where((zr["cls"] >= 1) & (zr["cls"] <= 3))[0]
        vol = np.clip(zr["vol"][keep].astype(np.float32), 0.0, 1.5)
        enc = CnnDofaLike(768, layers=12).to(dev)
        enc.load_state_dict(torch.load(os.path.join(EVAL, "encoder_dofa_distilled_big.pth"),
                                       map_location="cpu"))
        enc.eval()
        feats = np.zeros((len(vol), 4, 768), np.float16)
        with torch.no_grad():
            for i0 in range(0, len(vol), 128):
                feats[i0:i0 + 128] = enc(torch.from_numpy(vol[i0:i0 + 128]).to(dev),
                                         wl).cpu().numpy().astype(np.float16)
        out2 = {k: v[keep] for k, v in out.items()
                if k in ("dt", "doy", "split", "gi", "gj", "center_valid", "period", "spec")}
        out2.update({"center": feats, "y": (zr["cls"][keep] - 1).astype(np.int64)})
        for k in (3, 5, 6):
            out2[f"ctx{k}"] = out[f"ctx{k}"][keep]
            out2[f"valid{k}"] = out[f"valid{k}"][keep]
        dst2 = os.path.join(EVAL, "vozh_ctx_type.npz")
        np.savez(dst2, **out2)
        print(f"[ctx] saved {dst2} ({os.path.getsize(dst2) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
