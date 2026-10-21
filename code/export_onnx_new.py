"""Export TypeNet (big encoder + attention readout) to ONNX and verify against PyTorch.

1) encoder: CnnDofaLike 12x768 (dilated)   vol(B,6,32,32) -> feats(B,768)
2) typenet: TypeNetFeats (cross-attention readout)  feats(B,T,768) + dt(B,T) + doy(B,T,3) -> logits(B,3)

Outputs into qgis_plugin/forest_change/model/: typenet_encoder.onnx, typenet_cfc.onnx.
Verification on random and real val samples with onnxruntime CPU.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
PLUGIN_MODEL = os.path.join(ROOT, "qgis_plugin", "forest_change", "model")
sys.path.insert(0, os.path.join(ROOT, "src"))

from masknet_v2 import WL6, band_weights  # noqa: E402
from typenet_v3 import CnnDofaLike  # noqa: E402
from typenet_big import TypeNetFeats  # noqa: E402
from onnx_opt import fold_stem_  # noqa: E402
from export_onnx_mask_ctx import strip_unused_input  # noqa: E402


class EncWrap(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.register_buffer("wl", band_weights(WL6))

    def forward(self, vol):                       # (B,6,32,32) raw reflectance
        v = torch.clamp(vol, 0.0, 1.5)
        return self.enc(v[:, None], self.wl)[:, 0]


class TypeWrap(nn.Module):
    """Explicit 3-arg signature (no optional spec) for a clean ONNX graph."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, feats, dt, doy):
        return self.m(feats, dt, doy)


def export(model, args, path, names, dynamic):
    torch.onnx.export(model, args, path, input_names=names,
                      output_names=["logits"], opset_version=17, do_constant_folding=True,
                      dynamic_axes=dynamic, dynamo=False)
    print(f"[onnx] exported {os.path.basename(path)} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", default="attn", choices=("cfc", "attn"))
    ap.add_argument("--ckpt", default="typenet_big_w256_b2_attn.pth")
    ap.add_argument("--spec", action="store_true", help="9 texture features per date")
    ap.add_argument("--out", default="typenet_cfc.onnx")
    args = ap.parse_args()
    import onnxruntime as ort
    os.makedirs(PLUGIN_MODEL, exist_ok=True)

    # ---------------- 1) encoder
    enc = CnnDofaLike(768, layers=12)
    enc.load_state_dict(torch.load(os.path.join(EVAL, "encoder_dofa_distilled_big.pth"),
                                   map_location="cpu"))
    enc.eval()
    x = torch.rand(3, 6, 32, 32) * 0.5
    wl = band_weights(WL6)
    with torch.no_grad():
        ref0 = enc(x[:, None], wl)[:, 0].numpy()
    fold_stem_(enc, WL6)
    ew = EncWrap(enc).eval()
    with torch.no_grad():
        ref = ew(x).numpy()
    print(f"[onnx] fold stem: max|diff| vs original = {np.abs(ref - ref0).max():.2e}",
          flush=True)
    export(ew, (x,), os.path.join(PLUGIN_MODEL, "typenet_encoder.onnx"), ["vol"],
           {"vol": {0: "B"}})
    sess = ort.InferenceSession(os.path.join(PLUGIN_MODEL, "typenet_encoder.onnx"),
                                providers=["CPUExecutionProvider"])
    out = sess.run(None, {"vol": x.numpy()})[0]
    print(f"[onnx] enc   max|diff| = {np.abs(out - ref).max():.2e}", flush=True)

    # ---------------- 2) typenet readout
    tm = TypeNetFeats(cfc_width=256, blocks=2, head_width=256, head_mode="summary",
                      use_spec=args.spec, readout=args.readout)
    tm.load_state_dict(torch.load(os.path.join(EVAL, args.ckpt), map_location="cpu"))
    tm.eval()
    tw = torch.jit.script(tm)
    feats = torch.randn(2, 3, 768) * 0.5
    dt = torch.zeros(2, 3)
    dt[:, 1] = 0.7
    dt[:, 2] = 1.1
    doy = torch.randn(2, 3, 3) * 0.3
    spec = torch.zeros(2, 3, 9)
    with torch.no_grad():
        ref = (tw(feats, dt, doy, spec) if args.spec else tw(feats, dt, doy)).numpy()
    torch.onnx.export(tw, (feats, dt, doy, spec), os.path.join(PLUGIN_MODEL, args.out),
                      input_names=["feats", "dt", "doy", "spec"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True, dynamo=False,
                      dynamic_axes={"feats": {0: "B", 1: "T"}, "dt": {0: "B", 1: "T"},
                                    "doy": {0: "B", 1: "T"}, "spec": {0: "B", 1: "T"}})
    removed = False if args.spec else strip_unused_input(
        os.path.join(PLUGIN_MODEL, args.out), "spec")
    print(f"[onnx] exported {args.out} "
          f"({os.path.getsize(os.path.join(PLUGIN_MODEL, args.out)) / 1e6:.1f} MB, "
          f"spec {'used' if args.spec else ('stripped' if removed else 'unused')})",
          flush=True)
    sess = ort.InferenceSession(os.path.join(PLUGIN_MODEL, args.out),
                                providers=["CPUExecutionProvider"])
    for TT in (2, 3, 4, 8):
        f2 = torch.randn(2, TT, 768) * 0.5
        d2 = torch.zeros(2, TT)
        d2[:, 1:] = 0.5
        do2 = torch.randn(2, TT, 3) * 0.3
        s2 = torch.randn(2, TT, 9) * 0.1
        with torch.no_grad():
            r2 = (tw(f2, d2, do2, s2) if args.spec else tw(f2, d2, do2)).numpy()
        feed = {"feats": f2.numpy(), "dt": d2.numpy(), "doy": do2.numpy()}
        if args.spec:
            feed["spec"] = s2.numpy()
        o2 = sess.run(None, feed)[0]
        print(f"[onnx] typenet T={TT} max|diff| = {np.abs(o2 - r2).max():.2e}", flush=True)

    if args.spec:
        print("[onnx] spec model: real-val check skipped (needs texture from raw chips)",
              flush=True)
        return

    # ---------------- 3) real val sample check (exact lengths, mixed T)
    z = np.load(os.path.join(EVAL, "tn_vozh.npz"))
    idx = np.where(np.array([str(s) for s in z["split"]]) == "val")[0][:4]
    Ls = (z["dt"][idx] > 0).sum(1) + 1
    T = int(Ls.max())
    f = torch.from_numpy(z["feats"][idx][:, :T].astype(np.float32))
    d = torch.from_numpy(z["dt"][idx][:, :T].astype(np.float32))
    do = torch.from_numpy(z["doy"][idx][:, :T].astype(np.float32))
    with torch.no_grad():
        r = tw(f, d, do).numpy()
    o = sess.run(None, {"feats": f.numpy(), "dt": d.numpy(), "doy": do.numpy()})[0]
    print(f"[onnx] real vozh val (n=4, T={T}, lens={Ls.tolist()}): "
          f"max|diff| = {np.abs(o - r).max():.2e}", flush=True)


if __name__ == "__main__":
    main()
