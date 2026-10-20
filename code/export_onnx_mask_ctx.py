"""Export the context mask to ONNX: small encoder + length-aware CfC with dynamic T.

Outputs into qgis_plugin/forest_change/model/:
  masknet_encoder.onnx - CnnEncoder (distilled), vol (M,6,32,32) -> feats (M,256)
  masknet_cfc.onnx     - FeatMaskNet conv3, feats/ctx/valid/dt/doy/lengths -> logits
                         T is dynamic (ONNX Loop), lengths = valid dates per sample
"""
import os
import sys

import numpy as np
import torch

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
OUT = os.path.join(ROOT, "outputs", "vozh", "ctx_exp")
PLUGIN_MODEL = os.path.join(ROOT, "qgis_plugin", "forest_change", "model")
sys.path.insert(0, os.path.join(ROOT, "src"))

from exp_ctx_mask import FeatMaskNet  # noqa: E402
from masknet_v2 import CnnEncoder, WL6, band_weights  # noqa: E402
from onnx_opt import fold_stem_  # noqa: E402


class EncWrap(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.register_buffer("wl", band_weights(WL6))

    def forward(self, vol):                       # (B,6,32,32) raw reflectance
        v = torch.clamp(vol, 0.0, 1.5)
        return self.enc(v[:, None], self.wl)[:, 0]


def main():
    os.makedirs(PLUGIN_MODEL, exist_ok=True)
    import onnxruntime as ort

    # ---------------- encoder (folded stem)
    enc = CnnEncoder()
    enc.load_state_dict(torch.load(os.path.join(EVAL, "encoder_dofa_distilled.pth"),
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
    torch.onnx.export(ew, (x,), os.path.join(PLUGIN_MODEL, "masknet_encoder.onnx"),
                      input_names=["vol"], output_names=["feats"], opset_version=17,
                      dynamic_axes={"vol": {0: "B"}}, dynamo=False)
    print(f"[onnx] masknet_encoder.onnx: "
          f"{os.path.getsize(os.path.join(PLUGIN_MODEL, 'masknet_encoder.onnx')) / 1e6:.1f} MB",
          flush=True)
    sess = ort.InferenceSession(os.path.join(PLUGIN_MODEL, "masknet_encoder.onnx"),
                                providers=["CPUExecutionProvider"])
    out = sess.run(None, {"vol": x.numpy()})[0]
    print(f"[onnx] enc  max|diff| = {np.abs(out - ref).max():.2e}", flush=True)

    # ---------------- length-aware CfC (dynamic T)
    m = FeatMaskNet("conv", 3, ctx_width=256)
    m.load_state_dict(torch.load(os.path.join(OUT, "mask_ctx_conv3_len.pth"),
                                 map_location="cpu"))
    m.eval()
    sm = torch.jit.script(m)

    def mk(B, T, lens=None):
        f = torch.randn(B, T, 256) * 0.3
        c = torch.randn(B, T, 3, 3, 256) * 0.1
        v = torch.ones(B, T, 3, 3)
        d = torch.zeros(B, T)
        d[:, 1:] = 0.7
        do = torch.randn(B, T, 3) * 0.1
        L = (torch.full((B,), T, dtype=torch.int64) if lens is None
             else torch.tensor(lens, dtype=torch.int64))
        return f, c, v, d, do, L

    path = os.path.join(PLUGIN_MODEL, "masknet_cfc.onnx")
    args = mk(2, 3)
    torch.onnx.export(sm, args, path,
                      input_names=["feats", "ctx", "valid", "dt", "doy", "lengths"],
                      output_names=["logits"], opset_version=17, do_constant_folding=True,
                      dynamic_axes={"feats": {0: "B", 1: "T"}, "ctx": {0: "B", 1: "T"},
                                    "valid": {0: "B", 1: "T"}, "dt": {0: "B", 1: "T"},
                                    "doy": {0: "B", 1: "T"}, "lengths": {0: "B"}},
                      dynamo=False)
    print(f"[onnx] masknet_cfc.onnx: {os.path.getsize(path) / 1e6:.1f} MB (dynamic T)",
          flush=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for T, lens in ((2, None), (3, None), (4, None), (8, None), (5, [5, 3])):
        B = 2 if lens is None else 2
        f, c, v, d, do, L = mk(B, T, lens)
        with torch.no_grad():
            r = sm(f, c, v, d, do, L).numpy()
        o = sess.run(None, {"feats": f.numpy(), "ctx": c.numpy(), "valid": v.numpy(),
                            "dt": d.numpy(), "doy": do.numpy(), "lengths": L.numpy()})[0]
        print(f"[onnx] cfc T={T} lens={lens}: max|diff| = {np.abs(o - r).max():.2e}",
              flush=True)


if __name__ == "__main__":
    main()
