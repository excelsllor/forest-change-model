"""Export the two new models to ONNX and verify against PyTorch.

1) mask  : MaskNetV2(bicfcpn, cnn)         vol(B,8,6,32,32) + dt(B,8) + doy(B,8,3) -> logits(B,2)
2) encoder: CnnDofaLike 12x768 (dilated)   vol(B,6,32,32) -> feats(B,768)
3) typenet: TypeNetFeats 2x256             feats(B,8,768) + dt(B,8) + doy(B,8,3) -> logits(B,3)

Outputs into qgis_plugin/forest_change/model/: masknet_v2_cfc.onnx, typenet_encoder.onnx,
typenet_cfc.onnx. Verification on real val samples with onnxruntime CPU.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = r"C:\Users\artyom\Desktop\лесной хакатон"
EVAL = os.path.join(ROOT, "outputs", "vozh", "mask_eval")
PLUGIN_MODEL = os.path.join(ROOT, "qgis_plugin", "forest_change", "model")
sys.path.insert(0, os.path.join(ROOT, "src"))

from masknet_v2 import MaskNetV2, WL6, band_weights  # noqa: E402
from typenet_v3 import CnnDofaLike  # noqa: E402
from typenet_big import TypeNetFeats  # noqa: E402
from onnx_opt import fold_stem_  # noqa: E402

MASK_T = 8


class MaskWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("wl", band_weights(WL6))

    def forward(self, vol, dt, doy):
        return self.m(vol, self.wl, dt, doy)


class EncWrap(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.register_buffer("wl", band_weights(WL6))

    def forward(self, vol):                       # (B,6,32,32) raw reflectance
        v = torch.clamp(vol, 0.0, 1.5)
        return self.enc(v[:, None], self.wl)[:, 0]


class CfCWrap(nn.Module):
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
    import json
    os.makedirs(PLUGIN_MODEL, exist_ok=True)
    dev = torch.device("cpu")

    # ---------------- 1) mask
    with open(os.path.join(EVAL, "masknet_v2_bicfcpn_big.json"), encoding="utf-8") as fh:
        mask_meta = json.load(fh)
    mask_hw = int(mask_meta.get("head_width", 64))
    mask_hm = mask_meta.get("head_mode", "plain")
    print(f"[onnx] mask head {mask_hm}/{mask_hw}", flush=True)
    m = MaskNetV2(seq="bicfcpn", enc="cnn", head_width=mask_hw, head_mode=mask_hm)
    m.load_state_dict(torch.load(os.path.join(EVAL, "masknet_v2_bicfcpn_big.pth"),
                                 map_location="cpu"))
    m.eval()
    mw = MaskWrap(m).eval()
    vol = torch.randn(2, MASK_T, 6, 32, 32) * 0.3
    dt = torch.zeros(2, MASK_T)
    dt[:, 1] = 0.7
    doy = torch.randn(2, MASK_T, 3) * 0.3
    with torch.no_grad():
        ref = mw(vol, dt, doy).numpy()
    export(mw, (vol, dt, doy), os.path.join(PLUGIN_MODEL, "masknet_v2_cfc.onnx"),
           ["vol", "dt", "doy"], {"vol": {0: "B"}, "dt": {0: "B"}, "doy": {0: "B"}})
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(PLUGIN_MODEL, "masknet_v2_cfc.onnx"),
                                providers=["CPUExecutionProvider"])
    out = sess.run(None, {"vol": vol.numpy(), "dt": dt.numpy(), "doy": doy.numpy()})[0]
    print(f"[onnx] mask  max|diff| = {np.abs(out - ref).max():.2e}", flush=True)

    # ---------------- 2) encoder
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

    # ---------------- 3) typenet CfC
    with open(os.path.join(EVAL, "typenet_big_w256_b2.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    head_width = int(meta.get("head_width", 64))
    head_mode = meta.get("head_mode", "plain")
    print(f"[onnx] typenet head {head_mode}/{head_width}", flush=True)
    tm = TypeNetFeats(cfc_width=256, blocks=2, head_width=head_width, head_mode=head_mode)
    tm.load_state_dict(torch.load(os.path.join(EVAL, "typenet_big_w256_b2.pth"),
                                  map_location="cpu"))
    tm.eval()
    tw = torch.jit.script(tm)
    feats = torch.randn(2, 3, 768) * 0.5
    dt = torch.zeros(2, 3)
    dt[:, 1] = 0.7
    dt[:, 2] = 1.1
    doy = torch.randn(2, 3, 3) * 0.3
    with torch.no_grad():
        ref = tw(feats, dt, doy).numpy()
    torch.onnx.export(tw, (feats, dt, doy), os.path.join(PLUGIN_MODEL, "typenet_cfc.onnx"),
                      input_names=["feats", "dt", "doy"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True, dynamo=False,
                      dynamic_axes={"feats": {0: "B", 1: "T"}, "dt": {0: "B", 1: "T"},
                                    "doy": {0: "B", 1: "T"}})
    print(f"[onnx] exported typenet_cfc.onnx "
          f"({os.path.getsize(os.path.join(PLUGIN_MODEL, 'typenet_cfc.onnx')) / 1e6:.1f} MB)",
          flush=True)
    sess = ort.InferenceSession(os.path.join(PLUGIN_MODEL, "typenet_cfc.onnx"),
                                providers=["CPUExecutionProvider"])
    for TT in (3, 8):
        f2 = torch.randn(2, TT, 768) * 0.5
        d2 = torch.zeros(2, TT); d2[:, 1:] = 0.5
        do2 = torch.randn(2, TT, 3) * 0.3
        with torch.no_grad():
            r2 = tw(f2, d2, do2).numpy()
        o2 = sess.run(None, {"feats": f2.numpy(), "dt": d2.numpy(), "doy": do2.numpy()})[0]
        print(f"[onnx] cfc T={TT} max|diff| = {np.abs(o2 - r2).max():.2e}", flush=True)


if __name__ == "__main__":
    main()
