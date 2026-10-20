"""ONNX export helpers: fold the lambda-generated per-band stem into one dense conv."""
import torch
import torch.nn as nn


def fold_stem_(enc, wl):
    """Replace BandConvStem (6 per-band convs) with one dense Conv2d(C->out, k3).

    BandConvStem computes y = sum_c conv2d(x_c, w_c); this equals a dense conv with
    weight[o, c] = w_c[o]. Weights are materialised from the stem MLP and kept in fp32.
    """
    stem = enc.stem
    C, out, k = len(wl), stem.out_ch, stem.k
    with torch.no_grad():
        wl_t = torch.tensor(list(wl), dtype=torch.float32).reshape(-1, 1)
        w = stem.net(wl_t)                                  # (C, out*k*k)
        W = w.reshape(C, out, k, k).permute(1, 0, 2, 3).contiguous()
    conv = nn.Conv2d(C, out, k, padding=k // 2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(W)

    class FoldedStem(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv = c

        def forward(self, x, wl=None, alive=None):
            return self.conv(x)

    enc.stem = FoldedStem(conv)
    return enc


def gelu_tanh_(m):
    """Switch all GELUs to the tanh approximation (one Tanh instead of Erf+arithmetic)."""
    for name, child in m.named_children():
        if isinstance(child, nn.GELU):
            setattr(m, name, nn.GELU(approximate="tanh"))
        else:
            gelu_tanh_(child)
    return m
