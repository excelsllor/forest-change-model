"""TypeNet (big): DOFA-distilled CNN encoder (12 layers x 768, frozen) + 2 bicfcpn blocks.

Encoder: CnnDofaLike - lambda-generated per-band stem (any C) + 12 depthwise-separable
layers of width 768 with growing dilations (RF covers the whole 32x32 chip), GAP -> (B,T,768).
Sequence: CfCBiStack(blocks=2) - two bicfcpn blocks (pre-norm + residual + bidirectional CfC).
Inputs : vol (B,T,C,32,32) raw reflectance + temporal vector (sin DOY, cos DOY, year-2015, dt)
Output : logits (B,3)  [logging / windfall / fire]
"""

import torch
import torch.nn as nn

from cfc import CfCBiStack
from typenet_v3 import CnnDofaLike


class TypeNetBig(nn.Module):
    def __init__(self, cfc_width=512, blocks=2, nc=3, t_dim=4, cnn_dim=768, cnn_layers=12):
        super().__init__()
        self.encoder = CnnDofaLike(cnn_dim, layers=cnn_layers)
        self.t_dim = t_dim
        self.seq = CfCBiStack(cnn_dim + t_dim, cfc_width, blocks=blocks)
        out_dim = 2 * cfc_width
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, 64), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(64, nc))

    def forward(self, vol, wl, dt, doy):
        f = self.encoder(vol, wl)                                   # (B,T,768)
        tv = torch.cat([doy, dt.unsqueeze(-1)], dim=-1)             # (B,T,4)
        x = torch.cat([f, tv], dim=-1)                              # (B,T,772)
        return self.head(self.seq(x, dt))


class TypeNetFeats(nn.Module):
    """TypeNet (CfC only): cached per-date features (B,T,768) + temporal vector -> CfC stack -> head.

    Sequence: CfCBiStack(blocks=2, width 256) - two bicfcpn blocks
    (pre-norm + residual + bidirectional CfC, per-series dt normalisation).
    head_mode="summary" additionally feeds [mean_T(feats), last - first] (2*768) to the head.
    Inputs : feats (B,T,768), dt (B,T), doy (B,T,3)
    Output : logits (B,3)  [logging / windfall / fire]
    """

    def __init__(self, cfc_width=256, blocks=2, nc=3, t_dim=4, d_in=768, head_width=64,
                 head_mode="plain", use_spec=False):
        super().__init__()
        self.t_dim = t_dim
        self.head_mode = head_mode
        self.use_spec = use_spec
        self.seq = CfCBiStack(d_in + t_dim + (9 if use_spec else 0), cfc_width, blocks=blocks)
        out_dim = 2 * cfc_width
        h_in = out_dim + (2 * d_in if head_mode == "summary" else 0) + (18 if use_spec else 0)
        self.head = nn.Sequential(nn.LayerNorm(h_in), nn.Linear(h_in, head_width), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(head_width, nc))

    def forward(self, feats, dt, doy, spec=None):
        tv = torch.cat([doy, dt.unsqueeze(-1)], dim=-1)             # (B,T,4)
        parts = [feats]
        if self.use_spec:
            parts.append(spec)
        parts.append(tv)
        x = torch.cat(parts, dim=-1)
        h = self.seq(x, dt)
        if self.head_mode == "summary":
            b = feats.shape[0]
            idx = (dt > 0).sum(dim=1)                               # last valid date index
            rows = torch.arange(b, device=feats.device)
            n = (idx + 1).unsqueeze(1).clamp(min=1)
            mean = feats.sum(dim=1) / n
            last = feats[rows, idx]
            h = torch.cat([h, mean, last - feats[:, 0]], dim=-1)
            if self.use_spec:
                h = torch.cat([h, spec.sum(dim=1) / n,
                               spec[rows, idx] - spec[:, 0]], dim=-1)
        return self.head(h)
