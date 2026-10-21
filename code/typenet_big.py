"""TypeNet (big): DOFA-distilled CNN encoder (12 layers x 768, frozen) + 2 bicfcpn blocks.

Encoder: CnnDofaLike - lambda-generated per-band stem (any C) + 12 depthwise-separable
layers of width 768 with growing dilations (RF covers the whole 32x32 chip), GAP -> (B,T,768).
Sequence: CfCBiStack(blocks=2) - two bicfcpn blocks (pre-norm + residual + bidirectional CfC).
Inputs : vol (B,T,C,32,32) raw reflectance + temporal vector (sin DOY, cos DOY, year-2015, dt)
Output : logits (B,3)  [logging / windfall / fire]
"""

import math

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


class AttnBlock(nn.Module):
    """Cross-attention block: learned queries attend to the per-date tokens."""
    def __init__(self, dim, heads=8, ffn=1024):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.nq = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim)
        self.qp = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.o = nn.Linear(dim, dim)
        self.n1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))
        self.n2 = nn.LayerNorm(dim)

    def forward(self, q, x, valid):
        b, nq = q.shape[:2]
        t = x.shape[1]
        qq = self.qp(self.nq(q)).reshape(b, nq, self.h, self.dh).transpose(1, 2)
        kv = self.kv(self.nkv(x)).reshape(b, t, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        scores = torch.matmul(qq, kv[0].transpose(-1, -2)) / math.sqrt(self.dh)
        scores = torch.where(valid[:, None, None, :], scores, torch.full_like(scores, -1e9))
        w = torch.softmax(scores, dim=-1)
        out = self.o(torch.matmul(w, kv[1]).transpose(1, 2).reshape(b, nq, -1))
        q = q + out
        q = q + self.ffn(self.n1(q))
        return self.n2(q)


class AttnReadout(nn.Module):
    """Temporal cross-attention readout: 2 query tokens attend to the per-date features.

    Same output width as CfCBiStack (2*width), so the head and summaries stay unchanged.
    """
    def __init__(self, d_in, width=256, heads=8, ffn=1024, n_queries=2, max_t=64):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.slot = nn.Parameter(torch.zeros(max_t, width))
        nn.init.normal_(self.slot, std=0.02)
        self.q = nn.Parameter(torch.randn(n_queries, width) * 0.02)
        self.blocks = nn.ModuleList([AttnBlock(width, heads, ffn) for _ in range(2)])

    def forward(self, x, valid):
        b, t = x.shape[:2]
        pos = torch.arange(t, device=x.device).clamp(max=self.slot.shape[0] - 1)
        h = self.inp(x) + self.slot[pos].unsqueeze(0)
        m = valid.unsqueeze(-1).float()
        mean = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        q = self.q.unsqueeze(0) + mean.unsqueeze(1)
        for blk in self.blocks:
            q = blk(q, h, valid)
        return q.reshape(b, -1)


class TypeNetFeats(nn.Module):
    """TypeNet (readout only): cached per-date features (B,T,768) + temporal vector -> readout -> head.

    readout="cfc"  : CfCBiStack(blocks=2, width 256) - two bicfcpn blocks
                     (pre-norm + residual + bidirectional CfC, per-series dt normalisation).
    readout="attn" : 2 learned query tokens cross-attend to the per-date tokens
                     (masked by valid dates), same 2*width output.
    head_mode="summary" additionally feeds [mean_T(feats), last - first] (2*768) to the head.
    Inputs : feats (B,T,768), dt (B,T), doy (B,T,3)
    Output : logits (B,3)  [logging / windfall / fire]
    """

    def __init__(self, cfc_width=256, blocks=2, nc=3, t_dim=4, d_in=768, head_width=64,
                 head_mode="plain", use_spec=False, readout="cfc"):
        super().__init__()
        self.t_dim = t_dim
        self.head_mode = head_mode
        self.use_spec = use_spec
        self.readout = readout
        d_seq_in = d_in + t_dim + (9 if use_spec else 0)
        if readout == "attn":
            self.seq = AttnReadout(d_seq_in, cfc_width)
        else:
            self.seq = CfCBiStack(d_seq_in, cfc_width, blocks=blocks)
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
        if self.readout == "attn":
            L = (dt > 0).sum(dim=1) + 1
            valid = torch.arange(x.shape[1], device=x.device)[None] < L[:, None]
            h = self.seq(x, valid)
        else:
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
