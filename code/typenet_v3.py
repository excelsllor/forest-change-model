"""TypeNet-v3: encoder (DOFA features | CNN DOFA-like) + 2-layer CfC + decoder.

Sequence: TWO stacked CfC layers of the same width (256 / 512 / 768), real dt.
Encoder:
  "dofa" : consumes cached DOFA-base features (B,T,768) + spectral (B,T,9)
  "cnn"  : DOFA-like channel-agnostic CNN (12 depthwise-separable layers, width 768)
           over raw volumes (B,T,C,32,32) + the same spectral/date features
Decoder: LayerNorm -> Linear(h->256) -> GELU -> Dropout -> Linear(256->4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

WL6 = [0.4966, 0.5600, 0.6645, 0.8351, 1.6137, 2.2024]
T_DIM = 4


def band_weights(wl):
    return torch.tensor(list(wl), dtype=torch.float32)


class CnnDofaLike(nn.Module):
    """DOFA-like CNN: lambda-generated stem + 12 depthwise-separable layers (width 768).

    Depthwise kernels use growing dilations ([1,2,4,8]*3) so the receptive field covers
    the whole 32x32 chip (RF ~185 px); gradient checkpointing keeps it within 8 GB VRAM.
    """

    def __init__(self, dim=768, stem=48, layers=12, ckpt=True, dilations=None):
        super().__init__()
        from masknet_v2 import BandConvStem
        if dilations is None:
            dilations = [1, 2, 4, 8] * 3
        self.stem = BandConvStem(stem)
        self.proj = nn.Conv2d(stem, dim, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([self._block(dim, d) for d in dilations[:layers]])
        self.norm = nn.GroupNorm(8, dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ckpt = ckpt

    @staticmethod
    def _block(dim, dilation=1):
        return nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1), nn.GroupNorm(8, dim))

    def forward(self, vol, wl):
        B, T = vol.shape[0], vol.shape[1]
        x = vol.reshape(B * T, vol.shape[2], 32, 32)
        h = F.gelu(self.proj(self.stem(x, wl)))               # (B*T, dim, 16, 16)
        for blk in self.blocks:
            if self.ckpt and self.training and torch.is_grad_enabled():
                h = F.gelu(h + checkpoint(blk, h, use_reentrant=False))
            else:
                h = F.gelu(h + blk(h))                        # residual, any depth
        return self.pool(self.norm(h)).flatten(1).reshape(B, T, -1)


class CnnDofaLikeNext(nn.Module):
    """Big ConvNeXt encoder: lambda stem + 12 ConvNeXt blocks of width 768 (DOFA-like).

    Same layout as CnnDofaLike (stem -> proj stride 2 -> 12 blocks -> GN -> GAP),
    blocks replaced by ConvNeXt (7x7 dw -> LN -> inverted bottleneck -> layer scale).
    """

    def __init__(self, dim=768, stem=48, layers=12, mlp_ratio=1, ckpt=True):
        super().__init__()
        from masknet_v2 import BandConvStem, ConvNeXtBlock
        self.stem = BandConvStem(stem)
        self.proj = nn.Conv2d(stem, dim, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, mlp_ratio) for _ in range(layers)])
        self.norm = nn.GroupNorm(8, dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ckpt = ckpt

    def forward(self, vol, wl):
        B, T = vol.shape[0], vol.shape[1]
        x = vol.reshape(B * T, vol.shape[2], 32, 32)
        h = F.gelu(self.proj(self.stem(x, wl)))               # (B*T, dim, 16, 16)
        for blk in self.blocks:
            if self.ckpt and self.training and torch.is_grad_enabled():
                h = checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
        return self.pool(self.norm(h)).flatten(1).reshape(B, T, -1)


class CfCStack(nn.Module):
    """Two (or more) stacked CfC layers; the per-step sequence is passed between them,
    every hop followed by LayerNorm + Linear (post-norm projection)."""

    def __init__(self, d_in, h, layers=2, backbone=2):
        super().__init__()
        from cfc import CfCCell
        self.cells = nn.ModuleList([CfCCell(d_in, h, backbone_units=h, backbone_layers=backbone)] +
                                   [CfCCell(h, h, backbone_units=h, backbone_layers=backbone)
                                    for _ in range(max(0, layers - 1))])
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
                                   for _ in range(max(0, layers - 1))])

    def forward(self, x, dt):
        for li, cell in enumerate(self.cells):
            b, t, _ = x.shape
            h = x.new_zeros(b, cell.hidden_size)
            outs = []
            for step in range(t):
                h = cell(x[:, step], h, dt[:, step:step + 1])
                outs.append(h)
            x = torch.stack(outs, dim=1)                 # (B,T,h) -> next layer
            if li < len(self.proj):
                x = self.proj[li](x)                     # LayerNorm + Linear between layers
        return x.mean(dim=1)


class TypeNetV3(nn.Module):
    def __init__(self, enc="dofa", cfc_width=256, cfc_layers=2, nc=4, cnn_dim=768, cnn_layers=12):
        super().__init__()
        self.enc_kind = enc
        if enc == "dofa":
            self.encoder = None
            d_in = 768 + 9 + T_DIM
        else:
            self.encoder = CnnDofaLike(cnn_dim, layers=cnn_layers)
            d_in = cnn_dim + 9 + T_DIM
        h = cfc_width
        self.seq = CfCStack(d_in, h, layers=cfc_layers)
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, nc))

    def forward(self, feats, spec, dt, doy, vol=None, wl=None):
        if self.enc_kind == "dofa":
            f = feats                                            # (B,T,768) cached
        else:
            f = self.encoder(vol, wl)                            # (B,T,768)
        x = torch.cat([f, spec, doy, dt.unsqueeze(-1)], dim=-1)
        return self.head(self.seq(x, dt))
