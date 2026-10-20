"""MaskNet-v2 (latest architecture, 2026-10-18).

Pipeline:
  vol (B,T,C,32,32) reflectance -> CnnEncoder (channel-agnostic, any C) -> (B,T,256)
  + per-date temporal vector (sin DOY, cos DOY, [year-2015], dt) -> (B,T,t_dim)
  -> CfCBiEncoderPN: pre-norm + residual + BIDIRECTIONAL CfC with dt normalisation
  -> head: LayerNorm -> Linear(64) -> GELU -> Dropout(0.2) -> Linear(2), bias [1, -1]

Encoder (CnnEncoder):
  BandConvStem - per-band 3x3 kernels generated from the band wavelengths (any C),
  32x32 -> stem(48) -> 16x16(96) -> 8x8(192) -> 4x4(256) -> 2x2(dim) -> GAP -> (B,T,dim).
  stem_robust=True adds the recipe stem normalisation (sum / sqrt(alive channels) + GroupNorm)
  and supports channel dropout via the per-sample alive mask.

Sequence (seq modes, all bidirectional CfC with pre-norm + residual):
  "bicfcpn"  - dt / mean positive gap of the series (per-series);
  "bicfcpn2" - dt normalised as log1p(days / 30);
  "bicfcpng" - dt / global mean gap (buffer);
  "bicfcpn0" - raw dt in years.

Inputs : vol (B,T,C,32,32), wl (C,), dt (B,T), doy (B,T,t_dim-1), alive (B,C) optional
Output : logits (B,2)  [1 = change in the 32x32 tile]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

WL6 = [0.4966, 0.5600, 0.6645, 0.8351, 1.6137, 2.2024]   # B02,B03,B04,B08,B11,B12
ENC_DIM = 256
SEQ_DIM = 128
T_DIM = 4          # temporal vector size (sin, cos, year, dt); t_dim=3 drops the year


def band_weights(wl):
    return torch.tensor(list(wl), dtype=torch.float32)


# ---------------------------------------------------------------- encoder
class BandConvStem(nn.Module):
    """Channel-agnostic CNN stem: per-band 3x3 kernels generated from wavelengths.

    w_c = MLP(lambda_c) -> (out_ch, 1, k, k);   y = sum_c conv2d(x_c, w_c)
    (any number of bands C; the kernel weights are generated, not stored).

    robust=True: y = sum_c conv(x_c, w_c) * alive_c / sqrt(alive_count), followed by
    GroupNorm - survives channel dropout and missing bands.
    """

    def __init__(self, out_ch=48, k=3, emb=64, robust=False):
        super().__init__()
        self.out_ch, self.k = out_ch, k
        self.net = nn.Sequential(nn.Linear(1, emb), nn.GELU(), nn.Linear(emb, emb), nn.GELU(),
                                 nn.Linear(emb, out_ch * k * k))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
        self.robust = robust
        self.norm = nn.GroupNorm(8, out_ch) if robust else None

    def forward(self, x, wl, alive=None):
        """x (B,C,32,32), wl (C,), alive (B,C) 1/0 per channel (optional)."""
        C = x.shape[1]
        w = self.net(wl.reshape(-1, 1)).reshape(C, self.out_ch, 1, self.k, self.k)
        out = 0.0
        for c in range(C):
            contrib = F.conv2d(x[:, c:c + 1], w[c], padding=self.k // 2)
            if alive is not None:
                contrib = contrib * alive[:, c].reshape(-1, 1, 1, 1)
            out = out + contrib
        if self.robust:
            if alive is not None:
                n_alive = alive.sum(dim=1, keepdim=True).clamp(min=1.0)
            else:
                n_alive = torch.full((x.shape[0], 1), float(C), device=x.device)
            out = out / n_alive.sqrt().reshape(-1, 1, 1, 1)
            out = self.norm(out)
        return out


class ConvBlock(nn.Module):
    """Conv3x3(stride 2, optional dilation) + GroupNorm + GELU, with a residual shortcut."""

    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=2, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1, stride=2)   # always downsamples (stride 2)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)) + self.skip(x))


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (channels-first adaptation for ROCm): 7x7 depthwise -> GroupNorm ->
    1x1 expand -> GELU -> 1x1 project -> layer-scale residual."""

    def __init__(self, dim, mlp_ratio=4, layer_scale=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(8, dim)
        self.pw1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):
        h = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + h * self.gamma.reshape(1, -1, 1, 1)


class CnnEncoderNext(nn.Module):
    """Small ConvNeXt encoder (same layout as CnnEncoder): lambda stem + 4 ConvNeXt stages.

    32x32 -> stem(48) -> 16x16(96) -> 8x8(192) -> 4x4(256) -> 2x2(256) -> GAP -> (B,T,256)
    Each stage: Conv2x2(stride 2) + GroupNorm + ConvNeXtBlock.
    """

    def __init__(self, dim=ENC_DIM, stem=48, widths=(96, 192, 256, 256), mlp_ratio=4):
        super().__init__()
        self.stem = BandConvStem(stem)
        self.stages = nn.ModuleList()
        cin = stem
        for cout in widths:
            self.stages.append(nn.Sequential(
                nn.Conv2d(cin, cout, 2, stride=2), nn.GroupNorm(8, cout),
                ConvNeXtBlock(cout, mlp_ratio)))
            cin = cout
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, vol, wl):
        B, T = vol.shape[0], vol.shape[1]
        x = vol.reshape(B * T, vol.shape[2], 32, 32)
        h = self.stem(x, wl)
        for st in self.stages:
            h = st(h)
        return self.pool(h).flatten(1).reshape(B, T, -1)


class CnnEncoder(nn.Module):
    """Channel-agnostic CNN encoder (any C via the lambda-generated stem, any T).

    32x32 -> stem(48) -> 16x16(96) -> 8x8(192) -> 4x4(256) -> 2x2(dim) -> GAP -> (B,T,dim)
    Receptive field with plain 3x3 convs is already 63 px (> the whole 32 px chip).
    NOTE: dilations>1 on stride-2 convs hit a pathological MIOpen kernel on RDNA1
    (~220x slower: 211 ms vs 0.96 ms) - keep dilations=(1,1,1,1) for training speed.
    """

    def __init__(self, dim=ENC_DIM, stem=48, widths=(96, 192, 256), stem_robust=False,
                 dilations=(1, 1, 1, 1)):
        super().__init__()
        self.stem = BandConvStem(stem, robust=stem_robust)
        self.blocks = nn.Sequential(
            ConvBlock(stem, widths[0], dilations[0]),
            ConvBlock(widths[0], widths[1], dilations[1]),
            ConvBlock(widths[1], widths[2], dilations[2]),
            ConvBlock(widths[2], dim, dilations[3]))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, vol, wl, alive=None):
        B, T = vol.shape[0], vol.shape[1]
        C = vol.shape[2]
        x = vol.reshape(B * T, C, 32, 32)
        if alive is not None:
            x = x * alive.unsqueeze(1).expand(B, T, C).reshape(B * T, C, 1, 1)
        h = self.blocks(self.stem(x, wl, None if alive is None else
                                  alive.unsqueeze(1).expand(B, T, C).reshape(B * T, C)))
        return self.pool(h).flatten(1).reshape(B, T, -1)


# ---------------------------------------------------------------- full model
class MaskNetV2(nn.Module):
    def __init__(self, seq="bicfcpn", dim=ENC_DIM, cfc_hidden=128, enc="cnn",
                 stem_robust=False, t_dim=T_DIM, head_width=64, head_mode="plain"):
        super().__init__()
        from cfc import CfCBiEncoderPN
        modes = {"bicfcpn": "seq", "bicfcpn2": "log1p", "bicfcpng": "global", "bicfcpn0": "none"}
        if seq not in modes:
            raise ValueError(f"unknown seq: {seq}")
        if enc != "cnn":
            raise ValueError(f"unknown encoder: {enc}")
        self.seq_kind = seq
        self.enc_kind = enc
        self.encoder = CnnEncoder(dim, stem_robust=stem_robust)
        self.t_dim = t_dim
        self.head_mode = head_mode
        d_in = dim + t_dim
        self.seq = CfCBiEncoderPN(d_in, cfc_hidden, backbone_units=cfc_hidden,
                                  backbone_layers=2, dt_mode=modes[seq])
        out_dim = 2 * cfc_hidden
        h_in = out_dim + (2 * dim if head_mode == "summary" else 0)
        self.head = nn.Sequential(nn.LayerNorm(h_in), nn.Linear(h_in, head_width), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(head_width, 2))
        with torch.no_grad():                       # start biased to "no change"
            self.head[-1].bias.copy_(torch.tensor([1.0, -1.0]))

    def forward(self, vol, wl, dt, doy, alive=None):
        f = self.encoder(vol, wl, alive)                                  # (B,T,dim)
        tv = torch.cat([doy, dt.unsqueeze(-1)], dim=-1)                   # (B,T,t_dim)
        x = torch.cat([f, tv], dim=-1)                                    # (B,T,dim+t_dim)
        h = self.seq(x, dt)
        if self.head_mode == "summary":
            b = f.shape[0]
            idx = (dt > 0).sum(dim=1)                                     # last valid date index
            rows = torch.arange(b, device=f.device)
            n = (idx + 1).unsqueeze(1).clamp(min=1)
            mean = f.sum(dim=1) / n
            last = f[rows, idx]
            h = torch.cat([h, mean, last - f[:, 0]], dim=-1)
        return self.head(h)
