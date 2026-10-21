"""Closed-form Continuous-time (CfC) cell and encoder in pure PyTorch.

CfC interpolates between two feed-forward states with a time-dependent gate,
which lets it handle irregularly sampled sequences (cloud gaps, uneven revisit
intervals) without resampling. Reference: Hasani et al., "Closed-form
Continuous-time Neural Networks", Nature Machine Intelligence, 2022.

The temporal encoder here is designed to be fused with a spatial backbone
(e.g. ChangeMamba) or used standalone on per-object spectral time series.
"""

import torch
import torch.nn as nn


class CfCCell(nn.Module):
    def __init__(self, input_size, hidden_size, backbone_units=64, backbone_layers=2):
        super().__init__()
        layers = []
        in_dim = input_size + hidden_size
        for _ in range(backbone_layers):
            layers += [nn.Linear(in_dim, backbone_units), nn.LeakyReLU(0.1)]
            in_dim = backbone_units
        self.backbone = nn.Sequential(*layers)
        self.ff1 = nn.Linear(backbone_units, hidden_size)
        self.ff2 = nn.Linear(backbone_units, hidden_size)
        self.time_a = nn.Linear(backbone_units, hidden_size)
        self.time_b = nn.Linear(backbone_units, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, h, dt):
        """x: (B, input), h: (B, hidden), dt: (B, 1) elapsed time."""
        z = self.backbone(torch.cat([x, h], dim=1))
        gate = torch.sigmoid(self.time_a(z) * dt + self.time_b(z))
        return (1.0 - gate) * self.ff1(z) + gate * self.ff2(z)


class CfCEncoder(nn.Module):
    """Runs a CfCCell over a sequence and returns the last hidden state."""

    def __init__(self, input_size, hidden_size, backbone_units=64, backbone_layers=2, pool="last"):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size, backbone_units, backbone_layers)
        self.hidden_size = hidden_size
        self.pool = pool

    def forward(self, x, dt):
        # x: (B, T, input_size), dt: (B, T) elapsed time since previous step.
        b, t, _ = x.shape
        if dt.dim() == 1:
            dt = dt.unsqueeze(0).expand(b, -1)
        h = x.new_zeros(b, self.hidden_size)
        if self.pool == "mean":
            hs = []
            for step in range(t):
                h = self.cell(x[:, step], h, dt[:, step:step + 1])
                hs.append(h)
            return torch.stack(hs, dim=1).mean(dim=1)
        for step in range(t):
            h = self.cell(x[:, step], h, dt[:, step:step + 1])
        return h


class CfCBiEncoder(nn.Module):
    """Bidirectional CfC: forward + backward cells, per-step states concatenated and pooled.

    The backward pass runs over the reversed sequence; its elapsed times are the same
    date gaps reversed (dt[0] = 0 for the first step of each direction).
    """

    def __init__(self, input_size, hidden_size, backbone_units=64, backbone_layers=2, pool="mean"):
        super().__init__()
        self.fwd = CfCCell(input_size, hidden_size, backbone_units, backbone_layers)
        self.bwd = CfCCell(input_size, hidden_size, backbone_units, backbone_layers)
        self.hidden_size = hidden_size
        self.pool = pool

    def forward(self, x, dt):
        # x: (B, T, input_size), dt: (B, T) elapsed time since previous step.
        b, t, _ = x.shape
        if dt.dim() == 1:
            dt = dt.unsqueeze(0).expand(b, -1)
        hf = x.new_zeros(b, self.hidden_size)
        hs_f = []
        for s in range(t):
            hf = self.fwd(x[:, s], hf, dt[:, s:s + 1])
            hs_f.append(hf)
        hb = x.new_zeros(b, self.hidden_size)
        hs_b = []
        dtr = torch.flip(dt, dims=[1]).clone()
        dtr[:, 0] = 0.0
        for s in range(t):
            hb = self.bwd(x[:, t - 1 - s], hb, dtr[:, s:s + 1])
            hs_b.append(hb)
        hs_b.reverse()                                       # align to forward time order
        h = torch.cat([torch.stack(hs_f, dim=1), torch.stack(hs_b, dim=1)], dim=-1)  # (B,T,2h)
        if self.pool == "last":
            return h[:, -1]
        return h.mean(dim=1)


class CfCBiEncoderPN(nn.Module):
    """Bidirectional CfC with pre-norm, residual and dt normalisation.

    u  = LayerNorm(x)                       (pre-norm)
    xp = in_proj(u)                         (d_in -> hidden)
    h_d = CfC_d(xp, dt_norm)                (forward and backward over time)
    y_d = xp + h_d                          (residual around the cell)
    out = mean_T(concat(y_fwd, y_bwd))      (2 * hidden)

    dt_mode:
      "seq"    - dt / mean positive gap of that very sequence (per-series, dt[0]=0);
      "global" - dt / dt_ref (global mean positive gap of the training set, buffer);
      "log1p"  - log1p(days / 30), dt in years (global constant, recipe default);
      "none"   - raw dt in years.
    """

    def __init__(self, input_size, hidden_size, backbone_units=64, backbone_layers=2,
                 dt_mode="seq", n_cont=0, cont_mode="inline"):
        super().__init__()
        self.n_cont = n_cont
        self.cont_mode = cont_mode
        self.n_emb = input_size - n_cont if (n_cont and cont_mode == "sep") else input_size
        self.norm = nn.LayerNorm(self.n_emb)
        if n_cont and cont_mode == "sep":
            self.sc_proj = nn.Sequential(nn.Linear(n_cont, 4 * n_cont), nn.GELU(),
                                         nn.Linear(4 * n_cont, n_cont))
        else:
            self.sc_proj = nn.Identity()
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.fwd = CfCCell(hidden_size, hidden_size, backbone_units, backbone_layers)
        self.bwd = CfCCell(hidden_size, hidden_size, backbone_units, backbone_layers)
        self.hidden_size = hidden_size
        self.dt_mode = dt_mode
        self.register_buffer("dt_ref", torch.tensor(1.0))

    def norm_dt(self, dt):
        if self.dt_mode == "none":
            return dt
        if self.dt_mode == "log1p":
            return torch.log1p(dt * (365.25 / 30.0))          # days / 30, compressed
        if self.dt_mode == "global":
            return dt / self.dt_ref.clamp(min=1e-3)
        nz = (dt > 0).sum(dim=1, keepdim=True).clamp(min=1)
        gap = dt.sum(dim=1, keepdim=True) / nz
        return dt / gap.clamp(min=1e-3)

    def _direction(self, cell: CfCCell, xp: torch.Tensor, dt: torch.Tensor,
                   reverse: bool) -> torch.Tensor:
        b, t, _ = xp.shape
        if reverse:
            dt = torch.flip(dt, dims=[1]).clone()
            dt[:, 0] = 0.0
        h = xp.new_zeros(b, self.hidden_size)
        outs = []
        for s in range(t):
            i = t - 1 - s if reverse else s
            h = cell.forward(xp[:, i], h, dt[:, s:s + 1])
            outs.append(h)
        seq = torch.stack(outs, dim=1)                       # processing order
        return torch.flip(seq, dims=[1]) if reverse else seq

    def forward(self, x, dt, lengths=None):
        if dt.dim() == 1:
            dt = dt.unsqueeze(0).expand(x.shape[0], -1)
        if self.n_cont and self.cont_mode == "sep":
            u = self.norm(x[:, :, :self.n_emb])
            s = self.sc_proj(x[:, :, self.n_emb:])
            xin = torch.cat([u, s], dim=-1)
        else:
            xin = self.norm(x)
        xp = self.in_proj(xin)
        dtn = self.norm_dt(dt)
        if lengths is None:
            yf = xp + self._direction(self.fwd, xp, dtn, False)
            yb = xp + self._direction(self.bwd, xp, dtn, True)
            return torch.cat([yf, yb], dim=-1).mean(dim=1)
        b, t, _ = xp.shape
        rows = torch.arange(b, device=xp.device)
        ln = lengths.clamp(min=1, max=t)
        steps = torch.arange(t, device=xp.device).unsqueeze(0)
        valid = (steps < ln.unsqueeze(1)).float()                      # (B,T)
        h = xp.new_zeros(b, self.hidden_size)
        sum_f = xp.new_zeros(b, self.hidden_size)
        for s in range(t):
            h = self.fwd.forward(xp[:, s], h, dtn[:, s:s + 1])
            sum_f = sum_f + h * valid[:, s:s + 1]
        h = xp.new_zeros(b, self.hidden_size)
        sum_b = xp.new_zeros(b, self.hidden_size)
        for s in range(t):
            idx = (ln - 1 - s).clamp(min=0)                            # last valid date first
            d_idx = (idx + 1).clamp(max=t - 1)
            d_t = dtn[rows, d_idx] * (1.0 if s > 0 else 0.0)
            active = (s < ln).float().unsqueeze(1)
            hn = self.bwd.forward(xp[rows, idx], h, d_t.unsqueeze(1))
            h = h * (1.0 - active) + hn * active
            sum_b = sum_b + h * active
        mean_xp = (xp * valid.unsqueeze(-1)).sum(dim=1) / ln.unsqueeze(1)
        return torch.cat([mean_xp + sum_f / ln.unsqueeze(1),
                          mean_xp + sum_b / ln.unsqueeze(1)], dim=-1)


class CfCBiBlock(nn.Module):
    """One bicfcpn block: pre-norm + in_proj + bidirectional CfC + residual (per-step output).

    u  = LayerNorm(x);  xp = in_proj(u)  (d_in -> h)
    y_d = xp + CfC_d(xp, dt)             (forward and backward over time)
    out = concat(y_f, y_b)               (B, T, 2h)
    """

    def __init__(self, d_in, h, backbone=2, dt_mode="seq"):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.in_proj = nn.Linear(d_in, h)
        self.fwd = CfCCell(h, h, h, backbone)
        self.bwd = CfCCell(h, h, h, backbone)
        self.hidden_size = h
        self.dt_mode = dt_mode
        self.register_buffer("dt_ref", torch.tensor(1.0))

    def norm_dt(self, dt):
        if self.dt_mode == "none":
            return dt
        if self.dt_mode == "log1p":
            return torch.log1p(dt * (365.25 / 30.0))
        if self.dt_mode == "global":
            return dt / self.dt_ref.clamp(min=1e-3)
        nz = (dt > 0).sum(dim=1, keepdim=True).clamp(min=1)
        gap = dt.sum(dim=1, keepdim=True) / nz
        return dt / gap.clamp(min=1e-3)

    def _direction(self, cell: CfCCell, xp: torch.Tensor, dt: torch.Tensor,
                   reverse: bool) -> torch.Tensor:
        b, t, _ = xp.shape
        if reverse:
            dt = torch.flip(dt, dims=[1]).clone()
            dt[:, 0] = 0.0
        h = xp.new_zeros(b, self.hidden_size)
        outs = []
        for s in range(t):
            i = t - 1 - s if reverse else s
            h = cell.forward(xp[:, i], h, dt[:, s:s + 1])
            outs.append(h)
        seq = torch.stack(outs, dim=1)
        return torch.flip(seq, dims=[1]) if reverse else seq

    def forward(self, x, dt):
        if dt.dim() == 1:
            dt = dt.unsqueeze(0).expand(x.shape[0], -1)
        xp = self.in_proj(self.norm(x))
        dtn = self.norm_dt(dt)
        yf = xp + self._direction(self.fwd, xp, dtn, False)
        yb = xp + self._direction(self.bwd, xp, dtn, True)
        return torch.cat([yf, yb], dim=-1)                   # (B,T,2h)


class CfCBiStack(nn.Module):
    """Stack of bicfcpn blocks (pre-norm + residual + bidirectional CfC); mean over T."""

    def __init__(self, d_in, h, blocks=2, backbone=2, dt_mode="seq"):
        super().__init__()
        self.blocks = nn.ModuleList([
            CfCBiBlock(d_in if i == 0 else 2 * h, h, backbone, dt_mode)
            for i in range(blocks)])

    def forward(self, x, dt):
        for blk in self.blocks:
            x = blk(x, dt)
        return x.mean(dim=1)


class CfCClassifier(nn.Module):
    """CfC temporal encoder + linear head for object-level change classification."""

    def __init__(self, input_size, hidden_size=32, num_classes=3, dropout=0.1,
                 backbone_units=64, backbone_layers=2, pool="last"):
        super().__init__()
        self.encoder = CfCEncoder(input_size, hidden_size, backbone_units, backbone_layers, pool=pool)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, dt):
        return self.head(self.encoder(x, dt))
