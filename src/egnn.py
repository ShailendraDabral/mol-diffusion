"""E(3)-equivariant denoising network (EGNN, Satorras et al. 2021), dense formulation.

Molecules have at most 29 atoms, so we use full (B, N, N) pairwise tensors instead of
sparse edge lists. This is simple, fast enough on a GPU, and easy to read.

Equivariance: the network only sees pairwise *squared distances* (rotation/translation
invariant) and moves coordinates along *relative vectors* (rotation equivariant).
Atom order does not matter either, because messages are summed over neighbours.
"""
import torch
import torch.nn as nn

from src.data import N_TYPES
from src.utils import center_masked


class EGNNLayer(nn.Module):
    def __init__(self, nf: int, coords_range: float = 15.0, norm_factor: float = 1.0):
        super().__init__()
        self.coords_range = coords_range
        self.norm_factor = norm_factor
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * nf + 1, nf), nn.SiLU(), nn.Linear(nf, nf), nn.SiLU()
        )
        self.att = nn.Sequential(nn.Linear(nf, 1), nn.Sigmoid())
        self.coord_mlp = nn.Sequential(
            nn.Linear(nf, nf), nn.SiLU(), nn.Linear(nf, 1, bias=False)
        )
        # start with tiny coordinate updates for stable training
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)
        self.node_mlp = nn.Sequential(nn.Linear(2 * nf, nf), nn.SiLU(), nn.Linear(nf, nf))

    def forward(self, h, x, mask):
        # h: (B,N,nf)  x: (B,N,3)  mask: (B,N)
        B, N, _ = h.shape
        rel = x.unsqueeze(2) - x.unsqueeze(1)               # (B,N,N,3)  x_i - x_j
        d2 = (rel ** 2).sum(-1, keepdim=True)               # (B,N,N,1)  invariant
        hi = h.unsqueeze(2).expand(B, N, N, -1)
        hj = h.unsqueeze(1).expand(B, N, N, -1)

        eye = torch.eye(N, device=h.device).unsqueeze(0)    # no self-edges
        emask = (mask.unsqueeze(2) * mask.unsqueeze(1) * (1 - eye)).unsqueeze(-1)

        m = self.edge_mlp(torch.cat([hi, hj, d2], dim=-1))
        m = m * self.att(m) * emask                          # (B,N,N,nf)

        # coordinate update: move along normalized relative vectors, weights are invariant
        w = torch.tanh(self.coord_mlp(m)) * self.coords_range
        direction = rel / (torch.sqrt(d2 + 1e-8) + 1.0)
        dx = (direction * w * emask).sum(dim=2) / self.norm_factor
        x = (x + dx) * mask.unsqueeze(-1)

        # feature update
        agg = m.sum(dim=2) / self.norm_factor
        h = (h + self.node_mlp(torch.cat([h, agg], dim=-1))) * mask.unsqueeze(-1)
        return h, x


class EGNNDenoiser(nn.Module):
    """Predicts the noise on coordinates (zero-COM) and on atom-type features."""

    def __init__(self, n_types: int = N_TYPES, nf: int = 128, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(n_types + 1, nf)  # +1 for the time input
        self.layers = nn.ModuleList(
            [EGNNLayer(nf, coords_range=15.0 / n_layers) for _ in range(n_layers)]
        )
        self.out_h = nn.Linear(nf, n_types)

    def forward(self, x, h, t, mask):
        """x (B,N,3) noisy coords, h (B,N,T) noisy features, t (B,) in (0,1], mask (B,N)."""
        t_feat = t.view(-1, 1, 1).expand(-1, x.shape[1], 1)
        hh = self.embed(torch.cat([h, t_feat], dim=-1)) * mask.unsqueeze(-1)
        x_in, xx = x, x
        for layer in self.layers:
            hh, xx = layer(hh, xx, mask)
        eps_x = center_masked(xx - x_in, mask)               # keep prediction in zero-COM space
        eps_h = self.out_h(hh) * mask.unsqueeze(-1)
        return eps_x, eps_h
