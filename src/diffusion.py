"""Gaussian diffusion over (coordinates, atom-type features) for 3D molecules.

Same DDPM maths as Project 1, with two molecule-specific changes (EDM, Hoogeboom et al. 2022):
  1. Coordinate noise is sampled in the zero-center-of-mass subspace.
  2. One-hot atom types are scaled by `h_scale` so they have a magnitude comparable to
     coordinates (angstroms).
The network predicts the noise (epsilon) for both parts; the loss is a masked MSE.
"""
import math

import torch
import torch.nn as nn

from src.data import N_TYPES
from src.utils import center_masked, zero_com_noise


def make_betas(T: int, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, T)
    if schedule == "cosine":  # Nichol & Dhariwal, 2021
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        abar = f / f[0]
        betas = 1 - abar[1:] / abar[:-1]
        return betas.clamp(max=0.999).float()
    raise ValueError(schedule)


class GaussianDiffusion(nn.Module):
    def __init__(self, T: int = 1000, schedule: str = "cosine", h_scale: float = 0.25,
                 min_abar: float = 1e-4):
        super().__init__()
        self.T, self.h_scale = T, h_scale
        betas = make_betas(T, schedule)
        # Floor on abar: the raw cosine schedule ends near 1e-9, so the first sampling step
        # divides the network's noise error by sqrt(abar) ~ 1e4 and coordinates blow up
        # (images hide this by clipping x0 to [-1, 1]; molecules have no such clip).
        abar = torch.cumprod(1 - betas, dim=0).clamp(min=min_abar)
        self.register_buffer("betas", betas)
        self.register_buffer("abar", abar)

    # ------------------------------------------------------------------ training
    def q_sample(self, x0, h0, t, mask):
        """Forward process: z_t = sqrt(abar) z_0 + sqrt(1-abar) eps."""
        ab = self.abar[t].view(-1, 1, 1)
        eps_x = zero_com_noise(x0.shape, mask)
        eps_h = torch.randn_like(h0) * mask.unsqueeze(-1)
        zx = ab.sqrt() * x0 + (1 - ab).sqrt() * eps_x
        zh = ab.sqrt() * h0 + (1 - ab).sqrt() * eps_h
        return zx, zh, eps_x, eps_h

    def loss(self, model, x0, h0, mask):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        h0 = h0 * self.h_scale
        zx, zh, ex, eh = self.q_sample(x0, h0, t, mask)
        px, ph = model(zx, zh, (t.float() + 1) / self.T, mask)
        n = mask.sum(1)                                        # atoms per molecule
        err = ((px - ex) ** 2).sum((1, 2)) + ((ph - eh) ** 2).sum((1, 2))
        return (err / (n * (3 + N_TYPES))).mean()

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, model, n_atoms, steps=None, ddim=False, return_traj=False,
               traj_every=20, generator=None):
        """Generate molecules.

        n_atoms: list of ints (one per molecule)
        steps:   number of denoising steps (None -> all T). Fewer steps = faster.
        ddim:    deterministic DDIM update instead of ancestral DDPM.
        Returns (x, atom_idx, mask) and optionally a trajectory of (x, atom_idx) snapshots.
        """
        device = self.betas.device
        B, N = len(n_atoms), max(n_atoms)
        mask = torch.zeros(B, N, device=device)
        for b, n in enumerate(n_atoms):
            mask[b, :n] = 1
        x = zero_com_noise((B, N, 3), mask, generator)
        h = torch.randn(B, N, N_TYPES, device=device, generator=generator) * mask.unsqueeze(-1)

        if steps is None or steps >= self.T:
            ts = list(range(self.T - 1, -1, -1))
        else:
            ts = sorted({int(round(v)) for v in torch.linspace(0, self.T - 1, steps).tolist()},
                        reverse=True)

        traj = []
        for i, t in enumerate(ts):
            tt = torch.full((B,), t, device=device)
            ex, eh = model(x, h, (tt.float() + 1) / self.T, mask)
            ab_t = self.abar[t]
            ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)

            x0x = (x - (1 - ab_t).sqrt() * ex) / ab_t.sqrt()
            x0h = (h - (1 - ab_t).sqrt() * eh) / ab_t.sqrt()

            if ddim:
                x = ab_prev.sqrt() * x0x + (1 - ab_prev).sqrt() * ex
                h = ab_prev.sqrt() * x0h + (1 - ab_prev).sqrt() * eh
            else:
                alpha_eff = ab_t / ab_prev
                beta_eff = 1 - alpha_eff
                c0 = ab_prev.sqrt() * beta_eff / (1 - ab_t)
                ct = alpha_eff.sqrt() * (1 - ab_prev) / (1 - ab_t)
                var = (1 - ab_prev) / (1 - ab_t) * beta_eff
                x = c0 * x0x + ct * x
                h = c0 * x0h + ct * h
                if i + 1 < len(ts):
                    x = x + var.sqrt() * zero_com_noise(x.shape, mask, generator)
                    h = h + var.sqrt() * torch.randn(h.shape, device=device, generator=generator) \
                        * mask.unsqueeze(-1)
            x = center_masked(x, mask)
            h = h * mask.unsqueeze(-1)

            if return_traj and (i % traj_every == 0 or i + 1 == len(ts)):
                traj.append((x.clone().cpu(), h.argmax(-1).cpu()))

        atom_idx = h.argmax(-1)
        out = (x, atom_idx, mask)
        return (out, traj) if return_traj else out
