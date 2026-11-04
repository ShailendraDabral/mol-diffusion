"""Small shared helpers: masked centering, EMA, device selection."""
import copy

import torch


def center_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtract the (masked) center of mass so every molecule has zero mean.

    x:    (B, N, 3) coordinates (padded atoms may hold anything)
    mask: (B, N)    1 for real atoms, 0 for padding
    """
    m = mask.unsqueeze(-1)
    n = m.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (x * m).sum(dim=1, keepdim=True) / n
    return (x - mean) * m


def zero_com_noise(shape, mask: torch.Tensor, generator=None) -> torch.Tensor:
    """Gaussian noise living in the zero-center-of-mass subspace."""
    eps = torch.randn(shape, device=mask.device, generator=generator)
    return center_masked(eps * mask.unsqueeze(-1), mask)


class EMA:
    """Exponential moving average of model weights (evaluate with this, not the raw model)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
