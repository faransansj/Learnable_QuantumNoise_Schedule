from __future__ import annotations

import math

import torch

from .base import NoiseSchedule


def cosine_values(
    steps: int,
    offset: float = 0.001,
    squared: bool = False,
    beta_min: float = 1e-4,
    beta_max: float = 1.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    t = torch.arange(steps + 1, dtype=dtype, device=device)
    f = torch.cos(((t / steps + offset) / (1 + offset)) * math.pi / 2).square()
    alpha_bar = f / f[0]
    values = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(beta_min, beta_max)
    return values.square() if squared else values


class CosineSchedule(NoiseSchedule):
    def __init__(
        self,
        steps: int,
        offset: float = 0.001,
        *,
        squared: bool = False,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        super().__init__(steps)
        self.offset = offset
        self.squared = squared
        self.register_buffer("values", cosine_values(steps, offset, squared, dtype=dtype, device=device))

    def forward(self) -> torch.Tensor:
        return self.values
