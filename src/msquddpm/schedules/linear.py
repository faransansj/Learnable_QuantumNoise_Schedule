from __future__ import annotations

import torch

from .base import NoiseSchedule


class LinearSchedule(NoiseSchedule):
    def __init__(self, steps: int, *, dtype: torch.dtype = torch.float64, device: str | torch.device = "cpu"):
        super().__init__(steps)
        self.register_buffer("values", torch.linspace(1 / steps, 1, steps, dtype=dtype, device=device))

    def forward(self) -> torch.Tensor:
        return self.values
