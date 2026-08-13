from __future__ import annotations

import torch
from torch import nn

from .base import NoiseSchedule
from .cosine import cosine_values


class LearnableMonotonicSchedule(NoiseSchedule):
    """Global monotonic q_t=cumsum(softmax(logits)) with q_T=1."""

    def __init__(
        self,
        steps: int,
        init: str = "cosine",
        offset: float = 0.001,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        if steps < 2:
            raise ValueError("Learnable schedule requires T >= 2: rho_0 target has no schedule gradient")
        super().__init__(steps)
        if init == "cosine":
            q = cosine_values(steps, offset, dtype=dtype, device=device)
            increments = torch.diff(torch.cat((q.new_zeros(1), q)))
            if torch.any(increments <= 0):
                raise ValueError("Cosine initialization must be strictly increasing")
            logits = increments.log()
        elif init == "uniform":
            logits = torch.zeros(steps, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown learnable schedule initialization: {init}")
        self.logits = nn.Parameter(logits)
        self.init = init
        self.offset = offset

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0).cumsum(0)
