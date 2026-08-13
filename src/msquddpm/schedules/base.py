from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class NoiseSchedule(nn.Module, ABC):
    """Transition-noise q_t consumed directly by the depolarizing channel."""

    def __init__(self, steps: int):
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        self.steps = steps

    @abstractmethod
    def forward(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def betas(self) -> torch.Tensor:
        return self()

    @property
    def trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.parameters())

    def smoothness_loss(self) -> torch.Tensor:
        q = self()
        increments = torch.diff(torch.cat((q.new_zeros(1), q)))
        return (increments[1:] - increments[:-1]).square().mean() if self.steps > 1 else q.new_zeros(())
