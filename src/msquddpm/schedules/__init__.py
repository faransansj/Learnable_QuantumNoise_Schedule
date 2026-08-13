from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import NoiseSchedule
from .cosine import CosineSchedule, cosine_values
from .learnable import LearnableMonotonicSchedule
from .linear import LinearSchedule


def make_schedule(
    steps: int,
    config: str | Mapping | NoiseSchedule = "cosine",
    offset: float = 0.001,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> NoiseSchedule:
    if isinstance(config, NoiseSchedule):
        return config
    if isinstance(config, Mapping):
        kind = str(config.get("type", "cosine"))
        offset = float(config.get("offset", offset))
    else:
        kind = str(config)
    if kind == "linear":
        return LinearSchedule(steps, dtype=dtype, device=device)
    if kind in ("cosine", "sq_cosine"):
        return CosineSchedule(steps, offset, squared=kind == "sq_cosine", dtype=dtype, device=device)
    if kind == "learnable":
        init = str(config.get("init", "cosine")) if isinstance(config, Mapping) else "cosine"
        return LearnableMonotonicSchedule(steps, init, offset, dtype=dtype, device=device)
    raise ValueError(f"Unknown schedule: {kind}")


__all__ = [
    "NoiseSchedule", "LinearSchedule", "CosineSchedule", "LearnableMonotonicSchedule",
    "cosine_values", "make_schedule",
]
