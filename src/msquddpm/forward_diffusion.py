from __future__ import annotations

from collections.abc import Mapping

import torch

from .channels import depolarizing_channel
from .schedules import NoiseSchedule, make_schedule
from .states import validate_density_matrix
from .trajectory import Trajectory


def noise_schedule(
    steps: int,
    kind: str = "cosine",
    offset: float = 0.001,
    beta_min: float = 1e-4,
    beta_max: float = 1.0,
) -> torch.Tensor:
    """Compatibility API preserving the original numerical convention."""
    if beta_min != 1e-4 or beta_max != 1.0:
        from .schedules.cosine import cosine_values
        if kind == "linear":
            return torch.linspace(1 / steps, 1, steps, dtype=torch.float64)
        if kind not in ("cosine", "sq_cosine"):
            raise ValueError(f"Unknown schedule: {kind}")
        return cosine_values(steps, offset, kind == "sq_cosine", beta_min, beta_max)
    return make_schedule(steps, kind, offset)()


class ForwardDiffusion:
    def __init__(self, steps: int, schedule: str | Mapping | NoiseSchedule = "cosine", offset: float = 0.001):
        self.steps = steps
        self.schedule = make_schedule(steps, schedule, offset)
        self.schedule_name = (
            str(schedule.get("type", "cosine")) if isinstance(schedule, Mapping)
            else schedule.__class__.__name__ if isinstance(schedule, NoiseSchedule)
            else str(schedule)
        )
        self.offset = offset

    @property
    def betas(self) -> torch.Tensor:
        return self.schedule()

    def diffuse(self, rho_0: torch.Tensor, validate: bool = True) -> Trajectory:
        states = {0: rho_0.clone()}
        current = rho_0
        for t, beta in enumerate(self.betas, start=1):
            current = depolarizing_channel(current, beta.to(device=current.device, dtype=current.real.dtype))
            if validate:
                validate_density_matrix(current)
            states[t] = current
        return Trajectory(states, "forward", {"schedule": self.schedule_name, "betas": self.betas.detach().cpu().tolist()})
