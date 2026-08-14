from __future__ import annotations

from dataclasses import dataclass, field
import time

import pandas as pd
import torch

from .channels import depolarizing_channel
from .losses import mmd_loss, pairwise_superfidelity, wasserstein_loss
from .precision import precision_for


@dataclass
class TrainResult:
    history: pd.DataFrame
    schedule_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    forward_trajectory: object | None = None


def _forward_target(rho_0: torch.Tensor, q: torch.Tensor, timestep: int) -> torch.Tensor:
    current = rho_0
    for value in q[:timestep]:
        current = depolarizing_channel(current, value.to(device=current.device, dtype=current.real.dtype))
    return current


def train_greedy(
    model,
    forward_trajectory,
    epochs: int,
    learning_rate: float,
    loss_name: str = "wasserstein",
    gamma: float = 1.0,
    log_every: int = 1,
    *,
    schedule=None,
    schedule_lr: float | None = None,
    smoothness_weight: float = 0.0,
    progress_every: int = 0,
) -> TrainResult:
    """Paper T-to-1 greedy training; learnable phi is optimized only at stage T."""
    rows: list[dict] = []
    schedule_rows: list[dict] = []
    batch = len(forward_trajectory.get_state(0))
    rho_0 = forward_trajectory.get_state(0)
    mixed = torch.eye(2, dtype=precision_for(model.theta.device).complex, device=model.theta.device)[None].repeat(batch, 1, 1) / 2
    loss_function = wasserstein_loss if loss_name == "wasserstein" else mmd_loss
    learnable = schedule is not None and schedule.trainable
    if learnable and model.steps < 2:
        raise ValueError("Learnable schedule requires T >= 2")
    started = time.monotonic()
    total_iterations = model.steps * epochs

    for t in range(model.steps, 0, -1):
        optimize_schedule = learnable and t == model.steps
        groups = [{"params": [model.theta], "lr": learning_rate}]
        if optimize_schedule:
            groups.append({"params": list(schedule.parameters()), "lr": schedule_lr or learning_rate})
        optimizer = torch.optim.Adam(groups)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        target = None if optimize_schedule else forward_trajectory.get_state(t - 1).to(model.theta.device)
        ancilla_rng = torch.Generator().manual_seed(model.seed + 10_000 * t)
        fixed_ancillas = {
            stage_t: model._ancillas(batch, model.theta.device, ancilla_rng)
            for stage_t in range(model.steps, t - 1, -1)
        }
        measurement_rng = torch.Generator().manual_seed(model.seed + 20_000 * t)
        for epoch in range(epochs):
            optimizer.zero_grad()
            current = mixed
            with torch.no_grad():
                for fixed_t in range(model.steps, t, -1):
                    current = model.reverse_step(
                        current, fixed_t, validate=False, ancilla_states=fixed_ancillas[fixed_t],
                        measurement_generator=measurement_rng,
                    )
            output = model.reverse_step(
                current.detach(), t, validate=False, ancilla_states=fixed_ancillas[t],
                measurement_generator=measurement_rng,
            )
            if optimize_schedule:
                target = _forward_target(rho_0, schedule(), t - 1).to(model.theta.device)
            loss = loss_function(output, target)
            total_loss = loss + smoothness_weight * schedule.smoothness_loss() if optimize_schedule and smoothness_weight else loss
            q_before = schedule().detach().clone() if optimize_schedule else None
            total_loss.backward()
            if model.theta.grad is not None:
                mask = torch.zeros_like(model.theta.grad)
                mask[t - 1] = 1
                model.theta.grad.mul_(mask)
            grad_norm = 0.0
            if optimize_schedule:
                grads = [p.grad for p in schedule.parameters()]
                if any(g is None for g in grads):
                    raise RuntimeError("Learnable schedule gradient is missing")
                if not all(torch.isfinite(g).all() for g in grads):
                    raise FloatingPointError("Learnable schedule gradient is non-finite")
                grad_norm = float(torch.sqrt(sum(g.square().sum() for g in grads)).detach())
            optimizer.step()
            scheduler.step()
            update = float(torch.linalg.vector_norm(schedule().detach() - q_before)) if optimize_schedule else 0.0
            if optimize_schedule:
                q_now = schedule().detach().cpu()
                for timestep, value in enumerate(q_now, 1):
                    schedule_rows.append({
                        "step": t, "epoch": epoch, "global_iteration": epoch, "timestep": timestep,
                        "q": float(value), "schedule_grad_norm": grad_norm,
                        "schedule_update_magnitude": update, "loss": float(total_loss.detach()),
                    })
            completed = (model.steps - t) * epochs + epoch + 1
            if progress_every and (completed == 1 or completed % progress_every == 0 or completed == total_iterations):
                elapsed = time.monotonic() - started
                eta = elapsed / completed * (total_iterations - completed)
                print(
                    f"progress={100 * completed / total_iterations:6.2f}% "
                    f"iteration={completed}/{total_iterations} step={t} epoch={epoch + 1}/{epochs} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )
            if epoch % log_every == 0 or epoch == epochs - 1:
                rows.append({
                    "step": t, "epoch": epoch, "global_epoch": len(rows),
                    "total_loss": float(total_loss.detach()), "step_loss": float(loss.detach()),
                    "superfidelity": float(pairwise_superfidelity(output.detach(), target.detach()).mean()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "schedule_gradient_norm": grad_norm if optimize_schedule else 0.0,
                    "schedule_update_magnitude": update if optimize_schedule else 0.0,
                })
        if optimize_schedule:
            from .forward_diffusion import ForwardDiffusion
            for parameter in schedule.parameters():
                parameter.requires_grad_(False)
            forward_trajectory = ForwardDiffusion(model.steps, schedule).diffuse(rho_0)

    return TrainResult(pd.DataFrame(rows), pd.DataFrame(schedule_rows), forward_trajectory)
