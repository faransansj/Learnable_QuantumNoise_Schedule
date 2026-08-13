import math

import pandas as pd
import pytest
import torch

from msquddpm.datasets import clustered_states
from msquddpm.experiment import SCHEDULE_HISTORY_COLUMNS, restore_schedule, train_experiment
from msquddpm.forward_diffusion import ForwardDiffusion, noise_schedule
from msquddpm.schedules import CosineSchedule, LearnableMonotonicSchedule, LinearSchedule
from msquddpm.reverse_model import ReverseMSQuDDPM
from msquddpm.states import purity, validate_density_matrix
from msquddpm.trainer import train_greedy


def _legacy_cosine(steps, offset=0.001, squared=False):
    t = torch.arange(steps + 1, dtype=torch.float64)
    f = torch.cos(((t / steps + offset) / (1 + offset)) * math.pi / 2).square()
    alpha_bar = f / f[0]
    q = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-4, 1)
    return q.square() if squared else q


def test_cosine_regression_exact_legacy_formula():
    for steps in (2, 6, 20):
        assert torch.equal(noise_schedule(steps, "cosine"), _legacy_cosine(steps))
        assert torch.equal(noise_schedule(steps, "sq_cosine"), _legacy_cosine(steps, squared=True))


def test_linear_validity_and_fixed_schedules_have_no_parameters():
    schedule = LinearSchedule(6)
    assert torch.equal(schedule(), torch.linspace(1 / 6, 1, 6, dtype=torch.float64))
    assert list(schedule.parameters()) == []
    assert list(CosineSchedule(6).parameters()) == []


def test_learnable_range_monotonic_terminal_and_cosine_initialization():
    fixed = CosineSchedule(6)()
    learned = LearnableMonotonicSchedule(6)()
    assert torch.allclose(learned, fixed, atol=1e-15, rtol=1e-14)
    assert torch.all((learned > 0) & (learned <= 1))
    assert torch.all(torch.diff(learned) > 0)
    assert learned[-1] == 1
    uniform = LearnableMonotonicSchedule(5, "uniform")()
    assert torch.allclose(uniform, torch.arange(1, 6, dtype=torch.float64) / 5)


def test_learnable_gradient_is_finite_and_optimizer_changes_schedule():
    schedule = LearnableMonotonicSchedule(4)
    before = schedule().detach().clone()
    optimizer = torch.optim.Adam(schedule.parameters(), lr=0.05)
    optimizer.zero_grad()
    loss = (schedule()[1] - 0.7).square()
    loss.backward()
    assert schedule.logits.grad is not None
    assert torch.isfinite(schedule.logits.grad).all()
    optimizer.step()
    assert not torch.equal(schedule().detach(), before)


def test_forward_learnable_physical_and_purity_valid():
    states = clustered_states(8, seed=4)
    trajectory = ForwardDiffusion(4, LearnableMonotonicSchedule(4)).diffuse(states)
    for t in trajectory.steps:
        report = validate_density_matrix(trajectory.get_state(t))
        values = purity(trajectory.get_state(t))
        assert report["valid"]
        assert report["hermitian_error"] <= report["atol"]
        assert report["trace_error"] <= report["atol"]
        assert report["min_eigenvalue"] >= -report["atol"]
        assert torch.all((values >= 0.5 - 1e-7) & (values <= 1 + 1e-7))


def test_joint_greedy_stage_has_finite_gradient_and_updates_schedule():
    states = clustered_states(4, seed=3)
    schedule = LearnableMonotonicSchedule(2)
    forward = ForwardDiffusion(2, schedule).diffuse(states)
    model = ReverseMSQuDDPM(2, n_ancilla=1, depth=1, ancilla="zero", seed=3)
    before = schedule().detach().clone()
    result = train_greedy(
        model, forward, epochs=1, learning_rate=0.01, loss_name="mmd",
        schedule=schedule, schedule_lr=0.02,
    )
    row = result.schedule_history.iloc[0]
    assert row.schedule_grad_norm > 0 and torch.isfinite(torch.tensor(row.schedule_grad_norm))
    assert row.schedule_update_magnitude > 0
    assert not torch.equal(before, schedule().detach())
    assert not any(parameter.requires_grad for parameter in schedule.parameters())


def test_learnable_t_one_rejected():
    with pytest.raises(ValueError, match="T >= 2"):
        LearnableMonotonicSchedule(1)


def _experiment_config(tmp_path, name, schedule):
    return {
        "experiment": name, "paper_scale": False, "dataset": "clustered", "seed": 11,
        "dataset_size": 4, "T": 2, "schedule": schedule, "schedule_offset": 0.001,
        "n_ancilla": 1, "depth": 1, "ancilla": "zero", "loss": "mmd", "epochs": 1,
        "model_lr": 0.01, "schedule_lr": 0.02, "gamma": 1.0, "init": "normal",
        "device": "cpu", "output_root": str(tmp_path),
    }


def test_learnable_experiment_gradient_csv_and_checkpoint_restoration(tmp_path):
    result = train_experiment(_experiment_config(tmp_path, "learned", {"type": "learnable", "init": "cosine"}))
    history = result["schedule_history"]
    assert list(history.columns) == SCHEDULE_HISTORY_COLUMNS
    assert history.schedule_grad_norm.gt(0).all() and history.schedule_update_magnitude.gt(0).all()
    assert torch.isfinite(torch.tensor(history.schedule_grad_norm.to_numpy())).all()
    restored = restore_schedule(result["checkpoint"], "cpu")
    assert torch.equal(restored().detach(), result["schedule"]().detach().cpu())

    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    payload.pop("schedule_state")
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    legacy_schedule = restore_schedule(legacy, "cpu")
    assert torch.allclose(legacy_schedule().detach(), payload["betas"], atol=1e-15, rtol=1e-14)


def test_fixed_schedule_history_csv_has_stable_header(tmp_path):
    train_experiment(_experiment_config(tmp_path, "fixed", "cosine"))
    history = pd.read_csv(tmp_path / "schedules" / "fixed_schedule_history.csv")
    assert list(history.columns) == SCHEDULE_HISTORY_COLUMNS
    assert history.empty
