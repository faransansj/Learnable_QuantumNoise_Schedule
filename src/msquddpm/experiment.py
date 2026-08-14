from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .datasets import make_dataset
from .forward_diffusion import ForwardDiffusion
from .metrics import nearest_metrics
from .precision import precision_for
from .reverse_model import ReverseMSQuDDPM
from .schedules import LearnableMonotonicSchedule, make_schedule
from .states import purity
from .trainer import train_greedy
from .trajectory import save_teacher_trajectory, save_trajectory
from .utils import ensure_output_dirs, get_device, json_dump, set_seed


def _schedule_config(config: dict):
    value = config["schedule"]
    if isinstance(value, dict):
        return {**value, "offset": value.get("offset", config.get("schedule_offset", 0.001))}
    return value


SCHEDULE_HISTORY_COLUMNS = [
    "step", "epoch", "global_iteration", "timestep", "q",
    "schedule_grad_norm", "schedule_update_magnitude", "loss",
]


def _save_schedule_outputs(name, output, diffusion, result, forward):
    q = diffusion.betas.detach().cpu()
    pd.DataFrame({"timestep": range(1, len(q) + 1), "q": q.numpy()}).to_csv(
        output["schedules"] / f"{name}_schedule_final.csv", index=False
    )
    result.schedule_history.reindex(columns=SCHEDULE_HISTORY_COLUMNS).to_csv(
        output["schedules"] / f"{name}_schedule_history.csv", index=False
    )
    means = [float(purity(forward.get_state(t)).mean().detach().cpu()) for t in forward.steps]
    deltas = [float("nan")] + [means[t - 1] - means[t] for t in range(1, len(means))]
    pd.DataFrame({"timestep": forward.steps, "average_purity": means, "delta_purity": deltas}).to_csv(
        output["schedules"] / f"{name}_purity_trajectory.csv", index=False
    )


def train_experiment(config: dict) -> dict:
    seed = int(config["seed"]); set_seed(seed)
    device = get_device(config.get("device", "auto")); output = ensure_output_dirs(config.get("output_root", "outputs"))
    name = config["experiment"]
    dataset = make_dataset(config["dataset"], int(config["dataset_size"]), seed, device)
    schedule = make_schedule(
        int(config["T"]), _schedule_config(config), float(config.get("schedule_offset", 0.001)),
        dtype=precision_for(device).real, device=device,
    )
    diffusion = ForwardDiffusion(int(config["T"]), schedule)
    forward = diffusion.diffuse(dataset)
    model = ReverseMSQuDDPM(
        steps=int(config["T"]), n_ancilla=int(config["n_ancilla"]), depth=int(config["depth"]),
        ancilla=config["ancilla"], seed=seed, init=config.get("init", "normal"), device=device
    )
    model_lr = float(config.get("model_lr", config.get("learning_rate", 0.005)))
    result = train_greedy(
        model, forward, int(config["epochs"]), model_lr, config["loss"], float(config.get("gamma", 1.0)),
        schedule=schedule, schedule_lr=float(config.get("schedule_lr", model_lr)),
        smoothness_weight=float(config.get("smoothness_weight", config.get("schedule", {}).get("smoothness_weight", 0.0)) if isinstance(config.get("schedule"), dict) else config.get("smoothness_weight", 0.0)),
        progress_every=int(config.get("progress_every", 0)),
    )
    if result.forward_trajectory is not None:
        forward = result.forward_trajectory
        diffusion = ForwardDiffusion(int(config["T"]), schedule)
    mixed = torch.eye(2,dtype=precision_for(device).complex,device=device)[None].repeat(len(dataset),1,1)/2
    reverse = model.generate(mixed, return_trajectory=True)
    checkpoint = output["checkpoints"] / f"{name}.pt"
    torch.save({
        "model_state": model.state_dict(), "config": config, "betas": diffusion.betas.detach().cpu(),
        "schedule_state": schedule.state_dict(), "schedule_type": _schedule_config(config),
        "dataset": dataset.detach().cpu(),
    }, checkpoint)
    history_path=output["histories"]/f"{name}.csv"; result.history.to_csv(history_path,index=False)
    forward_pt=output["trajectories"]/f"{name}_forward.pt"; reverse_pt=output["trajectories"]/f"{name}_reverse.pt"
    save_trajectory(forward,forward_pt); save_trajectory(reverse,reverse_pt)
    save_trajectory(forward,forward_pt.with_suffix('.npz')); save_trajectory(reverse,reverse_pt.with_suffix('.npz'))
    teacher=output["trajectories"]/f"{name}_teacher.npz"; save_teacher_trajectory(forward,reverse,teacher)
    metrics=nearest_metrics(reverse.get_state(0).detach(),dataset)
    pd.DataFrame([{"experiment":name,"T":config["T"],**metrics}]).to_csv(output["metrics"]/f"{name}.csv",index=False)
    _save_schedule_outputs(name, output, diffusion, result, forward)
    json_dump({"device":str(device),"real_dtype":str(model.theta.dtype),"complex_dtype":str(dataset.dtype),"checkpoint":str(checkpoint),"metrics":metrics},output["metrics"]/f"{name}_summary.json")
    return {"dataset":dataset,"forward":forward,"reverse":reverse,"model":model,"schedule":schedule,"diffusion":diffusion,"history":result.history,"schedule_history":result.schedule_history,"metrics":metrics,"checkpoint":checkpoint,"outputs":output}


def restore_schedule(checkpoint: str | Path, device: str = "auto"):
    """Restore current schedule state, with legacy checkpoint fallback to saved betas."""
    target = get_device(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    schedule = make_schedule(
        int(config["T"]), _schedule_config(config), float(config.get("schedule_offset", 0.001)),
        dtype=precision_for(target).real, device=target,
    )
    if "schedule_state" in payload:
        schedule.load_state_dict(payload["schedule_state"])
    elif isinstance(schedule, LearnableMonotonicSchedule) and "betas" in payload:
        q = payload["betas"].to(device=target, dtype=precision_for(target).real)
        increments = torch.diff(torch.cat((q.new_zeros(1), q)))
        if torch.any(increments <= 0):
            raise ValueError("Legacy learnable checkpoint betas must be strictly increasing")
        with torch.no_grad():
            schedule.logits.copy_(increments.log())
    return schedule


def load_experiment(checkpoint: str | Path, device: str = "auto") -> tuple[ReverseMSQuDDPM,dict,torch.Tensor]:
    target=get_device(device); payload=torch.load(checkpoint,map_location="cpu",weights_only=False); config=payload["config"]
    model=ReverseMSQuDDPM(int(config["T"]),n_ancilla=int(config["n_ancilla"]),depth=int(config["depth"]),ancilla=config["ancilla"],seed=int(config["seed"]),init=config.get("init","normal"),device=target)
    model.load_state_dict(payload["model_state"])
    return model,config,payload["dataset"].to(device=target,dtype=precision_for(target).complex)
