#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from msquddpm.experiment import train_experiment
from msquddpm.utils import load_config


def schedule_fields(config):
    schedule = config["schedule"]
    if isinstance(schedule, dict):
        return {
            "schedule": schedule.get("type", "cosine"),
            "schedule_init": schedule.get("init", ""),
            "smoothness_weight": schedule.get("smoothness_weight", config.get("smoothness_weight", 0.0)),
        }
    return {"schedule": schedule, "schedule_init": "", "smoothness_weight": config.get("smoothness_weight", 0.0)}


def effective_conditions(config):
    schedule = config["schedule"]
    nested_offset = schedule.get("offset") if isinstance(schedule, dict) else None
    conditions = {
        key: config.get(key)
        for key in ("dataset", "seed", "dataset_size", "T", "n_ancilla", "depth", "ancilla", "init", "epochs", "loss", "gamma", "device")
    }
    conditions["model_lr"] = config.get("model_lr", config.get("learning_rate", 0.005))
    conditions["schedule_offset"] = nested_offset if nested_offset is not None else config.get("schedule_offset", 0.001)
    for key in ("evaluation_samples", "evaluation_sample_count", "eval_samples"):
        if any(key in candidate for candidate in configs):
            conditions[key] = config.get(key)
    return conditions


p = argparse.ArgumentParser()
p.add_argument("--configs", nargs="+", required=True)
p.add_argument("--output", default="outputs/metrics/schedule_comparison.csv")
args = p.parse_args()
configs = [load_config(path) for path in args.configs]
reference = effective_conditions(configs[0])
for config in configs[1:]:
    current = effective_conditions(config)
    mismatches = [key for key in reference if current[key] != reference[key]]
    if mismatches:
        raise ValueError(f"Unfair schedule comparison; mismatched effective fields: {mismatches}")

rows = []
model_parameter_count = None
for config in configs:
    result = train_experiment(config)
    current_model_count = sum(parameter.numel() for parameter in result["model"].parameters())
    if model_parameter_count is None:
        model_parameter_count = current_model_count
    elif current_model_count != model_parameter_count:
        raise RuntimeError("Reverse-model parameter counts differ across schedules")
    rows.append({
        "experiment": config["experiment"],
        **schedule_fields(config),
        "schedule_lr": config.get("schedule_lr", ""),
        **reference,
        "model_parameter_count": current_model_count,
        "schedule_trainable_parameter_count": sum(parameter.numel() for parameter in result["schedule"].parameters()),
        **result["metrics"],
    })
out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(out)
print("Schedule CSV inputs:")
for row in rows:
    print(f"outputs/schedules/{row['experiment']}_schedule_final.csv")
