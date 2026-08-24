#!/usr/bin/env python3
"""Multi-seed fixed-vs-learnable schedule comparison at reduced scale.

Runs seeds x schedules with identical reduced settings, records quality
metrics plus physical validity of every generated density matrix, and
writes per-run and mean/std summary CSVs.
"""
import argparse

import pandas as pd

from msquddpm.experiment import train_experiment

BASE = {
    "dataset": "clustered",
    "dataset_size": 50,
    "T": 6,
    "schedule_offset": 0.001,
    "n_ancilla": 2,
    "depth": 4,
    "ancilla": "haar",
    "loss": "mmd",
    "epochs": 300,
    "learning_rate": 0.01,
    "gamma": 1.0,
    "init": "normal",
    "device": "cpu",
    "output_root": "outputs",
}

SCHEDULES = {
    "linear": {"schedule": "linear"},
    "cosine": {"schedule": "cosine"},
    "learnable": {"schedule": {"type": "learnable", "init": "cosine", "smoothness_weight": 0.0}},
}


def validity_row(trajectory) -> dict:
    steps = trajectory.validate().values()
    row = {"all_valid": all(step["valid"] for step in steps)}
    for key in ("hermitian_error", "trace_error", "min_eigenvalue"):
        row[f"max_{key}"] = max(abs(step[key]) for step in steps)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for name, override in SCHEDULES.items():
            config = {**BASE, **override, "seed": seed, "device": args.device,
                      "output_root": args.output_root, "experiment": f"multiseed_{name}_s{seed}"}
            result = train_experiment(config)
            rows.append({"schedule": name, "seed": seed, **result["metrics"], **validity_row(result["reverse"])})
            print(f"done: schedule={name} seed={seed} fidelity={result['metrics']['superfidelity']:.4f}")

    runs = pd.DataFrame(rows)
    runs_path = f"{args.output_root}/metrics/multiseed_runs.csv"
    runs.to_csv(runs_path, index=False)

    metric_cols = [c for c in runs.columns if c not in ("schedule", "seed")]
    grouped = runs.groupby("schedule")[metric_cols]
    summary = pd.concat(
        {"mean": grouped.mean(), "std": grouped.std(ddof=1)},
        axis=1,
    ).swaplevel(axis=1).sort_index(axis=1)
    summary_path = f"{args.output_root}/metrics/multiseed_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nruns:    {runs_path}")
    print(f"summary: {summary_path}")
    print("\nmean +/- std across seeds:")
    for schedule, row in summary.iterrows():
        parts = [f"{col}={row[(col, 'mean')]:.4f}±{row[(col, 'std')]:.4f}" for col in ("superfidelity", "trace_distance", "wasserstein")]
        valid = runs[runs.schedule == schedule]["all_valid"].all()
        print(f"  {schedule:10s} {' '.join(parts)} all_valid={valid}")


if __name__ == "__main__":
    main()
