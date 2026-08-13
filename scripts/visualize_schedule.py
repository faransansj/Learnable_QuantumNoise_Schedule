#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--experiment", required=True)
p.add_argument("--cosine", required=True)
p.add_argument("--linear", required=True)
p.add_argument("--comparison", required=True)
p.add_argument("--output-root", default="outputs")
a = p.parse_args(); root = Path(a.output_root); out = root / "figures" / a.experiment; out.mkdir(parents=True, exist_ok=True)
learned = pd.read_csv(root / "schedules" / f"{a.experiment}_schedule_final.csv")
cosine = pd.read_csv(a.cosine); linear = pd.read_csv(a.linear)
fig, ax = plt.subplots(); ax.plot(linear.timestep, linear.q, label="Linear"); ax.plot(cosine.timestep, cosine.q, label="Original cosine"); ax.plot(learned.timestep, learned.q, label="Learned"); ax.set(xlabel="Timestep", ylabel="Noise strength q_t"); ax.legend(); fig.tight_layout(); fig.savefig(out / "01_noise_schedule.png"); plt.close(fig)
purity = pd.read_csv(root / "schedules" / f"{a.experiment}_purity_trajectory.csv")
for column, name, ylabel in (("average_purity", "02_forward_purity.png", "Average purity"), ("delta_purity", "03_delta_purity.png", "Purity change")):
    fig, ax = plt.subplots(); ax.plot(purity.timestep, purity[column], "o-"); ax.set(xlabel="Timestep", ylabel=ylabel); fig.tight_layout(); fig.savefig(out / name); plt.close(fig)
comparison = pd.read_csv(a.comparison)
available = [c for c in ("superfidelity", "wasserstein", "mmd", "trace_distance") if c in comparison]
if not available:
    raise ValueError("Comparison CSV needs one of: superfidelity, wasserstein, mmd, trace_distance")
metric = available[0]
fig, ax = plt.subplots(); ax.bar(comparison.experiment, comparison[metric]); ax.set(ylabel=metric); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(out / "04_generation_quality.png"); plt.close(fig)
