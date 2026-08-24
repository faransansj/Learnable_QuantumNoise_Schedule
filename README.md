# Original MSQuDDPM Reproduction

PyTorch density-matrix reproduction of *Mixed-State Quantum Denoising Diffusion Probabilistic Model* ([arXiv:2411.17608v2](https://arxiv.org/abs/2411.17608v2)), based primarily on the authors' [`gkwun/msquddpm`](https://github.com/gkwun/msquddpm) implementation at commit `158df6e9474aca6a9ab00d01b60fe4d65cc093ba`. The same RX/RY/CZ density-matrix circuit is implemented directly in PyTorch rather than through TensorCircuit, avoiding an extra runtime layer while retaining autograd and the official framework's PyTorch backend semantics.

The original MSQuDDPM teacher behavior remains the default. This fork additionally supports fixed linear and global learnable monotonic depolarizing schedules; see [`docs/LEARNABLE_SCHEDULE.md`](docs/LEARNABLE_SCHEDULE.md). It does **not** implement Few-Step students or distillation.

## Installation

```bash
uv venv --python 3.11
uv pip install -e '.[test]'
```

`device: auto` selects CUDA, then Intel XPU, Apple MPS, and CPU. CPU/CUDA use `float64/complex128`; XPU/MPS use `float32/complex64` with `atol=2e-5`. Circuit evolution and differentiable losses run on the selected accelerator. Reproducible RNG/sampling, POT's detached transport-plan solve, and detached eigendecomposition diagnostics are CPU control operations. See [`docs/ACCELERATORS.md`](docs/ACCELERATORS.md) for official CUDA/XPU installation, support scope, and exact smoke commands.

## CUDA server quick start

The project does not install a CUDA toolkit itself. Use a server/driver-supported PyTorch CUDA wheel, then install this repository without replacing that wheel.

```bash
git clone https://github.com/faransansj/Learnable_QuantumNoise_Schedule.git
cd Learnable_QuantumNoise_Schedule

# Python 3.11 environment with the CUDA wheel matching the server driver.
# Example for CUDA 12.8; use --torch-backend=auto to detect it instead.
uv venv --python 3.11
uv pip install --torch-backend=cu128 -e '.[test]'
```

The accelerator commands use `uv run --no-sync` to preserve the selected PyTorch backend.

Confirm that the GPU is really visible:

```bash
nvidia-smi
uv run --no-sync python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available()
PY
```

Run tests and a CUDA smoke training before paper-scale jobs:

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/train.py --config configs/smoke_clustered_cuda.yaml
uv run --no-sync python scripts/train.py --config configs/smoke_circular_cuda.yaml
uv run --no-sync python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/smoke_clustered_cuda.pt --device cuda
```

Paper-scale configs use `device: auto`, which selects CUDA on a CUDA server:

```bash
uv run --no-sync python scripts/train.py --config configs/1q_clustered.yaml
uv run --no-sync python scripts/train.py --config configs/1q_circular.yaml
```

For an unattended server job, keep stdout and timing information:

```bash
mkdir -p outputs/logs
nohup /usr/bin/time -v uv run --no-sync python scripts/train.py --config configs/1q_clustered.yaml \
  > outputs/logs/1q_clustered_cuda.log 2>&1 &
echo $! > outputs/logs/1q_clustered_cuda.pid
```

Paper configs print periodic `progress`, elapsed time, and ETA lines; redirect stdout to a log and follow it with `tail -f`. Generated checkpoints, trajectories, figures, metrics, histories, and logs live under `outputs/` and are intentionally git-ignored. Copy them separately from the server. CUDA currently uses `float64/complex128`; verify that the selected GPU supports efficient FP64 if runtime matters. The POT transport-plan solve and measurement sampling remain CPU-assisted, so additional CPU cores and fast host-device transfers still help.

## Intel Arc XPU

Native PyTorch XPU is supported without IPEX. Install a current Intel driver and wheel, then verify and run the schedule smoke:

```bash
uv venv --python 3.11
uv pip install --torch-backend=xpu -e '.[test]'
uv run --no-sync python scripts/check_accelerator.py --device xpu
uv run --no-sync python scripts/train.py --config configs/smoke_schedule_learnable_xpu.yaml
uv run --no-sync python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke_schedule_learnable_xpu.pt --device xpu
```

Arc runs float32/complex64 without AMP/GradScaler. Full dependency commands, official Arc A/B Linux/Windows scope, and honest validation status are in [`docs/ACCELERATORS.md`](docs/ACCELERATORS.md).

## Apple Silicon MPS

Apple Silicon smoke run:

```bash
.venv/bin/python scripts/train.py --config configs/smoke_clustered_mps.yaml
.venv/bin/python scripts/train.py --config configs/smoke_circular_mps.yaml
.venv/bin/python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke_clustered_mps.pt --device mps
.venv/bin/python scripts/quality_sweep.py --config configs/smoke_clustered_mps.yaml --steps 1 2
.venv/bin/python scripts/visualize.py --experiment smoke_clustered_mps
```

Known limit: this is a hybrid MPS execution path, not a claim that every auxiliary operation is GPU-native. MPS results are lower-precision stochastic diagnostics and are not expected to equal CPU metrics exactly. Checkpoint loading stages through CPU to avoid materializing float64 schedule metadata on MPS.

## Dataset

- Clustered: paper-v2 `|ψ> ∝ |0> + 0.08 c|1>`, complex-normal `c`, `q ~ U[0,0.01)`.
- Circular: literal paper `RY(θ)|0>`, `θ ~ U[0,2π)`, `q ~ U[0,0.04)`.
- Every generated state is checked for Hermiticity, trace one, PSD, and valid purity.

## Training

Paper-scale configuration (expensive; Table-I result is not claimed until run):

```bash
.venv/bin/python scripts/train.py --config configs/1q_clustered.yaml
.venv/bin/python scripts/train.py --config configs/1q_circular.yaml
```

CPU smoke reproduction:

```bash
.venv/bin/python scripts/train.py --config configs/smoke_clustered.yaml
.venv/bin/python scripts/train.py --config configs/smoke_circular.yaml
```

Training follows the paper's greedy `T → 1` process. Each RX/RY/CZ block is separately addressable with `model.reverse_step(rho, t)`. Ancilla Z measurements sample conditional post-measurement states; outcomes are not postselected or retained as labels.

## Evaluation

```bash
.venv/bin/python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke_clustered.pt
```

CSV output contains nearest-state fidelity/superfidelity, trace distance, MMD, Wasserstein, purity error, Bloch radii, and the clustered paper metric `F_data_0`/`F_gen_0 = mean(<0|rho|0>)`. Distributional MMD/Wasserstein are primary; nearest-state metrics are diagnostics.

## Visualization

```bash
.venv/bin/python scripts/visualize.py --experiment smoke_clustered
```

First run the bounded **additional smoke experiment** (not a paper/Table-I sweep), then render figures:

```bash
.venv/bin/python scripts/quality_sweep.py --config configs/smoke_clustered.yaml --steps 1 2
.venv/bin/python scripts/visualize.py --experiment smoke_clustered
```

This creates all required files `01_dataset_bloch.png` through `12_eigenvalue_evolution.png` under `outputs/figures/<experiment>/`. Figure 10 consumes `outputs/metrics/<experiment>_quality_vs_steps.csv` and requires at least two actual trained T values.

## Trajectory inspection

```bash
.venv/bin/python scripts/inspect_trajectory.py --experiment smoke_clustered --direction forward
.venv/bin/python scripts/inspect_trajectory.py --experiment smoke_clustered --direction reverse
```

API:

```python
trajectory = model.generate(rho_T, return_trajectory=True)
rho_4 = trajectory.get_state(4)
rho_4_again = model.get_state(4)
rho_next = model.reverse_step(rho, t=8)
save_trajectory(trajectory, "trajectory.pt")  # also .npz
trajectory = load_trajectory("trajectory.pt")
```

Teacher `.npz` files expose `forward_sample_id`, `reverse_sample_id`, legacy positional `sample_id`, `paired=false`, `rho_0...rho_T`, and `reverse_rho_0...reverse_rho_T`. Forward and reverse rows are independent ensemble paths; equal row indices do not imply coupled samples.

## Reproduction procedure

1. Read [`docs/STEP1_PAPER_AND_OFFICIAL_CODE_ANALYSIS.md`](docs/STEP1_PAPER_AND_OFFICIAL_CODE_ANALYSIS.md).
2. Run tests.
3. Run both smoke configs and inspect numerical/figure outputs.
4. Run paper configs with recorded hardware/runtime and multiple seeds.
5. Compare with Table I using [`docs/REPRODUCTION_REPORT.md`](docs/REPRODUCTION_REPORT.md); never label smoke output paper reproduction.

## Documented assumptions

- `epsilon=0.001` and beta clipping follow official code because the paper omits the offset value.
- Paper-scale LR/epochs use official CLI defaults but are not paper-attested per-task settings. `gamma=1` means Adam with **no effective learning-rate decay**; the paper's decay factor/cadence are unknown.
- Dataset interval parameters are sampled uniformly, following official code.
- The implementation fixes the official multi-qubit repeated-global-channel defect, default training crash, unsupported README option, and unusable generation parameter path.

## Learnable schedule smoke run

```bash
.venv/bin/python scripts/train.py --config configs/smoke_schedule_cosine.yaml
.venv/bin/python scripts/train.py --config configs/smoke_schedule_learnable.yaml
```

Schedule values, gradients/updates, and forward purity diagnostics are written under `outputs/schedules/`. Fixed schedules have a header-only history CSV with the same schema. `msquddpm.experiment.restore_schedule(checkpoint, device)` restores current learnable schedule state while `load_experiment` retains its existing return signature and legacy-checkpoint compatibility.

Bounded A–D smoke comparison and its four figures:

```bash
PYTHONPATH=src .venv/bin/python scripts/schedule_sweep.py --configs \
  configs/smoke_schedule_linear.yaml configs/smoke_schedule_cosine.yaml \
  configs/smoke_schedule_learnable.yaml configs/smoke_schedule_learnable_smooth.yaml \
  --output outputs/metrics/smoke_schedule_comparison.csv
PYTHONPATH=src .venv/bin/python scripts/visualize_schedule.py \
  --experiment smoke_schedule_learnable \
  --linear outputs/schedules/smoke_schedule_linear_schedule_final.csv \
  --cosine outputs/schedules/smoke_schedule_cosine_schedule_final.csv \
  --comparison outputs/metrics/smoke_schedule_comparison.csv
```

The second command requires the first command's CSV outputs and writes four PNGs under `outputs/figures/smoke_schedule_learnable/`. Research configs keep the original clustered `T=6`; smoke configs are explicitly not paper-scale.

## Progress

- [x] Step 1: paper/official-code analysis
- [x] Steps 2–8: modular implementation and CPU smoke validation
- [ ] Independent paper-scale multi-seed Table-I reproduction (resource-intensive)
