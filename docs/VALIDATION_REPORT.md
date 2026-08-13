# Validation Report — Steps 2–8

Date: 2026-08-13. Environment: Python 3.11.15, PyTorch 2.13.0, CPU smoke configurations.

## Commands and outcomes

| Command | Exit | Result |
|---|---:|---|
| `uv venv --python /opt/homebrew/bin/python3.11 .venv` | 0 | Environment created |
| `uv pip install --python .venv/bin/python -e '.[test]'` | 0 | Dependencies installed |
| `.venv/bin/pytest -q` | 0 | 8 passed, 1 CUDA-only test skipped |
| `.venv/bin/python scripts/train.py --config configs/smoke_clustered.yaml` | 0 | 2 reverse blocks × 12 epochs |
| `.venv/bin/python scripts/train.py --config configs/smoke_circular.yaml` | 0 | 2 reverse blocks × 12 epochs |
| `.venv/bin/python scripts/quality_sweep.py --config ... --steps 1 2` (both) | 0 | two-point additional smoke sweep CSVs |
| `.venv/bin/python scripts/evaluate.py --checkpoint ...` (both) | 0 | finite metrics, complete valid trajectories |
| `.venv/bin/python scripts/inspect_trajectory.py ...` (four directions) | 0 | forward/reverse step 0–2 inspected for both datasets |
| `.venv/bin/python scripts/visualize.py --experiment ...` (both) | 0 | 12 PNGs each, Figure 10 consumes sweep CSV |

Development/review exposed and fixed detached categorical probability NaNs, differentiation through complex eigendecomposition, RX/RY action order, frozen measurement draws, and incomplete teacher/metric contracts. One corrected training invocation returned shell exit 1 only because `tee` targeted a deleted output directory; the training itself finished. The directory was created and the identical deterministic clustered run was rerun successfully.

## Numerical evidence

### Physical forward process

Clustered mean Bloch radius: `0.996110 → 0.497275 → 0.000000`; mean purity: `0.996121 → 0.623642 → 0.500000`. All matrices passed Hermitian/trace/PSD/purity checks.

Both final forward and reverse trajectories for clustered and circular report `valid=True` at every step. `.pt` versus `.npz` round-trip maximum absolute difference is `0.0`.

### Training

Corrected runs use fixed per-stage Haar inputs and advancing projective-measurement RNGs. Losses are therefore stochastic and need not decrease monotonically.

- Cluster step 2: `0.119073 → 0.120959`, minimum `0.100941`; step 1: `0.549477 → 0.645476`, minimum `0.497061`.
- Circle step 2: `0.047184 → 0.029099`; step 1: `0.243997 → 0.153914`, minimum `0.152253`.

This validates finite optimization and parameter updates, not convergence quality or Table-I reproduction.

### Final smoke metrics

| Experiment | Nearest superfidelity | MMD | Wasserstein | Generated radius |
|---|---:|---:|---:|---:|
| clustered | 0.5920 | 0.7041 | 0.4303 | 0.5634 |
| circular | 0.8692 | 0.1456 | 0.3146 | 0.7333 |

Cluster paper metric: `F_data_0=0.97698`, `F_gen_0=0.48024`. Circular diagnostic overlaps are `0.52008` and `0.45303` respectively.

Smoke quality is intentionally bounded by `T=2`, depth 2, 10 states, 12 epochs.

## Trajectory evidence

Each teacher file contains:

```text
forward_sample_id, reverse_sample_id, sample_id, paired=false, steps,
rho_0, rho_1, rho_2,
reverse_rho_0, reverse_rho_1, reverse_rho_2
```

Forward and reverse rows are independent ensemble paths; `sample_id` is a legacy positional index and does not assert pairing. All teacher matrices passed validation: worst Hermiticity residual `2.02e-16`, trace residual `2.23e-16`, minimum eigenvalue `3.06e-4`. Production paper configs yield the same schema through `rho_6`.

## Figure evidence

Both smoke experiments contain exactly 12 nonempty PNGs, dimensions `960×640` through `2880×960`. Figure 10 uses actual additional-smoke sweep points: clustered `(T=1, 0.6160)`, `(T=2, 0.5585)`; circular `(T=1, 0.8266)`, `(T=2, 0.8147)`. These two-point stochastic smoke trends are diagnostics, not evidence that quality must increase with T.

## Apple Silicon MPS validation

PyTorch 2.13.0 reported MPS built/available. Core tensors were observed on `mps:0`: model parameters `float32`, datasets and forward/reverse states `complex64`; the conditional MPS test performs a real optimizer update and confirms an MPS-resident gradient. Full suite: `9 passed, 1 CUDA-only skipped`.

Actual smoke runs completed without `PYTORCH_ENABLE_MPS_FALLBACK`:

| Experiment | Train wall time | Superfidelity | MMD | Wasserstein | Device/dtypes |
|---|---:|---:|---:|---:|---|
| clustered MPS | 5.91 s | 0.5173 | 0.7852 | 0.4856 | `mps`, `float32/complex64` |
| circular MPS | 4.29 s | 0.8316 | 0.1275 | 0.2443 | `mps`, `float32/complex64` |

Both evaluation, forward/reverse inspection, two-point quality sweeps, and all 12 figures completed. Worst physical residuals across saved MPS trajectories: Hermiticity `1.08e-7`, trace `1.92e-7`, minimum eigenvalue `3.46e-4`; all pass the documented low-precision `2e-5` tolerance. CPU metrics above are comparison diagnostics only because stochastic measurement paths and numeric precision differ.

MPS does not support `float64/complex128` or complex `eigh/eigvalsh`. The implementation therefore selects `float32/complex64` centrally for MPS. Circuit evolution, differentiable costs, and parameter gradients remain on MPS. Reproducible categorical measurement draws use CPU probabilities/outcome indices, and Wasserstein uses POT on CPU to solve a detached transport plan; the chosen cost is then weighted and differentiated on MPS. Detached validation, state-fidelity, trace-distance, and eigenvalue-plot inputs also use CPU `complex128`. Thus the validated path is intentionally hybrid rather than fully GPU-native.

## Bounded resource exception

Paper configs require six greedy blocks, 100/200 samples, deeper circuits, Wasserstein OT each epoch, and 2001 assumed epochs per block. They were not executed as part of CPU smoke validation, so this report does not claim reproduction of Table-I values. The configs are preserved for an explicitly budgeted paper-scale run.

## Learnable Schedule Validation — 2026-08-13

This section records the separate bounded CPU validation for the global learnable schedule; the historical reproduction evidence above is unchanged. Before the accepted review fixes, the full suite reported `16 passed, 1 skipped`. After checkpoint/history/fairness integration tests were added, `PYTHONPATH=src .../.venv/bin/python -m pytest -q` reported `18 passed, 1 skipped` (CUDA unavailable).

Commands completed with exit 0:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --config configs/smoke_schedule_learnable.yaml
PYTHONPATH=src .venv/bin/python scripts/schedule_sweep.py --configs configs/smoke_schedule_linear.yaml configs/smoke_schedule_cosine.yaml configs/smoke_schedule_learnable.yaml configs/smoke_schedule_learnable_smooth.yaml --output outputs/metrics/smoke_schedule_comparison.csv
PYTHONPATH=src .venv/bin/python scripts/visualize_schedule.py --experiment smoke_schedule_learnable --linear outputs/schedules/smoke_schedule_linear_schedule_final.csv --cosine outputs/schedules/smoke_schedule_cosine_schedule_final.csv --comparison outputs/metrics/smoke_schedule_comparison.csv
```

The exact cosine-init error before optimization was `0.0`. The learned V0 schedule was `q=[0.5207633552821634, 1.0]`; per-epoch schedule gradient norms were `[0.1950603299022585, 0.1897876150017552]` and update magnitudes were `[0.0099983283515043, 0.0099816430096003]`. Average forward purity was `[0.9964507511488842, 0.6140187327810643, 0.5]`, with finite changes `[0.3824320183678198, 0.1140187327810643]` after the initial state.

Every learned forward state passed Hermiticity, trace-one, PSD, and purity validation. Worst residuals were Hermiticity `8.98e-19`, trace `4.44e-16`, and minimum eigenvalue `6.60e-05`. The A–D aggregate is `outputs/metrics/smoke_schedule_comparison.csv`; reverse-model parameter count was 8 for every schedule, while trainable schedule counts were 0/0/2/2. Four nonempty figures were generated under `outputs/figures/smoke_schedule_learnable/`.

These `T=2`, six-sample, two-epoch runs validate execution, gradient flow, persistence, fair wiring, physicality, and plotting only. They do not establish generation-quality improvement. The paper-scale `T=6`, 2001-epoch configs were not run.
