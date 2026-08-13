# Learnable Quantum Noise Schedule

## Research question

Can the depolarizing forward-noise schedule of MSQuDDPM be optimized jointly with the reverse model rather than fixed by a handcrafted cosine-exponent schedule?

Baselines are linear, original cosine-exponent, and global learnable monotonic schedules. The hypothesis is that learned allocation of information loss across the same `T` transitions better matches the data and reverse model. This milestone does not reduce `T`.

## Existing convention

The channel is

`rho_t = (1-q_t) rho_{t-1} + q_t I/d`.

For cosine, `f(t)=cos(((t/T+epsilon)/(1+epsilon))*pi/2)^2`, `alpha_bar_t=f(t)/f(0)`, and `q_t=clamp(1-alpha_bar_t/alpha_bar_(t-1), 1e-4, 1)`. `sq_cosine` squares `q_t` after clipping. Thus `q_t` is the per-transition channel probability, not a newly introduced DDPM alpha/beta convention. Linear remains exactly `linspace(1/T,1,T)`.

## Learnable parameterization and V0 gradient path

With logits `a`, `w=softmax(a)` and `q_t=sum_(i<=t) w_i`; therefore `0<q_1<...<q_T=1`. Cosine initialization uses `a_t=log(q_t-q_(t-1))`, `q_0=0`. Smoothness is optionally `mean((Delta_(t+1)-Delta_t)^2)` and weight zero disables it.

Original greedy training precomputes detached forward targets, so it has no schedule gradient. V0 preserves the greedy algorithm: only at the first reverse stage `T`, it recomputes `rho_(T-1)` from `rho_0` every epoch, jointly updates `theta_T` and schedule `phi` with separate learning rates, then freezes `phi`, rebuilds the forward trajectory, and trains stages `T-1...1` unchanged. Updating a global schedule later would invalidate already frozen later reverse blocks. `T=1` is rejected because its target is `rho_0`, independent of the schedule.

## Commands and artifacts

```bash
python scripts/train.py --config configs/smoke_schedule_cosine.yaml
python scripts/train.py --config configs/smoke_schedule_learnable.yaml
python scripts/schedule_sweep.py --configs configs/schedule_linear.yaml configs/schedule_cosine.yaml configs/schedule_learnable.yaml configs/schedule_learnable_smooth.yaml
```

Artifacts live in `outputs/schedules/`: final and history schedules plus purity trajectory. Fixed schedules write a header-only `*_schedule_history.csv` using the same stable columns as learnable histories. Restore a checkpoint schedule independently with `msquddpm.experiment.restore_schedule(path, device)`; the existing three-value `load_experiment` API remains unchanged, and legacy checkpoints fall back to saved `betas` when available.

After the bounded A–D smoke sweep below, generate the four comparison figures with its concrete outputs:

```bash
python scripts/schedule_sweep.py --configs \
  configs/smoke_schedule_linear.yaml configs/smoke_schedule_cosine.yaml \
  configs/smoke_schedule_learnable.yaml configs/smoke_schedule_learnable_smooth.yaml \
  --output outputs/metrics/smoke_schedule_comparison.csv
python scripts/visualize_schedule.py \
  --experiment smoke_schedule_learnable \
  --linear outputs/schedules/smoke_schedule_linear_schedule_final.csv \
  --cosine outputs/schedules/smoke_schedule_cosine_schedule_final.csv \
  --comparison outputs/metrics/smoke_schedule_comparison.csv
```

The visualization command writes four PNGs under `outputs/figures/smoke_schedule_learnable/`; inputs must already exist from the bounded sweep.

## Non-goals

No CPTP few-step student, teacher-student distillation, per-state adaptive schedule, hardware conditioning, QPU execution, timestep pruning, or automatic important-step selection is implemented.
