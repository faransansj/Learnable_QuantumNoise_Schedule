# Accelerator support

| Backend | PyTorch device | Precision | Local validation |
|---|---|---|---|
| CPU | `cpu` | float64 / complex128 | tested |
| NVIDIA | `cuda` | float64 / complex128 | **unverified** — no NVIDIA hardware available; conditional tests and static checks only |
| Intel Arc | `xpu` | float32 / complex64 | Arc B580 tested (smoke + paper-scale, fixed schedules) |
| Apple Silicon | `mps` | float32 / complex64 | tested when available |

`device: auto` selects CUDA, then XPU, then MPS, then CPU. Intel support uses native PyTorch XPU; Intel Extension for PyTorch (IPEX) is neither installed nor required.

## NVIDIA CUDA

Install the PyTorch wheel matching the server driver from the [PyTorch selector](https://pytorch.org/get-started/locally/) through uv:

```bash
uv venv --python 3.11
uv pip install --torch-backend=cu128 -e '.[test]'  # example only; auto also works
uv run --no-sync python scripts/check_accelerator.py --device cuda
uv run --no-sync python scripts/train.py --config configs/smoke_schedule_learnable_cuda.yaml
uv run --no-sync python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke_schedule_learnable_cuda.pt --device cuda
```

`--no-sync` preserves the selected accelerator wheel when running commands.

CUDA retains float64/complex128 research precision. GPU FP64 throughput varies.

**Validation status: unverified.** No NVIDIA device was available during development, so CUDA code paths are covered only by conditional tests (skipped without hardware) and static verification. Real-device behavior and performance have not been measured; paper-scale results on this fork come from CPU or Arc B580 runs.

## Intel Arc XPU

Install a current Intel GPU driver, then the official native XPU wheel through uv:

```bash
uv venv --python 3.11
uv pip install --torch-backend=xpu -e '.[test]'
uv run --no-sync python scripts/check_accelerator.py --device xpu
uv run --no-sync python scripts/train.py --config configs/smoke_schedule_learnable_xpu.yaml
uv run --no-sync python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke_schedule_learnable_xpu.pt --device xpu
```

Official PyTorch XPU scope currently validates Arc A/B client GPUs on Windows 11 and supported Ubuntu releases. See [Getting Started on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) and [Intel GPU prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html). Binary-wheel users need the driver and wheel, not Intel Deep Learning Essentials.

Arc uses float32/complex64 because Arc A-series lacks native FP64. Results are lower-precision diagnostics and are not expected to exactly equal CPU/CUDA. No AMP or GradScaler is used.

## Hybrid execution

The density-matrix circuit, differentiable loss, model parameters, and learnable schedule stay on the selected accelerator. Deterministic dataset/Haar RNG, categorical measurement sampling, POT's detached transport-plan solve, and detached eigendecomposition diagnostics intentionally use CPU. Intel Arc B580 validation used PyTorch `2.13.0+xpu`: accelerator detection and finite complex matmul passed, the full suite reported 21 passed and 2 unavailable-backend skips, the `smoke_schedule_learnable_xpu` train/evaluate flow completed with valid density matrices, and a 6-job paper-scale fixed-schedule run (see `docs/VALIDATION_REPORT.md`, 2026-08-14) completed with checksummed artifacts under `outputs/`. A learnable-schedule paper-scale XPU run remains pending. CUDA stays unverified without NVIDIA hardware.
