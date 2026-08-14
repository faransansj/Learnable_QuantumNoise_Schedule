# Accelerator support

| Backend | PyTorch device | Precision | Local validation |
|---|---|---|---|
| CPU | `cpu` | float64 / complex128 | tested |
| NVIDIA | `cuda` | float64 / complex128 | conditional test; unavailable on this machine |
| Intel Arc | `xpu` | float32 / complex64 | conditional test; unavailable on this machine |
| Apple Silicon | `mps` | float32 / complex64 | tested when available |

`device: auto` selects CUDA, then XPU, then MPS, then CPU. Intel support uses native PyTorch XPU; Intel Extension for PyTorch (IPEX) is neither installed nor required.

## NVIDIA CUDA

Install the PyTorch wheel matching the server driver from the [PyTorch selector](https://pytorch.org/get-started/locally/), then install this project without replacing Torch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128  # example only
pip install -e '.[test]' --no-deps
pip install 'numpy>=1.26,<2' 'scipy>=1.11,<2' 'matplotlib>=3.8,<4' 'pandas>=2.1,<3' 'pyyaml>=6,<7' 'POT>=0.9,<1' 'pytest>=8,<9'
python scripts/check_accelerator.py --device cuda
python scripts/train.py --config configs/smoke_schedule_learnable_cuda.yaml
python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke_schedule_learnable_cuda.pt --device cuda
```

CUDA retains float64/complex128 research precision. GPU FP64 throughput varies.

## Intel Arc XPU

Install a current Intel GPU driver, then the official native XPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install -e '.[test]' --no-deps
pip install 'numpy>=1.26,<2' 'scipy>=1.11,<2' 'matplotlib>=3.8,<4' 'pandas>=2.1,<3' 'pyyaml>=6,<7' 'POT>=0.9,<1' 'pytest>=8,<9'
python scripts/check_accelerator.py --device xpu
python scripts/train.py --config configs/smoke_schedule_learnable_xpu.yaml
python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke_schedule_learnable_xpu.pt --device xpu
```

Official PyTorch XPU scope currently validates Arc A/B client GPUs on Windows 11 and supported Ubuntu releases. See [Getting Started on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) and [Intel GPU prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html). Binary-wheel users need the driver and wheel, not Intel Deep Learning Essentials.

Arc uses float32/complex64 because Arc A-series lacks native FP64. Results are lower-precision diagnostics and are not expected to exactly equal CPU/CUDA. No AMP or GradScaler is used.

## Hybrid execution

The density-matrix circuit, differentiable loss, model parameters, and learnable schedule stay on the selected accelerator. Deterministic dataset/Haar RNG, categorical measurement sampling, POT's detached transport-plan solve, and detached eigendecomposition diagnostics intentionally use CPU. This machine has no CUDA/XPU device, so real hardware tests remain conditional and no Arc performance or correctness claim is made beyond code-path tests.
