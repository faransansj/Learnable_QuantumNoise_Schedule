#!/usr/bin/env python3
import argparse
import json

import torch

from msquddpm.precision import precision_for
from msquddpm.utils import get_device


p = argparse.ArgumentParser(description="Check an MSQuDDPM accelerator backend")
p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "xpu", "mps"))
a = p.parse_args()
try:
    device = get_device(a.device)
    precision = precision_for(device)
    matrix = torch.tensor([[1, 1j], [-1j, 1]], dtype=precision.complex, device=device)
    product = matrix @ matrix.mH
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    elif device.type == "xpu":
        name = torch.xpu.get_device_name(device)
    elif device.type == "mps":
        name = "Apple Metal"
    else:
        name = "CPU"
    print(json.dumps({
        "torch": torch.__version__, "requested": a.device, "selected": str(device),
        "available": True, "device_name": name, "real_dtype": str(precision.real),
        "complex_dtype": str(precision.complex), "matmul_finite": bool(torch.isfinite(product).all().cpu()),
    }, indent=2))
except (RuntimeError, AssertionError) as error:
    p.exit(1, f"accelerator check failed: {error}\n")
