from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not callable(getattr(xpu, "is_available", None)):
        return False
    try:
        return xpu.is_available()
    except (AssertionError, RuntimeError):
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _xpu_available() and callable(getattr(torch.xpu, "manual_seed_all", None)):
        torch.xpu.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _xpu_available():
            return torch.device("xpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; install a driver-compatible PyTorch CUDA wheel")
    if device.type == "xpu" and not _xpu_available():
        raise RuntimeError(
            "Intel XPU was requested but is unavailable; install a current Intel GPU driver "
            "and the PyTorch XPU wheel from https://download.pytorch.org/whl/xpu"
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available; use device: cpu")
    return device


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["config_path"] = str(path)
    return config


def ensure_output_dirs(root: str | Path = "outputs") -> dict[str, Path]:
    root = Path(root)
    paths = {name: root / name for name in ("checkpoints", "trajectories", "metrics", "figures", "histories", "schedules")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def json_dump(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
