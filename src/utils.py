import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)

    # Reproducible hơn, có thể hơi chậm hơn.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(device_name: str = "cuda") -> torch.device:
    if device_name == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_name == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)