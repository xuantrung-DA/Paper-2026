from typing import Dict, Any

from src.models.baseline import BaselineMobileNetV3
from src.models.aqb_fas import AQBFAS


def build_model(cfg: Dict[str, Any]):
    model_cfg = cfg["model"]
    name = model_cfg["name"]

    if name == "baseline_mbv3":
        return BaselineMobileNetV3(
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )

    if name == "aqb_fas":
        return AQBFAS(
            dz=int(model_cfg.get("dz", 64)),
            bits=int(model_cfg.get("bits", 8)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )

    raise ValueError(f"Unknown model name: {name}")