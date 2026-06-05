from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.models.baseline import BaselineMobileNetV3
from src.models.aqb_fas import AQBFAS
from src.losses import compute_loss


def check_baseline():
    print("=" * 80)
    print("[CHECK] BaselineMobileNetV3")

    model = BaselineMobileNetV3(pretrained=False)
    model.train()

    x = torch.randn(4, 3, 224, 224)

    targets = {
        "label": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "spoof_type": torch.tensor([0, 1, 0, 8], dtype=torch.long),
        "illumination": torch.tensor([0, 1, 0, 2], dtype=torch.long),
        "environment": torch.tensor([0, 1, 0, 2], dtype=torch.long),
    }

    outputs = model(x)

    print("pad_logit:", outputs["pad_logit"].shape)

    assert outputs["pad_logit"].shape == (4,)

    loss, logs = compute_loss(
        outputs,
        targets,
        {
            "lambda_spoof": 0.0,
            "lambda_illum": 0.0,
            "lambda_env": 0.0,
            "gamma_proto": 0.0,
        },
    )

    print("loss:", float(loss.detach()))
    print("logs:", logs)
    print("[OK] Baseline forward + loss OK")


def check_aqb():
    print("=" * 80)
    print("[CHECK] AQBFAS")

    model = AQBFAS(dz=64, bits=8, pretrained=False)
    model.train()

    x = torch.randn(4, 3, 224, 224)

    targets = {
        "label": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "spoof_type": torch.tensor([0, 1, 0, 8], dtype=torch.long),
        "illumination": torch.tensor([0, 1, 0, 2], dtype=torch.long),
        "environment": torch.tensor([0, 1, 0, 2], dtype=torch.long),
    }

    outputs = model(x)

    expected_shapes = {
        "pad_logit": (4,),
        "spoof_logits": (4, 11),
        "illum_logits": (4, 5),
        "env_logits": (4, 3),
        "proto_logits": (4, 2),
        "z": (4, 64),
        "zq": (4, 64),
    }

    for k, shape in expected_shapes.items():
        print(k, outputs[k].shape)
        assert tuple(outputs[k].shape) == shape

    assert torch.isfinite(outputs["z"]).all()
    assert torch.isfinite(outputs["zq"]).all()

    loss, logs = compute_loss(
        outputs,
        targets,
        {
            "lambda_spoof": 0.3,
            "lambda_illum": 0.1,
            "lambda_env": 0.1,
            "gamma_proto": 0.05,
        },
    )

    print("latent_bits:", model.latent_bits())
    print("loss:", float(loss.detach()))
    print("logs:", logs)
    print("[OK] AQB-FAS forward + loss OK")


if __name__ == "__main__":
    check_baseline()
    check_aqb()
    print("=" * 80)
    print("[DONE] All model checks passed.")