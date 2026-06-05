from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Cross entropy có hỗ trợ ignore_index=-1.

    Cần cái này để sau này test CASIA:
      spoof_type = -1
      illumination = -1
      environment = -1
    thì semantic losses không bị crash.
    """
    mask = target != ignore_index

    if mask.sum() == 0:
        return logits.sum() * 0.0

    return F.cross_entropy(logits[mask], target[mask])


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    loss_cfg: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    outputs: output từ model
    targets:
      label: 0 live, 1 spoof
      spoof_type
      illumination
      environment

    loss_cfg ví dụ:
      {
        "lambda_spoof": 0.3,
        "lambda_illum": 0.1,
        "lambda_env": 0.1,
        "gamma_proto": 0.05
      }
    """
    device = outputs["pad_logit"].device

    y = targets["label"].to(device).long()
    y_float = y.float()

    loss_pad = F.binary_cross_entropy_with_logits(
        outputs["pad_logit"],
        y_float,
    )

    total_loss = loss_pad

    logs = {
        "loss_pad": float(loss_pad.detach().cpu()),
    }

    lambda_spoof = float(loss_cfg.get("lambda_spoof", 0.0))
    lambda_illum = float(loss_cfg.get("lambda_illum", 0.0))
    lambda_env = float(loss_cfg.get("lambda_env", 0.0))
    gamma_proto = float(loss_cfg.get("gamma_proto", 0.0))

    if lambda_spoof > 0 and "spoof_logits" in outputs:
        spoof_type = targets["spoof_type"].to(device).long()
        loss_spoof = masked_cross_entropy(outputs["spoof_logits"], spoof_type)
        total_loss = total_loss + lambda_spoof * loss_spoof
        logs["loss_spoof"] = float(loss_spoof.detach().cpu())

    if lambda_illum > 0 and "illum_logits" in outputs:
        illumination = targets["illumination"].to(device).long()
        loss_illum = masked_cross_entropy(outputs["illum_logits"], illumination)
        total_loss = total_loss + lambda_illum * loss_illum
        logs["loss_illum"] = float(loss_illum.detach().cpu())

    if lambda_env > 0 and "env_logits" in outputs:
        environment = targets["environment"].to(device).long()
        loss_env = masked_cross_entropy(outputs["env_logits"], environment)
        total_loss = total_loss + lambda_env * loss_env
        logs["loss_env"] = float(loss_env.detach().cpu())

    if gamma_proto > 0 and "proto_logits" in outputs:
        loss_proto = F.cross_entropy(outputs["proto_logits"], y)
        total_loss = total_loss + gamma_proto * loss_proto
        logs["loss_proto"] = float(loss_proto.detach().cpu())

    logs["loss_total"] = float(total_loss.detach().cpu())

    return total_loss, logs