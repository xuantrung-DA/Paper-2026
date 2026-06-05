from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from src.metrics import compute_pad_metrics, find_best_threshold


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[list, list]:
    model.eval()

    y_true = []
    y_score = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        scores = torch.sigmoid(outputs["pad_logit"])

        y_true.extend(targets["label"].cpu().numpy().tolist())
        y_score.extend(scores.detach().cpu().numpy().tolist())

    return y_true, y_score


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: Optional[float] = None,
    choose_threshold: bool = False,
) -> Dict[str, float]:
    y_true, y_score = predict_loader(model, loader, device)

    if choose_threshold:
        best_threshold, metrics = find_best_threshold(y_true, y_score)
        metrics["threshold"] = float(best_threshold)
        return metrics

    if threshold is None:
        threshold = 0.5

    return compute_pad_metrics(y_true, y_score, threshold=threshold)