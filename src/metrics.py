from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def compute_pad_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Quy ước:
      y_true = 0 live, 1 spoof
      y_score = probability of spoof

    APCER:
      spoof bị predict nhầm thành live / tổng spoof

    BPCER:
      live bị predict nhầm thành spoof / tổng live

    ACER:
      (APCER + BPCER) / 2
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # spoof/attack là positive class
    apcer = fn / max(fn + tp, 1)  # attack classified as live
    bpcer = fp / max(fp + tn, 1)  # live classified as attack
    acer = 0.5 * (apcer + bpcer)

    return {
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(
    y_true,
    y_score,
    num_thresholds: int = 199,
) -> Tuple[float, Dict[str, float]]:
    """
    Chọn threshold trên validation set theo ACER thấp nhất.
    Tuyệt đối không chọn threshold bằng test set.
    """
    thresholds = np.linspace(0.001, 0.999, num_thresholds)

    best_threshold = 0.5
    best_metrics = None

    for th in thresholds:
        metrics = compute_pad_metrics(y_true, y_score, threshold=float(th))

        if best_metrics is None:
            best_threshold = float(th)
            best_metrics = metrics
            continue

        if metrics["acer"] < best_metrics["acer"]:
            best_threshold = float(th)
            best_metrics = metrics

    return best_threshold, best_metrics


def latent_bits(dz: int, bits: int) -> int:
    return int(dz) * int(bits)


def raw_image_bits(h: int = 224, w: int = 224, c: int = 3, bits: int = 8) -> int:
    return h * w * c * bits