from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
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
        "precision": float(precision),
        "recall": float(recall),
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
    num_thresholds: int = 999,
    prefer_balanced: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Chọn threshold trên validation set theo ACER thấp nhất.

    Tie-break:
      1. ACER nhỏ hơn.
      2. Nếu ACER gần bằng nhau, chọn threshold có |APCER - BPCER| nhỏ hơn.
      3. Nếu vẫn gần bằng nhau, chọn threshold gần 0.5 hơn để tránh threshold cực đoan.
    """
    thresholds = np.linspace(0.001, 0.999, num_thresholds)

    best_threshold = 0.5
    best_metrics = None
    eps = 1e-12

    for th in thresholds:
        metrics = compute_pad_metrics(y_true, y_score, threshold=float(th))

        if best_metrics is None:
            best_threshold = float(th)
            best_metrics = metrics
            continue

        cur_acer = metrics["acer"]
        best_acer = best_metrics["acer"]

        better_acer = cur_acer < best_acer - eps

        if prefer_balanced:
            cur_balance = abs(metrics["apcer"] - metrics["bpcer"])
            best_balance = abs(best_metrics["apcer"] - best_metrics["bpcer"])
            better_tie = (
                abs(cur_acer - best_acer) <= eps
                and (
                    cur_balance < best_balance - eps
                    or (
                        abs(cur_balance - best_balance) <= eps
                        and abs(float(th) - 0.5) < abs(best_threshold - 0.5)
                    )
                )
            )
        else:
            better_tie = (
                abs(cur_acer - best_acer) <= eps
                and abs(float(th) - 0.5) < abs(best_threshold - 0.5)
            )

        if better_acer or better_tie:
            best_threshold = float(th)
            best_metrics = metrics

    return best_threshold, best_metrics


def find_threshold_by_target_bpcer(
    y_true,
    y_score,
    target_bpcer: float = 0.02,
) -> Tuple[float, Dict[str, float]]:
    y_score = np.asarray(y_score).astype(float)

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 1001),
                np.logspace(-8, 0, 1000),
            ]
        )
    )
    thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]

    all_metrics = [
        compute_pad_metrics(y_true, y_score, threshold=float(th))
        for th in thresholds
    ]
    valid = [m for m in all_metrics if m["bpcer"] <= target_bpcer]

    if valid:
        best = min(valid, key=lambda m: m["threshold"])
    else:
        best = min(all_metrics, key=lambda m: (m["bpcer"], m["acer"]))

    return float(best["threshold"]), best


def latent_bits(dz: int, bits: int) -> int:
    return int(dz) * int(bits)


def raw_image_bits(h: int = 224, w: int = 224, c: int = 3, bits: int = 8) -> int:
    return h * w * c * bits
