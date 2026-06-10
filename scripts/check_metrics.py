from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.metrics import compute_pad_metrics, find_best_threshold, latent_bits, raw_image_bits


def main():
    # y_true:
    # 0,0 là live
    # 1,1 là spoof
    y_true = np.array([0, 0, 1, 1])

    # threshold = 0.5 -> prediction:
    # score 0.1 -> live đúng
    # score 0.8 -> live bị nhầm spoof => FP
    # score 0.2 -> spoof bị nhầm live => FN
    # score 0.9 -> spoof đúng
    y_score = np.array([0.1, 0.8, 0.2, 0.9])

    metrics = compute_pad_metrics(y_true, y_score, threshold=0.5)

    print(metrics)

    assert abs(metrics["apcer"] - 0.5) < 1e-8
    assert abs(metrics["bpcer"] - 0.5) < 1e-8
    assert abs(metrics["acer"] - 0.5) < 1e-8

    best_th, best_metrics = find_best_threshold(y_true, y_score)

    print("best_threshold:", best_th)
    print("best_metrics:", best_metrics)

    tie_y_true = np.array([0, 1])
    tie_y_score = np.array([0.2, 0.8])
    tie_best_th, tie_best_metrics = find_best_threshold(tie_y_true, tie_y_score)

    print("tie_best_threshold:", tie_best_th)
    print("tie_best_metrics:", tie_best_metrics)

    assert abs(tie_best_th - 0.5) < 1e-12
    assert abs(tie_best_metrics["acer"] - 0.0) < 1e-12

    assert latent_bits(64, 8) == 512
    assert raw_image_bits(224, 224, 3, 8) == 1204224

    print("[DONE] Metrics check passed.")


if __name__ == "__main__":
    main()