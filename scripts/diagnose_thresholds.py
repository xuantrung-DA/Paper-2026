from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.eval import predict_loader
from src.metrics import compute_pad_metrics, find_best_threshold
from src.models.factory import build_model
from src.train import make_loader
from src.utils import get_device, load_yaml, save_json


REQUIRED_COLUMNS = {"image_path", "label", "spoof_type", "illumination", "environment"}


def validate_csv_schema(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns {sorted(missing)}. "
            "Rebuild processed indexes with: python scripts/build_index.py --full"
        )


def load_checkpoint_threshold(run_dir: Path):
    metrics_path = run_dir / "best_metrics.json"
    if not metrics_path.exists():
        return None, "missing_best_metrics"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    try:
        return float(metrics["val_metrics"]["threshold"]), "val_best_metrics"
    except (KeyError, TypeError, ValueError):
        return None, "missing_val_threshold"


def score_summary(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    summary = {
        "count": int(y_score.size),
        "score_min": float(np.min(y_score)) if y_score.size else float("nan"),
        "score_p01": float(np.quantile(y_score, 0.01)) if y_score.size else float("nan"),
        "score_p05": float(np.quantile(y_score, 0.05)) if y_score.size else float("nan"),
        "score_p50": float(np.quantile(y_score, 0.50)) if y_score.size else float("nan"),
        "score_p95": float(np.quantile(y_score, 0.95)) if y_score.size else float("nan"),
        "score_p99": float(np.quantile(y_score, 0.99)) if y_score.size else float("nan"),
        "score_max": float(np.max(y_score)) if y_score.size else float("nan"),
    }

    for label, name in ((0, "live"), (1, "spoof")):
        scores = y_score[y_true == label]
        summary[f"{name}_count"] = int(scores.size)
        summary[f"{name}_mean"] = float(np.mean(scores)) if scores.size else float("nan")
        summary[f"{name}_p05"] = float(np.quantile(scores, 0.05)) if scores.size else float("nan")
        summary[f"{name}_p50"] = float(np.quantile(scores, 0.50)) if scores.size else float("nan")
        summary[f"{name}_p95"] = float(np.quantile(scores, 0.95)) if scores.size else float("nan")

    return summary


def write_predictions(path: Path, y_true, y_score, threshold: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "label", "score", "pred", "correct"])
        writer.writeheader()
        for index, (label, score) in enumerate(zip(y_true, y_score)):
            pred = int(float(score) >= threshold)
            writer.writerow(
                {
                    "index": index,
                    "label": int(label),
                    "score": float(score),
                    "pred": pred,
                    "correct": int(pred == int(label)),
                }
            )


def diagnose_split(model, split_name: str, csv_path: str, cfg: dict, threshold: float, device):
    data_cfg = cfg["data"]
    validate_csv_schema(csv_path)
    loader = make_loader(
        csv_path,
        image_size=int(data_cfg.get("image_size", 224)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        train=False,
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )

    y_true, y_score = predict_loader(model, loader, device)
    best_threshold, best_metrics = find_best_threshold(y_true, y_score)

    return {
        "split": split_name,
        "metrics_at_checkpoint_threshold": compute_pad_metrics(y_true, y_score, threshold=threshold),
        "metrics_at_0_5": compute_pad_metrics(y_true, y_score, threshold=0.5),
        "best_threshold_on_split": float(best_threshold),
        "metrics_at_best_split_threshold": best_metrics,
        "score_summary": score_summary(y_true, y_score),
        "y_true": y_true,
        "y_score": y_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    run_dir = Path(cfg["output"]["run_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "diagnostics"

    try:
        validate_csv_schema(data_cfg["val_csv"])
        validate_csv_schema(data_cfg["test_csv"])
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    checkpoint_threshold, threshold_source = load_checkpoint_threshold(run_dir)
    threshold = args.threshold if args.threshold is not None else checkpoint_threshold
    if threshold is None:
        threshold = 0.5
        threshold_source = "default_0.5"
    elif args.threshold is not None:
        threshold_source = "cli"

    device = get_device(str(cfg.get("device", "auto")))
    print(f"[DEVICE] {device}")
    print(f"[CHECKPOINT] {checkpoint_path}")
    print(f"[THRESHOLD] {threshold:.6f} source={threshold_source}")

    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    report = {
        "checkpoint": str(checkpoint_path),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "splits": {},
    }

    for split_name, csv_key in (("val", "val_csv"), ("test", "test_csv")):
        result = diagnose_split(model, split_name, data_cfg[csv_key], cfg, threshold, device)
        report["splits"][split_name] = {
            key: value
            for key, value in result.items()
            if key not in {"y_true", "y_score"}
        }
        write_predictions(
            out_dir / f"{split_name}_predictions.csv",
            result["y_true"],
            result["y_score"],
            threshold,
        )

        checkpoint_metrics = result["metrics_at_checkpoint_threshold"]
        split_best_metrics = result["metrics_at_best_split_threshold"]
        print(
            f"[{split_name.upper()}] checkpoint_th={threshold:.4f} "
            f"acer={checkpoint_metrics['acer']:.4f} apcer={checkpoint_metrics['apcer']:.4f} "
            f"bpcer={checkpoint_metrics['bpcer']:.4f} auc={checkpoint_metrics['auc']:.4f} | "
            f"best_split_th={result['best_threshold_on_split']:.4f} "
            f"best_split_acer={split_best_metrics['acer']:.4f}"
        )

    save_json(report, str(out_dir / "threshold_diagnostics.json"))
    print(f"[WRITE] {out_dir / 'threshold_diagnostics.json'}")
    print(f"[WRITE] {out_dir / 'val_predictions.csv'}")
    print(f"[WRITE] {out_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()