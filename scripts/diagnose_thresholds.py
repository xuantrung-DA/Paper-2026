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
from src.metrics import compute_pad_metrics, find_best_threshold, find_threshold_by_target_bpcer
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


def resolve_split_csv(configured_csv_path: str, split_name: str) -> Path:
    try:
        validate_csv_schema(configured_csv_path)
        return Path(configured_csv_path)
    except (FileNotFoundError, ValueError) as configured_error:
        fallback_path = ROOT / "data" / f"{split_name}.csv"
        try:
            validate_csv_schema(str(fallback_path))
        except (FileNotFoundError, ValueError):
            raise configured_error

        print(
            f"[WARN] {configured_error} Using fallback CSV: {fallback_path}"
        )
        return fallback_path


def default_raw_root() -> Path:
    candidates = [
        ROOT / "data" / "raw" / "CelebA-Spoof",
        ROOT / "data" / "raw" / "celeba_spoof",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def first_image_path(csv_path: Path):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if row is None:
            return None, None
        return row.get("image_path"), row.get("rel_path")


def prepare_csv_for_loader(csv_path: Path, split_name: str, out_dir: Path, raw_root: Path) -> Path:
    image_path, rel_path = first_image_path(csv_path)
    if image_path and Path(image_path).exists():
        return csv_path

    if not rel_path:
        missing_path = image_path if image_path else "<empty>"
        raise FileNotFoundError(f"First image path does not exist: {missing_path}")

    remapped_first_path = raw_root / rel_path
    if not remapped_first_path.exists():
        raise FileNotFoundError(
            f"First image path does not exist: {image_path}. "
            f"Remapped path also does not exist: {remapped_first_path}. "
            "Pass --raw-root or place CelebA-Spoof under data/raw/CelebA-Spoof."
        )

    df = pd.read_csv(csv_path)
    df["image_path"] = df["rel_path"].map(lambda value: str(raw_root / str(value)))
    resolved_path = out_dir / f"{split_name}_resolved_paths.csv"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved_path, index=False)
    print(f"[WRITE] {resolved_path}")
    return resolved_path


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


def write_predictions(path: Path, csv_path: str, y_true, y_score, threshold: float):
    df = pd.read_csv(csv_path).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "image_path",
            "label",
            "score",
            "pred",
            "correct",
            "spoof_type",
            "illumination",
            "environment",
            "source_split",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for index, (label, score) in enumerate(zip(y_true, y_score)):
            row = df.iloc[index]
            pred = int(float(score) >= threshold)

            writer.writerow(
                {
                    "index": index,
                    "image_path": row.get("image_path", ""),
                    "label": int(label),
                    "score": float(score),
                    "pred": pred,
                    "correct": int(pred == int(label)),
                    "spoof_type": row.get("spoof_type", ""),
                    "illumination": row.get("illumination", ""),
                    "environment": row.get("environment", ""),
                    "source_split": row.get("source_split", row.get("split", "")),
                }
            )

def flatten_metrics(prefix: str, metrics: dict):
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def make_threshold_policy_comparison(
    val_true,
    val_score,
    test_true,
    test_score,
    checkpoint_threshold: float,
    target_bpcers=(0.005, 0.01, 0.02, 0.05),
):
    """
    Tất cả threshold hợp lệ cho paper phải chọn từ val.
    test_oracle chỉ dùng để diagnostic, không dùng làm main result.
    """
    rows = []

    def add_policy(name: str, threshold: float, source: str):
        val_metrics = compute_pad_metrics(val_true, val_score, threshold=threshold)
        test_metrics = compute_pad_metrics(test_true, test_score, threshold=threshold)

        row = {
            "policy": name,
            "threshold": float(threshold),
            "threshold_source": source,
        }
        row.update(flatten_metrics("val", val_metrics))
        row.update(flatten_metrics("test", test_metrics))
        rows.append(row)

    # 1. Fixed threshold
    add_policy("fixed_0_5", 0.5, "fixed")

    # 2. Current checkpoint threshold
    add_policy("checkpoint_threshold", checkpoint_threshold, "best_metrics_json")

    # 3. Val min-ACER threshold
    val_best_th, _ = find_best_threshold(val_true, val_score)
    add_policy("val_min_acer", val_best_th, "val")

    # 4. Val target-BPCER thresholds
    for target in target_bpcers:
        th, _ = find_threshold_by_target_bpcer(
            val_true,
            val_score,
            target_bpcer=float(target),
        )
        add_policy(f"val_target_bpcer_{target:.3f}", th, "val")

    # 5. Test oracle diagnostic only
    test_oracle_th, _ = find_best_threshold(test_true, test_score)
    add_policy("test_oracle_do_not_report_as_main", test_oracle_th, "test_oracle")

    return rows


def write_policy_comparison_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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
    parser.add_argument("--raw-root", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    run_dir = Path(cfg["output"]["run_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "diagnostics"
    raw_root = Path(args.raw_root) if args.raw_root else default_raw_root()

    try:
        split_csv_paths = {
            "val": resolve_split_csv(data_cfg["val_csv"], "val"),
            "test": resolve_split_csv(data_cfg["test_csv"], "test"),
        }
        split_csv_paths = {
            split_name: prepare_csv_for_loader(csv_path, split_name, out_dir, raw_root)
            for split_name, csv_path in split_csv_paths.items()
        }
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
    raw_scores = {}

    for split_name, csv_key in (("val", "val_csv"), ("test", "test_csv")):
        result = diagnose_split(model, split_name, str(split_csv_paths[split_name]), cfg, threshold, device)
        raw_scores[split_name] = {
            "y_true": result["y_true"],
            "y_score": result["y_score"],
        }
        report["splits"][split_name] = {
            key: value
            for key, value in result.items()
            if key not in {"y_true", "y_score"}
        }
        write_predictions(
            out_dir / f"{split_name}_predictions.csv",
            str(split_csv_paths[split_name]),
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

    policy_rows = make_threshold_policy_comparison(
        val_true=raw_scores["val"]["y_true"],
        val_score=raw_scores["val"]["y_score"],
        test_true=raw_scores["test"]["y_true"],
        test_score=raw_scores["test"]["y_score"],
        checkpoint_threshold=threshold,
    )

    report["threshold_policy_comparison"] = policy_rows

    policy_csv_path = out_dir / "threshold_policy_comparison.csv"
    write_policy_comparison_csv(policy_csv_path, policy_rows)

    print("\n[THRESHOLD POLICY COMPARISON]")
    for row in policy_rows:
        print(
            f"{row['policy']}: th={row['threshold']:.8f} "
            f"test_acer={row['test_acer']:.4f} "
            f"test_apcer={row['test_apcer']:.4f} "
            f"test_bpcer={row['test_bpcer']:.4f} "
            f"test_auc={row['test_auc']:.4f}"
        )

    print(f"[WRITE] {policy_csv_path}")

    save_json(report, str(out_dir / "threshold_diagnostics.json"))
    print(f"[WRITE] {out_dir / 'threshold_diagnostics.json'}")
    print(f"[WRITE] {out_dir / 'val_predictions.csv'}")
    print(f"[WRITE] {out_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()