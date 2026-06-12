from pathlib import Path
import argparse
import csv
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.eval import predict_loader
from src.metrics import compute_pad_metrics
from src.models.factory import build_model
from src.train import make_loader
from src.utils import get_device, load_yaml, save_json


DEFAULT_RUNS = [
    "baseline_mbv3",
    "aqb_z64_b8",
    "ablation_noattr",
    "ablation_attr",
    "ablation_proto",
    "aqb_z16_b8",
    "aqb_z32_b8",
    "aqb_z128_b8",
    "aqb_z64_b4",
]

METHOD_NAMES = {
    "baseline_mbv3": "MobileNetV3 baseline",
    "aqb_z64_b8": "AQB-FAS z64/b8",
    "ablation_noattr": "Bottleneck-only",
    "ablation_attr": "+ Attribute heads",
    "ablation_proto": "+ Attribute heads + Prototype",
    "aqb_z16_b8": "AQB-FAS z16/b8",
    "aqb_z32_b8": "AQB-FAS z32/b8",
    "aqb_z128_b8": "AQB-FAS z128/b8",
    "aqb_z64_b4": "AQB-FAS z64/b4",
}


def method_name(run_name: str) -> str:
    return METHOD_NAMES.get(run_name, run_name)


def load_threshold(run_dir: Path, policy: str) -> float:
    policy_path = run_dir / "diagnostics" / "threshold_policy_comparison.csv"
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)

    df = pd.read_csv(policy_path)
    match = df[df["policy"] == policy]
    if match.empty:
        raise ValueError(f"Policy {policy} not found in {policy_path}")

    return float(match.iloc[0]["threshold"])


def prepare_eval_csv(csv_path: Path, max_samples: int | None):
    if max_samples is None:
        return csv_path, None

    df = pd.read_csv(csv_path).reset_index(names="_original_index")
    if "label" in df.columns and df["label"].nunique() > 1:
        per_label = max(1, int(max_samples) // int(df["label"].nunique()))
        remainder = max(0, int(max_samples) - per_label * int(df["label"].nunique()))
        parts = []
        for _, group in df.groupby("label", sort=True):
            take = per_label + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)
            parts.append(group.head(take))
        df = pd.concat(parts, ignore_index=True).head(int(max_samples))
        df = df.sort_values("_original_index")
    else:
        df = df.head(int(max_samples))
    df = df.drop(columns=["_original_index"]).reset_index(drop=True)

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name) / "external_eval_subset.csv"
    df.to_csv(temp_path, index=False)
    return temp_path, temp_dir


def write_predictions(path: Path, source_csv: Path, y_true, y_score, threshold: float) -> None:
    df = pd.read_csv(source_csv).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "image_path",
        "rel_path",
        "label",
        "score",
        "pred",
        "correct",
        "split",
        "source_dataset",
        "source_split",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for index, (label, score) in enumerate(zip(y_true, y_score)):
            row = df.iloc[index]
            pred = int(float(score) >= threshold)
            writer.writerow(
                {
                    "index": index,
                    "image_path": row.get("image_path", ""),
                    "rel_path": row.get("rel_path", ""),
                    "label": int(label),
                    "score": float(score),
                    "pred": pred,
                    "correct": int(pred == int(label)),
                    "split": row.get("split", ""),
                    "source_dataset": row.get("source_dataset", ""),
                    "source_split": row.get("source_split", ""),
                }
            )


def evaluate_run(
    run_name: str,
    eval_csv: Path,
    configs_dir: Path,
    runs_dir: Path,
    policy: str,
    dataset_name: str,
    batch_size_override: int | None,
    num_workers_override: int | None,
    device_override: str | None,
):
    cfg_path = configs_dir / f"{run_name}.yaml"
    if not cfg_path.exists():
        print(f"[SKIP] Missing config: {cfg_path}")
        return None

    cfg = load_yaml(str(cfg_path))
    data_cfg = cfg["data"]
    run_dir = Path(cfg.get("output", {}).get("run_dir", runs_dir / run_name))
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        print(f"[SKIP] Missing checkpoint: {checkpoint_path}")
        return None

    threshold = load_threshold(run_dir, policy)
    device = get_device(device_override if device_override else str(cfg.get("device", "auto")))

    print(f"[EXTERNAL] {run_name}")
    print(f"  checkpoint={checkpoint_path}")
    print(f"  threshold={threshold:.8f} policy={policy}")

    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    loader = make_loader(
        str(eval_csv),
        image_size=int(data_cfg.get("image_size", 224)),
        batch_size=int(batch_size_override or data_cfg.get("batch_size", 32)),
        num_workers=int(num_workers_override if num_workers_override is not None else data_cfg.get("num_workers", 0)),
        train=False,
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )

    y_true, y_score = predict_loader(model, loader, device)
    metrics = compute_pad_metrics(y_true, y_score, threshold=threshold)
    metrics.update(
        {
            "run_name": run_name,
            "method": method_name(run_name),
            "dataset": dataset_name,
            "num_samples": int(len(y_true)),
            "policy": policy,
            "threshold": float(threshold),
            "threshold_source": str(run_dir / "diagnostics" / "threshold_policy_comparison.csv"),
            "checkpoint": str(checkpoint_path),
        }
    )

    out_dir = run_dir / "external_eval" / dataset_name
    save_json(metrics, str(out_dir / "metrics.json"))
    write_predictions(out_dir / "predictions.csv", eval_csv, y_true, y_score, threshold)

    print(
        f"  acer={metrics['acer']:.4f} apcer={metrics['apcer']:.4f} "
        f"bpcer={metrics['bpcer']:.4f} auc={metrics['auc']:.4f}"
    )
    print(f"  [WRITE] {out_dir / 'metrics.json'}")
    print(f"  [WRITE] {out_dir / 'predictions.csv'}")
    return metrics


def write_summary(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = [
        "run_name",
        "method",
        "dataset",
        "num_samples",
        "policy",
        "threshold",
        "acc",
        "precision",
        "recall",
        "f1",
        "auc",
        "apcer",
        "bpcer",
        "acer",
        "tn",
        "fp",
        "fn",
        "tp",
        "checkpoint",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained runs on an external LCC_FASD CSV index.")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/evaluation/evaluation.csv")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--configs-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--policy", default="val_target_bpcer_0.020")
    parser.add_argument("--dataset-name", default="lcc_fasd")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/results/lcc_fasd_external_results.csv")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    eval_csv, temp_dir = prepare_eval_csv(args.csv, args.max_samples)
    dataset_name = args.dataset_name if args.max_samples is None else f"{args.dataset_name}_smoke{args.max_samples}"
    out_path = args.out if args.max_samples is None else args.out.with_name(f"{args.out.stem}_smoke{args.max_samples}{args.out.suffix}")

    try:
        rows = []
        for run_name in args.runs:
            row = evaluate_run(
                run_name=run_name,
                eval_csv=eval_csv,
                configs_dir=args.configs_dir,
                runs_dir=args.runs_dir,
                policy=args.policy,
                dataset_name=dataset_name,
                batch_size_override=args.batch_size,
                num_workers_override=args.num_workers,
                device_override=args.device,
            )
            if row is not None:
                rows.append(row)

        write_summary(out_path, rows)
        print(f"[WRITE] {out_path} rows={len(rows)}")
        print(pd.DataFrame(rows).to_string(index=False))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
