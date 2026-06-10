from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.eval import evaluate_loader
from src.models.factory import build_model
from src.train import make_loader
from src.utils import get_device, load_yaml, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    output_cfg = cfg["output"]

    run_dir = Path(output_cfg["run_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    best_metrics_path = run_dir / "best_metrics.json"
    out_path = Path(args.out) if args.out else run_dir / "test_metrics.json"

    device = get_device(str(cfg.get("device", "auto")))
    print(f"[DEVICE] {device}")
    print(f"[CHECKPOINT] {checkpoint_path}")

    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loader = make_loader(
        data_cfg["test_csv"],
        image_size=int(data_cfg.get("image_size", 224)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        train=False,
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )

    threshold = args.threshold
    threshold_source = "cli"

    if threshold is None and best_metrics_path.exists():
        import json

        with open(best_metrics_path, "r", encoding="utf-8") as f:
            best_metrics = json.load(f)
        threshold = float(best_metrics["val_metrics"]["threshold"])
        threshold_source = "val_best_metrics"

    if threshold is None:
        threshold = 0.5
        threshold_source = "default_0.5"

    metrics = evaluate_loader(model, test_loader, device, threshold=threshold)
    metrics["threshold_source"] = threshold_source
    save_json(metrics, str(out_path))

    print(
        f"[TEST] acer={metrics['acer']:.4f} auc={metrics['auc']:.4f} "
        f"acc={metrics['acc']:.4f} threshold={metrics['threshold']:.4f}"
    )
    print(f"[WRITE] {out_path}")


if __name__ == "__main__":
    main()
