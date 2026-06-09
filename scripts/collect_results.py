from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config_map(config_dir: Path):
    configs = {}
    for path in config_dir.glob("*.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        run_name = cfg.get("run_name")
        if run_name:
            configs[run_name] = cfg
    return configs


def method_name(run_name: str, cfg: dict):
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", run_name)
    if name == "baseline_mbv3":
        return "Baseline MobileNetV3"
    if name == "aqb_fas":
        dz = model_cfg.get("dz")
        bits = model_cfg.get("bits")
        return f"AQB-FAS z={dz}, b={bits}"
    return name


def bitrate_bits(cfg: dict):
    model_cfg = cfg.get("model", {})
    if model_cfg.get("name") != "aqb_fas":
        return None
    dz = model_cfg.get("dz")
    bits = model_cfg.get("bits")
    if dz is None or bits is None:
        return None
    return int(dz) * int(bits)


def metric_value(metrics: dict, key: str):
    if key in metrics:
        return metrics[key]

    tp = metrics.get("tp")
    fp = metrics.get("fp")
    fn = metrics.get("fn")
    if tp is None or fp is None or fn is None:
        return ""

    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)

    if key == "precision":
        return precision
    if key == "recall":
        return recall
    if key == "f1":
        return 2.0 * precision * recall / max(precision + recall, 1e-12)

    return ""


def collect_run(run_dir: Path, cfg: dict):
    metrics_path = run_dir / "best_metrics.json"
    if not metrics_path.exists():
        return None

    metrics = load_json(metrics_path)
    test_metrics_path = run_dir / "test_metrics.json"
    test = load_json(test_metrics_path) if test_metrics_path.exists() else {}
    val = metrics.get("val_metrics", {})
    run_name = run_dir.name
    model_cfg = cfg.get("model", {})

    return {
        "run_name": run_name,
        "method": method_name(run_name, cfg),
        "model": model_cfg.get("name", ""),
        "dz": model_cfg.get("dz", ""),
        "bits": model_cfg.get("bits", ""),
        "bitrate_bits": bitrate_bits(cfg),
        "best_epoch": metrics.get("epoch", ""),
        "train_loss": metrics.get("train_loss", ""),
        "val_acc": val.get("acc", ""),
        "val_precision": metric_value(val, "precision"),
        "val_recall": metric_value(val, "recall"),
        "val_f1": metric_value(val, "f1"),
        "val_auc": val.get("auc", ""),
        "val_apcer": val.get("apcer", ""),
        "val_bpcer": val.get("bpcer", ""),
        "val_acer": val.get("acer", ""),
        "val_threshold": val.get("threshold", ""),
        "test_acc": test.get("acc", ""),
        "test_precision": metric_value(test, "precision"),
        "test_recall": metric_value(test, "recall"),
        "test_f1": metric_value(test, "f1"),
        "test_auc": test.get("auc", ""),
        "test_apcer": test.get("apcer", ""),
        "test_bpcer": test.get("bpcer", ""),
        "test_acer": test.get("acer", ""),
        "test_threshold": test.get("threshold", ""),
        "checkpoint": str(run_dir / "best.pt") if (run_dir / "best.pt").exists() else "",
        "history": str(run_dir / "history.csv") if (run_dir / "history.csv").exists() else "",
    }


def sort_summary(df: pd.DataFrame):
    if df.empty:
        return df
    df = df.copy()
    df["_is_baseline"] = (df["model"] != "baseline_mbv3").astype(int)
    df["_bitrate"] = pd.to_numeric(df["bitrate_bits"], errors="coerce").fillna(-1)
    df = df.sort_values(["_is_baseline", "_bitrate", "run_name"])
    return df.drop(columns=["_is_baseline", "_bitrate"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/results/summary.csv")
    parser.add_argument("--include-debug", action="store_true")
    args = parser.parse_args()

    config_map = load_config_map(args.config_dir)
    rows = []

    if args.runs_dir.exists():
        for run_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
            if run_dir.name.startswith("debug") and not args.include_debug:
                continue
            cfg = config_map.get(run_dir.name, {})
            row = collect_run(run_dir, cfg)
            if row:
                rows.append(row)

    df = sort_summary(pd.DataFrame(rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[WRITE] {args.out} rows={len(df)}")


if __name__ == "__main__":
    main()
