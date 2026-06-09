from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_recall_f1(metrics: dict):
    tp = float(metrics["tp"])
    fp = float(metrics["fp"])
    fn = float(metrics["fn"])
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/results/summary.csv")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/figures/confusion_matrices.png")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    rows = []
    for _, row in summary.iterrows():
        metrics_path = args.runs_dir / row["run_name"] / "test_metrics.json"
        if not metrics_path.exists():
            continue
        rows.append((row["run_name"], row["method"], load_json(metrics_path)))

    if not rows:
        print("[SKIP] no test_metrics.json files found")
        return

    fig, axes = plt.subplots(1, len(rows), figsize=(5.5 * len(rows), 4.8))
    if len(rows) == 1:
        axes = [axes]

    for ax, (run_name, method, metrics) in zip(axes, rows):
        cm = [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ]
        precision, recall, f1 = precision_recall_f1(metrics)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(run_name)
        ax.set_xticks([0, 1], ["pred live", "pred spoof"])
        ax.set_yticks([0, 1], ["true live", "true spoof"])
        ax.set_xlabel(
            f"Precision: {precision * 100:.2f}%\n"
            f"Recall: {recall * 100:.2f}%\n"
            f"F1: {f1 * 100:.2f}%"
        )
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Test confusion matrices")
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {args.out}")


if __name__ == "__main__":
    main()
