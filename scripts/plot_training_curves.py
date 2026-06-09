from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def plot_run(history_path: Path, out_dir: Path):
    df = pd.read_csv(history_path)
    if df.empty:
        return None

    run_name = history_path.parent.name
    out_path = out_dir / f"training_curves_{run_name}.png"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    epoch = df["epoch"] + 1

    axes[0].plot(epoch, df["train_loss"], marker="o", linewidth=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    if "val_acer" in df:
        axes[1].plot(epoch, df["val_acer"] * 100, marker="o", linewidth=2)
    axes[1].set_title("Validation ACER")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ACER (%)")
    axes[1].grid(True, alpha=0.3)

    if "val_auc" in df:
        axes[2].plot(epoch, df["val_auc"] * 100, marker="o", linewidth=2)
    axes[2].set_title("Validation AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC (%)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(run_name)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/figures")
    parser.add_argument("--include-debug", action="store_true")
    args = parser.parse_args()

    written = []
    for history_path in sorted(args.runs_dir.glob("*/history.csv")):
        if history_path.parent.name.startswith("debug") and not args.include_debug:
            continue
        out_path = plot_run(history_path, args.out_dir)
        if out_path:
            written.append(out_path)
            print(f"[WRITE] {out_path}")

    if not written:
        print("[SKIP] no history.csv files found")


if __name__ == "__main__":
    main()
