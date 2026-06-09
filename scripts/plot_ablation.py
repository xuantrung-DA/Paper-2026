from pathlib import Path
import argparse
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ABLATION_LABELS = {
    "ablation_noattr": "No attr/proto",
    "ablation_attr": "Attr only",
    "ablation_proto": "Attr + proto",
    "aqb_z64_b8": "Full AQB-FAS",
}


def ensure_summary(summary_path: Path):
    if summary_path.exists():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/collect_results.py"), "--out", str(summary_path)],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/results/summary.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/figures/ablation_acer.png")
    args = parser.parse_args()

    ensure_summary(args.summary)
    df = pd.read_csv(args.summary)
    metric_prefix = "test" if "test_acer" in df.columns and df["test_acer"].notna().any() else "val"
    split_name = "Test" if metric_prefix == "test" else "Validation"
    df = df[df["run_name"].isin(ABLATION_LABELS)].copy()
    if df.empty:
        print("[SKIP] no ablation runs found")
        return

    df["label"] = df["run_name"].map(ABLATION_LABELS)
    order = [name for name in ABLATION_LABELS if name in set(df["run_name"])]
    df["_order"] = df["run_name"].map({name: i for i, name in enumerate(order)})
    df = df.sort_values("_order")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(df["label"], df[f"{metric_prefix}_acer"] * 100, color="tab:blue")
    ax.set_ylabel(f"{split_name} ACER (%)")
    ax.set_title("AQB-FAS ablation")
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.xticks(rotation=15, ha="right")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"[WRITE] {args.out}")


if __name__ == "__main__":
    main()
