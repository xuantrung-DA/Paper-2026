from pathlib import Path
import argparse
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/figures/utility_bitrate.png")
    args = parser.parse_args()

    ensure_summary(args.summary)
    df = pd.read_csv(args.summary)
    metric_prefix = "test" if "test_acer" in df.columns and df["test_acer"].notna().any() else "val"
    split_name = "Test" if metric_prefix == "test" else "Validation"
    aqb = df[df["model"] == "aqb_fas"].copy()
    aqb["bitrate_bits"] = pd.to_numeric(aqb["bitrate_bits"], errors="coerce")
    aqb = aqb.dropna(subset=["bitrate_bits"]).sort_values("bitrate_bits")

    if aqb.empty:
        print("[SKIP] no AQB-FAS runs with bitrate found")
        return

    baseline = df[df["model"] == "baseline_mbv3"].head(1)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(
        aqb["bitrate_bits"],
        aqb[f"{metric_prefix}_acer"] * 100,
        marker="o",
        linewidth=2,
        label=f"AQB-FAS {split_name} ACER",
    )
    ax1.set_xlabel("Latent bitrate (bits)")
    ax1.set_ylabel("ACER (%)")
    ax1.grid(True, alpha=0.3)

    if not baseline.empty:
        baseline_acer = float(baseline.iloc[0][f"{metric_prefix}_acer"]) * 100
        ax1.axhline(
            baseline_acer,
            linestyle="--",
            color="tab:red",
            label=f"Baseline {split_name} ACER",
        )

    ax2 = ax1.twinx()
    ax2.plot(
        aqb["bitrate_bits"],
        aqb[f"{metric_prefix}_auc"] * 100,
        marker="s",
        linewidth=2,
        color="tab:green",
        label=f"AQB-FAS {split_name} AUC",
    )
    ax2.set_ylabel("AUC (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"[WRITE] {args.out}")


if __name__ == "__main__":
    main()
