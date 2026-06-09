from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

import pandas as pd


def ensure_summary(summary_path: Path):
    if summary_path.exists():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/collect_results.py"), "--out", str(summary_path)],
        cwd=ROOT,
        check=True,
    )


def fmt_percent(value):
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.3f}"


def build_main_table(summary: pd.DataFrame):
    if summary.empty:
        return pd.DataFrame()

    df = summary.copy()
    metric_prefix = "test" if "test_acer" in df.columns and df["test_acer"].notna().any() else "val"
    split_name = "Test" if metric_prefix == "test" else "Val"
    table = pd.DataFrame(
        {
            "Method": df["method"],
            "Latent bits": df["bitrate_bits"].fillna("Full image"),
            "Best epoch": df["best_epoch"],
            f"{split_name} ACC (%)": df[f"{metric_prefix}_acc"].map(fmt_percent),
            f"{split_name} Precision (%)": df[f"{metric_prefix}_precision"].map(fmt_percent),
            f"{split_name} Recall (%)": df[f"{metric_prefix}_recall"].map(fmt_percent),
            f"{split_name} F1 (%)": df[f"{metric_prefix}_f1"].map(fmt_percent),
            f"{split_name} AUC (%)": df[f"{metric_prefix}_auc"].map(fmt_percent),
            f"{split_name} APCER (%)": df[f"{metric_prefix}_apcer"].map(fmt_percent),
            f"{split_name} BPCER (%)": df[f"{metric_prefix}_bpcer"].map(fmt_percent),
            f"{split_name} ACER (%)": df[f"{metric_prefix}_acer"].map(fmt_percent),
            "Val ACER (%)": df["val_acer"].map(fmt_percent),
        }
    )
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/results/summary.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/tables")
    args = parser.parse_args()

    ensure_summary(args.summary)
    summary = pd.read_csv(args.summary)
    table = build_main_table(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "main_results.csv"
    tex_path = args.out_dir / "main_results.tex"

    table.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=True))

    print(f"[WRITE] {csv_path} rows={len(table)}")
    print(f"[WRITE] {tex_path}")


if __name__ == "__main__":
    main()
