from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


DEFAULT_TARGETS_MS = [5.0, 10.0, 20.0, 33.3, 50.0]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def merge_budget_tables(results_df: pd.DataFrame, latency_df: pd.DataFrame) -> pd.DataFrame:
    results_df = results_df.copy()
    latency_df = latency_df.copy()

    keep_latency_cols = [
        "run_name",
        "device",
        "batch_size",
        "image_size",
        "amp",
        "latency_ms_mean",
        "latency_ms_median",
        "latency_ms_p95",
        "latency_ms_std",
        "latency_ms_per_image",
        "throughput_fps",
        "params",
        "checkpoint_mb",
        "raw_image_bits",
        "compression_ratio",
    ]
    keep_latency_cols = [col for col in keep_latency_cols if col in latency_df.columns]

    merged = results_df.merge(
        latency_df[keep_latency_cols],
        on="run_name",
        how="left",
        suffixes=("", "_latency"),
    )

    merged = numeric(
        merged,
        [
            "latent_bits",
            "test_acc",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
            "test_apcer",
            "test_bpcer",
            "test_acer",
            "latency_ms_mean",
            "latency_ms_p95",
            "latency_ms_per_image",
            "throughput_fps",
            "params",
            "checkpoint_mb",
            "compression_ratio",
        ],
    )

    return merged


def candidate_frame(df: pd.DataFrame, include_baseline: bool) -> pd.DataFrame:
    candidates = df.copy()
    if not include_baseline and "latent_bits" in candidates.columns:
        candidates = candidates[candidates["latent_bits"].notna()]
    return candidates


def choose_budget(
    latency_target_ms: float,
    budget_df: pd.DataFrame,
    metric: str = "test_acer",
    latency_column: str = "latency_ms_per_image",
    include_baseline: bool = False,
) -> Dict:
    candidates = candidate_frame(budget_df, include_baseline=include_baseline)
    candidates = candidates[candidates[latency_column].notna()]

    if candidates.empty:
        return {
            "target_latency_ms": float(latency_target_ms),
            "status": "no_latency_rows",
        }

    feasible = candidates[candidates[latency_column] <= float(latency_target_ms)]
    if feasible.empty:
        chosen = candidates.sort_values([latency_column, metric], ascending=[True, True]).iloc[0]
        status = "fastest_available_exceeds_target"
    else:
        chosen = feasible.sort_values([metric, latency_column], ascending=[True, True]).iloc[0]
        status = "meets_target"

    return {
        "target_latency_ms": float(latency_target_ms),
        "status": status,
        "run_name": chosen.get("run_name", ""),
        "method": chosen.get("method", ""),
        "dz": chosen.get("dz", ""),
        "bits": chosen.get("bits", ""),
        "latent_bits": chosen.get("latent_bits", ""),
        "latency_ms_per_image": chosen.get(latency_column, ""),
        "latency_ms_p95": chosen.get("latency_ms_p95", ""),
        "throughput_fps": chosen.get("throughput_fps", ""),
        "test_acer": chosen.get("test_acer", ""),
        "test_apcer": chosen.get("test_apcer", ""),
        "test_bpcer": chosen.get("test_bpcer", ""),
        "test_auc": chosen.get("test_auc", ""),
    }


def write_json(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def sort_budget_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "latent_bits" in df.columns:
        df["_latent_sort"] = pd.to_numeric(df["latent_bits"], errors="coerce").fillna(-1)
    else:
        df["_latent_sort"] = -1

    if "test_acer" in df.columns:
        df["_acer_sort"] = pd.to_numeric(df["test_acer"], errors="coerce").fillna(999)
    else:
        df["_acer_sort"] = 999

    df = df.sort_values(["_latent_sort", "_acer_sort", "run_name"])
    return df.drop(columns=["_latent_sort", "_acer_sort"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge paper metrics with latency and choose adaptive inference budgets."
    )
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/tables/main_results_bpcer2.csv")
    parser.add_argument("--latency", type=Path, default=ROOT / "outputs/results/latency.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "outputs/results/infer_budget_summary.csv")
    parser.add_argument("--out-choices", type=Path, default=ROOT / "outputs/results/infer_budget_choices.csv")
    parser.add_argument("--json-out", type=Path, default=ROOT / "outputs/results/infer_budget_choices.json")
    parser.add_argument("--targets-ms", nargs="*", type=float, default=DEFAULT_TARGETS_MS)
    parser.add_argument("--metric", default="test_acer")
    parser.add_argument("--latency-column", default="latency_ms_per_image")
    parser.add_argument("--include-baseline-in-choice", action="store_true")
    args = parser.parse_args()

    results_df = read_csv(args.results)
    latency_df = read_csv(args.latency)
    summary_df = sort_budget_summary(merge_budget_tables(results_df, latency_df))

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False)

    choices = [
        choose_budget(
            latency_target_ms=target,
            budget_df=summary_df,
            metric=args.metric,
            latency_column=args.latency_column,
            include_baseline=args.include_baseline_in_choice,
        )
        for target in args.targets_ms
    ]
    choices_df = pd.DataFrame(choices)

    args.out_choices.parent.mkdir(parents=True, exist_ok=True)
    choices_df.to_csv(args.out_choices, index=False)
    write_json(args.json_out, choices)

    print(f"[WRITE] {args.out_summary} rows={len(summary_df)}")
    print(f"[WRITE] {args.out_choices} rows={len(choices_df)}")
    print(f"[WRITE] {args.json_out}")
    print(choices_df.to_string(index=False))


if __name__ == "__main__":
    main()
