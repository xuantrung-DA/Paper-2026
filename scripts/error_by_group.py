from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def summarize_group(df, group_col):
    rows = []

    for value, g in df.groupby(group_col):
        label = g["label"].astype(int)
        pred = g["pred_policy"].astype(int)
        score = g["score"].astype(float)

        spoof_mask = label == 1
        live_mask = label == 0

        spoof_count = int(spoof_mask.sum())
        live_count = int(live_mask.sum())

        fn_count = int(((label == 1) & (pred == 0)).sum())  # spoof -> live
        fp_count = int(((label == 0) & (pred == 1)).sum())  # live -> spoof

        apcer = fn_count / max(spoof_count, 1)
        bpcer = fp_count / max(live_count, 1)

        if spoof_count > 0 and live_count > 0:
            group_acer = 0.5 * (apcer + bpcer)
        elif spoof_count > 0:
            group_acer = apcer
        elif live_count > 0:
            group_acer = bpcer
        else:
            group_acer = np.nan

        rows.append(
            {
                "group_by": group_col,
                "value": value,
                "total_count": int(len(g)),
                "spoof_count": spoof_count,
                "live_count": live_count,
                "fn_spoof_as_live": fn_count,
                "fp_live_as_spoof": fp_count,
                "group_apcer": apcer,
                "group_bpcer": bpcer,
                "group_error": group_acer,
                "score_mean": float(score.mean()),
                "score_p05": float(score.quantile(0.05)),
                "score_p50": float(score.quantile(0.50)),
                "score_p95": float(score.quantile(0.95)),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--policy", default="val_target_bpcer_0.020")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pred_path = run_dir / "diagnostics" / "test_predictions.csv"
    policy_path = run_dir / "diagnostics" / "threshold_policy_comparison.csv"

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)

    pred_df = pd.read_csv(pred_path)
    policy_df = pd.read_csv(policy_path)

    policy_row = policy_df[policy_df["policy"] == args.policy]
    if policy_row.empty:
        raise ValueError(f"Policy {args.policy} not found in {policy_path}")

    threshold = float(policy_row.iloc[0]["threshold"])

    pred_df["label"] = pred_df["label"].astype(int)
    pred_df["score"] = pred_df["score"].astype(float)
    pred_df["pred_policy"] = (pred_df["score"] >= threshold).astype(int)
    pred_df["correct_policy"] = (pred_df["pred_policy"] == pred_df["label"]).astype(int)

    rows = []
    for group_col in ["spoof_type", "illumination", "environment"]:
        if group_col in pred_df.columns:
            rows.extend(summarize_group(pred_df, group_col))

    out_df = pd.DataFrame(rows)

    if args.out is None:
        out_path = run_dir / "diagnostics" / f"error_by_group_{args.policy}.csv"
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[POLICY] {args.policy}")
    print(f"[THRESHOLD] {threshold}")
    print(out_df.sort_values(["group_by", "group_error"], ascending=[True, False]).to_string(index=False))
    print(f"\n[WRITE] {out_path}")


if __name__ == "__main__":
    main()