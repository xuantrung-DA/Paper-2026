from pathlib import Path
import argparse
import pandas as pd
import yaml


RUNS = [
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


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_method_name(run_name):
    names = {
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
    return names.get(run_name, run_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--policy", default="val_target_bpcer_0.020")
    parser.add_argument("--out", default="outputs/tables/main_results_bpcer2.csv")
    args = parser.parse_args()

    rows = []

    for run_name in RUNS:
        run_dir = Path(args.runs_dir) / run_name
        csv_path = run_dir / "diagnostics" / "threshold_policy_comparison.csv"
        cfg_path = Path(args.configs_dir) / f"{run_name}.yaml"

        if not csv_path.exists():
            print(f"[SKIP] Missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        match = df[df["policy"] == args.policy]

        if match.empty:
            print(f"[SKIP] Policy {args.policy} not found in {csv_path}")
            continue

        row = match.iloc[0].to_dict()

        cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
        model_cfg = cfg.get("model", {})

        dz = model_cfg.get("dz", None)
        bits = model_cfg.get("bits", None)

        if dz is not None and bits is not None:
            latent_bits = int(dz) * int(bits)
        else:
            latent_bits = None

        rows.append(
            {
                "run_name": run_name,
                "method": infer_method_name(run_name),
                "dz": dz if dz is not None else "-",
                "bits": bits if bits is not None else "-",
                "latent_bits": latent_bits if latent_bits is not None else "-",
                "policy": args.policy,
                "threshold": row["threshold"],
                "test_acc": row.get("test_acc"),
                "test_precision": row.get("test_precision"),
                "test_recall": row.get("test_recall"),
                "test_f1": row.get("test_f1"),
                "test_auc": row.get("test_auc"),
                "test_apcer": row.get("test_apcer"),
                "test_bpcer": row.get("test_bpcer"),
                "test_acer": row.get("test_acer"),
                "test_tn": row.get("test_tn"),
                "test_fp": row.get("test_fp"),
                "test_fn": row.get("test_fn"),
                "test_tp": row.get("test_tp"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(out_df.to_string(index=False))
    print(f"\n[WRITE] {out_path}")


if __name__ == "__main__":
    main()