from pathlib import Path
import argparse
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def labels_to_rows(label_json: Path, raw_root: Path):
    with open(label_json, "r", encoding="utf-8") as f:
        labels = json.load(f)

    rows = []
    missing = 0

    for rel_path, values in labels.items():
        image_path = raw_root / rel_path
        if not image_path.exists():
            missing += 1
            continue

        parts = Path(rel_path).parts
        rows.append(
            {
                "image_path": str(image_path),
                "rel_path": rel_path,
                "split": parts[1] if len(parts) > 1 else "",
                "identity": parts[2] if len(parts) > 2 else "",
                "label": int(values[43]),
                "spoof_type": int(values[40]),
                "illumination": int(values[41]),
                "environment": int(values[42]),
            }
        )

    return pd.DataFrame(rows), missing


def balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    per_class = max(n // 2, 1)
    chunks = []
    selected_indices = []

    for label in [0, 1]:
        group = df[df["label"] == label]
        take = min(len(group), per_class)
        sample = group.sample(n=take, random_state=seed)
        chunks.append(sample)
        selected_indices.extend(sample.index.tolist())

    sampled = pd.concat(chunks, ignore_index=True)
    if len(sampled) < n:
        remainder = df.drop(selected_indices, errors="ignore")
        extra = min(n - len(sampled), len(remainder))
        if extra > 0:
            sampled = pd.concat(
                [sampled, remainder.sample(n=extra, random_state=seed)],
                ignore_index=True,
            )

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[WRITE] {path} rows={len(df)}")


def split_train_val_by_identity(df: pd.DataFrame, val_ratio: float, seed: int):
    identities = list(df["identity"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(identities)

    val_count = max(1, int(len(identities) * val_ratio))
    val_ids = set(identities[:val_count])

    is_val = df["identity"].astype(str).isin(val_ids)
    val_df = df[is_val].reset_index(drop=True)
    train_df = df[~is_val].reset_index(drop=True)

    overlap = set(train_df["identity"].astype(str)) & set(val_df["identity"].astype(str))
    if overlap:
        raise RuntimeError(f"Identity leakage detected: {len(overlap)} overlapping ids")

    return train_df, val_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/CelebA-Spoof")
    parser.add_argument("--meta", type=str, default="intra_test")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--debug-size", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Also write full train/val/test CSVs.")
    args = parser.parse_args()

    random.seed(args.seed)

    meta_root = args.raw_root / "metas" / args.meta
    train_json = meta_root / "train_label.json"
    test_json = meta_root / "test_label.json"

    if not train_json.exists() or not test_json.exists():
        raise FileNotFoundError(f"Cannot find train/test labels under {meta_root}")

    train_df, train_missing = labels_to_rows(train_json, args.raw_root)
    test_df, test_missing = labels_to_rows(test_json, args.raw_root)

    print(f"[LOAD] train rows={len(train_df)} missing_files={train_missing}")
    print(f"[LOAD] test rows={len(test_df)} missing_files={test_missing}")

    debug_df = balanced_sample(train_df, args.debug_size, args.seed)
    write_csv(debug_df, args.out_dir / "debug_2k.csv")

    if args.full:
        train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        train_split_df, val_df = split_train_val_by_identity(
            train_df,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        write_csv(train_split_df, args.out_dir / "train.csv")
        write_csv(val_df, args.out_dir / "val.csv")
        write_csv(test_df, args.out_dir / "test.csv")


if __name__ == "__main__":
    main()
