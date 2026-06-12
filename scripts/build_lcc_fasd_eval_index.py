from pathlib import Path
import argparse
import csv
from collections import Counter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = {
    "real": 0,
    "spoof": 1,
}


def iter_rows(root: Path):
    root = root.resolve()

    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        source_split = split_dir.name
        for class_name, label in LABELS.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"[WARN] Missing class folder: {class_dir}")
                continue

            for image_path in sorted(path for path in class_dir.rglob("*") if path.is_file()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                yield {
                    "image_path": str(image_path.resolve()),
                    "rel_path": image_path.resolve().relative_to(root).as_posix(),
                    "split": "evaluation",
                    "label": int(label),
                    "spoof_type": 0,
                    "illumination": 0,
                    "environment": 0,
                    "source_dataset": "LCC_FASD",
                    "source_split": source_split,
                }


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "rel_path",
        "split",
        "label",
        "spoof_type",
        "illumination",
        "environment",
        "source_dataset",
        "source_split",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Build a CSV index for the LCC_FASD external evaluation set.")
    parser.add_argument("--root", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--out", type=Path, default=Path("data/evaluation/evaluation.csv"))
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(args.root)

    rows = write_csv(args.out, iter_rows(args.root))
    counts = Counter(row["label"] for row in rows)
    split_counts = Counter((row["source_split"], row["label"]) for row in rows)

    missing_paths = sum(1 for row in rows if not Path(row["image_path"]).exists())
    print(f"[WRITE] {args.out} rows={len(rows)}")
    print(f"[COUNT] real={counts.get(0, 0)} spoof={counts.get(1, 0)} missing_paths={missing_paths}")

    for (source_split, label), count in sorted(split_counts.items()):
        name = "real" if label == 0 else "spoof"
        print(f"[SPLIT] {source_split}/{name}: {count}")

    if not rows:
        raise RuntimeError(f"No images found under {args.root}")
    if missing_paths:
        raise RuntimeError(f"Found {missing_paths} rows with missing image paths")


if __name__ == "__main__":
    main()
