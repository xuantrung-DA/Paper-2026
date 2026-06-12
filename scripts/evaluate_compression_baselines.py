from pathlib import Path
import argparse
import csv
import io
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, features
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.metrics import compute_pad_metrics, find_threshold_by_target_bpcer, latent_bits, raw_image_bits
from src.models.factory import build_model
from src.utils import get_device, load_yaml, save_json

ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_RUNS = ["baseline_mbv3"]
DEFAULT_CODECS = ["jpeg", "webp"]
DEFAULT_QUALITIES = [10, 30, 50]
REQUIRED_COLUMNS = {"image_path", "label", "spoof_type", "illumination", "environment"}

METHOD_NAMES = {
    "baseline_mbv3": "MobileNetV3",
    "aqb_z64_b8": "AQB-FAS z64/b8",
    "ablation_noattr": "Bottleneck-only",
    "ablation_attr": "+ Attribute heads",
    "ablation_proto": "+ Attribute heads + Prototype",
    "aqb_z16_b8": "AQB-FAS z16/b8",
    "aqb_z32_b8": "AQB-FAS z32/b8",
    "aqb_z128_b8": "AQB-FAS z128/b8",
    "aqb_z64_b4": "AQB-FAS z64/b4",
}


def method_name(run_name: str) -> str:
    return METHOD_NAMES.get(run_name, run_name)


def validate_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")


def subset_frame(df: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df.reset_index(drop=True)

    if "label" not in df.columns or df["label"].nunique() <= 1:
        return df.head(int(max_samples)).reset_index(drop=True)

    indexed = df.reset_index(names="_original_index")
    labels = list(indexed.groupby("label", sort=True))
    per_label = max(1, int(max_samples) // len(labels))
    remainder = max(0, int(max_samples) - per_label * len(labels))

    parts = []
    for _, group in labels:
        take = per_label + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        parts.append(group.head(take))

    out = pd.concat(parts, ignore_index=True).head(int(max_samples))
    out = out.sort_values("_original_index").drop(columns=["_original_index"])
    return out.reset_index(drop=True)


def post_decode_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def encode_decode_resized_image(image: Image.Image, codec: str, quality: int, image_size: int):
    image = image.resize((image_size, image_size), Image.BILINEAR)

    if codec == "raw":
        return image, raw_image_bits(h=image_size, w=image_size, c=3, bits=8)

    buffer = io.BytesIO()
    if codec == "jpeg":
        image.save(buffer, format="JPEG", quality=int(quality))
    elif codec == "webp":
        image.save(buffer, format="WEBP", quality=int(quality), lossless=False, method=4)
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    encoded = buffer.getvalue()
    decoded = Image.open(io.BytesIO(encoded)).convert("RGB")
    return decoded, len(encoded) * 8


class CompressedImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        codec: str,
        quality: int,
        image_size: int,
        max_samples: int | None = None,
        source_split: str | None = None,
    ):
        validate_csv(csv_path)
        df = pd.read_csv(csv_path)
        if source_split:
            if "source_split" not in df.columns:
                raise ValueError(f"{csv_path} has no source_split column")
            df = df[df["source_split"] == source_split].reset_index(drop=True)
            if df.empty:
                raise ValueError(f"No rows found in {csv_path} for source_split={source_split}")

        self.df = subset_frame(df, max_samples)
        self.codec = codec
        self.quality = int(quality)
        self.image_size = int(image_size)
        self.transform = post_decode_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(str(row["image_path"]))
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image, transmitted_bits = encode_decode_resized_image(
            image=image,
            codec=self.codec,
            quality=self.quality,
            image_size=self.image_size,
        )

        target = torch.tensor(int(row["label"]), dtype=torch.long)
        return self.transform(image), target, torch.tensor(int(transmitted_bits), dtype=torch.long)


def make_compression_loader(
    csv_path: Path,
    codec: str,
    quality: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    max_samples: int | None = None,
    source_split: str | None = None,
):
    dataset = CompressedImageDataset(
        csv_path=csv_path,
        codec=codec,
        quality=quality,
        image_size=image_size,
        max_samples=max_samples,
        source_split=source_split,
    )
    loader_kwargs = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


@torch.no_grad()
def predict_compressed(model, loader, device, desc: str):
    model.eval()
    y_true = []
    y_score = []
    bits = []

    for images, labels, transmitted_bits in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        scores = torch.sigmoid(outputs["pad_logit"]).detach().cpu().numpy()

        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        y_score.extend(np.asarray(scores, dtype=float).reshape(-1).tolist())
        bits.extend(transmitted_bits.cpu().numpy().astype(int).tolist())

    return y_true, y_score, bits


def bit_summary(bits):
    bits = np.asarray(bits, dtype=float)
    if bits.size == 0:
        return {
            "avg_bits": float("nan"),
            "min_bits": float("nan"),
            "max_bits": float("nan"),
        }
    return {
        "avg_bits": float(np.mean(bits)),
        "min_bits": float(np.min(bits)),
        "max_bits": float(np.max(bits)),
    }


def prefixed_metrics(prefix: str, metrics: dict):
    keys = ["acc", "precision", "recall", "f1", "auc", "apcer", "bpcer", "acer", "tn", "fp", "fn", "tp"]
    return {f"{prefix}_{key}": metrics.get(key, "") for key in keys}


def prefixed_bits(prefix: str, bits):
    return {f"{prefix}_{key}": value for key, value in bit_summary(bits).items()}


def load_model_for_run(run_name: str, configs_dir: Path, runs_dir: Path, device_override: str | None):
    cfg_path = configs_dir / f"{run_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg = load_yaml(str(cfg_path))
    run_dir = Path(cfg.get("output", {}).get("run_dir", runs_dir / run_name))
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = get_device(device_override if device_override else str(cfg.get("device", "auto")))
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return cfg, run_dir, checkpoint_path, model, device


def evaluate_condition(
    run_name: str,
    cfg: dict,
    model,
    device,
    codec: str,
    quality: int,
    val_csv: Path,
    test_csv: Path,
    external_csv: Path | None,
    external_source_split: str | None,
    dataset_name: str,
    target_bpcer: float,
    batch_size_override: int | None,
    num_workers_override: int | None,
    max_samples: int | None,
):
    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(batch_size_override or data_cfg.get("batch_size", 32))
    num_workers = int(num_workers_override if num_workers_override is not None else data_cfg.get("num_workers", 0))
    persistent_workers = bool(data_cfg.get("persistent_workers", False))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))

    display_quality = "" if codec == "raw" else str(int(quality))
    condition = "raw" if codec == "raw" else f"{codec}_q{quality}"

    val_loader = make_compression_loader(
        val_csv,
        codec=codec,
        quality=quality,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        max_samples=max_samples,
    )
    val_true, val_score, val_bits = predict_compressed(model, val_loader, device, desc=f"{run_name} {condition} val")
    threshold, val_metrics = find_threshold_by_target_bpcer(val_true, val_score, target_bpcer=target_bpcer)

    test_loader = make_compression_loader(
        test_csv,
        codec=codec,
        quality=quality,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        max_samples=max_samples,
    )
    test_true, test_score, test_bits = predict_compressed(model, test_loader, device, desc=f"{run_name} {condition} test")
    test_metrics = compute_pad_metrics(test_true, test_score, threshold=threshold)

    external_metrics = {}
    external_bits = []
    external_count = 0
    if external_csv is not None:
        external_loader = make_compression_loader(
            external_csv,
            codec=codec,
            quality=quality,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            max_samples=max_samples,
            source_split=external_source_split,
        )
        external_true, external_score, external_bits = predict_compressed(
            model,
            external_loader,
            device,
            desc=f"{run_name} {condition} {dataset_name}",
        )
        external_metrics = compute_pad_metrics(external_true, external_score, threshold=threshold)
        external_count = len(external_true)

    raw_bits = raw_image_bits(h=image_size, w=image_size, c=3, bits=8)
    avg_bits = bit_summary(test_bits)["avg_bits"]
    row = {
        "method": f"Raw image + {method_name(run_name)}" if codec == "raw" else f"{codec.upper()} Q={quality} + {method_name(run_name)}",
        "run_name": run_name,
        "codec": codec,
        "quality": display_quality,
        "tx_payload": "raw_image" if codec == "raw" else "compressed_image",
        "avg_bits": avg_bits,
        "avg_kbits": avg_bits / 1000.0,
        "raw_bits_reference": raw_bits,
        "bit_ratio_vs_raw": avg_bits / raw_bits if raw_bits else float("nan"),
        "threshold_policy": f"val_target_bpcer_{target_bpcer:.3f}",
        "threshold": float(threshold),
        "val_count": len(val_true),
        "celeba_test_count": len(test_true),
        "external_dataset": dataset_name,
        "external_source_split": external_source_split or "all",
        "external_count": external_count,
    }
    row.update(prefixed_bits("val", val_bits))
    row.update(prefixed_metrics("val", val_metrics))
    row.update(prefixed_bits("celeba", test_bits))
    row.update(prefixed_metrics("celeba", test_metrics))

    if external_csv is not None:
        row.update(prefixed_bits("lcc", external_bits))
        row.update(prefixed_metrics("lcc", external_metrics))

    return row


def read_policy_row(run_dir: Path, policy: str):
    path = run_dir / "diagnostics" / "threshold_policy_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    match = df[df["policy"] == policy]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def append_aqb_reference(rows, aqb_run: str, configs_dir: Path, runs_dir: Path, policy: str, dataset_name: str):
    cfg_path = configs_dir / f"{aqb_run}.yaml"
    if not cfg_path.exists():
        print(f"[WARN] AQB reference config not found: {cfg_path}")
        return

    cfg = load_yaml(str(cfg_path))
    run_dir = Path(cfg.get("output", {}).get("run_dir", runs_dir / aqb_run))
    policy_row = read_policy_row(run_dir, policy)
    external_path = run_dir / "external_eval" / dataset_name / "metrics.json"

    if policy_row is None:
        print(f"[WARN] AQB reference policy not found: {run_dir / 'diagnostics' / 'threshold_policy_comparison.csv'}")
        return
    if not external_path.exists():
        print(f"[WARN] AQB reference external metrics not found: {external_path}")
        return

    with open(external_path, "r", encoding="utf-8") as f:
        external_metrics = json.load(f)

    model_cfg = cfg.get("model", {})
    bits = latent_bits(int(model_cfg.get("dz", 0)), int(model_cfg.get("bits", 0)))
    row = {
        "method": method_name(aqb_run),
        "run_name": aqb_run,
        "codec": "latent",
        "quality": "",
        "tx_payload": "quantized_latent",
        "avg_bits": bits,
        "avg_kbits": bits / 1000.0,
        "raw_bits_reference": raw_image_bits(),
        "bit_ratio_vs_raw": bits / raw_image_bits(),
        "threshold_policy": policy,
        "threshold": policy_row.get("threshold", ""),
        "val_count": "",
        "celeba_test_count": "",
        "external_dataset": dataset_name,
        "external_source_split": external_metrics.get("dataset", dataset_name),
        "external_count": external_metrics.get("num_samples", ""),
    }

    for prefix in ("val", "test"):
        out_prefix = "celeba" if prefix == "test" else prefix
        metrics = {}
        for key in ["acc", "precision", "recall", "f1", "auc", "apcer", "bpcer", "acer", "tn", "fp", "fn", "tp"]:
            metrics[key] = policy_row.get(f"{prefix}_{key}", "")
        row.update(prefixed_metrics(out_prefix, metrics))
        row[f"{out_prefix}_avg_bits"] = bits
        row[f"{out_prefix}_min_bits"] = bits
        row[f"{out_prefix}_max_bits"] = bits

    row.update(prefixed_metrics("lcc", external_metrics))
    row["lcc_avg_bits"] = bits
    row["lcc_min_bits"] = bits
    row["lcc_max_bits"] = bits
    rows.append(row)


def write_rows_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Evaluate JPEG/WebP communication baselines without retraining.")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--configs-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--val-csv", type=Path, default=ROOT / "data/processed/val.csv")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data/processed/test.csv")
    parser.add_argument("--external-csv", type=Path, default=ROOT / "data/evaluation/evaluation.csv")
    parser.add_argument("--external-source-split", default=None)
    parser.add_argument("--dataset-name", default="lcc_fasd")
    parser.add_argument("--codecs", nargs="*", default=DEFAULT_CODECS, choices=["jpeg", "webp"])
    parser.add_argument("--qualities", nargs="*", type=int, default=DEFAULT_QUALITIES)
    parser.add_argument("--target-bpcer", type=float, default=0.02)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--aqb-reference-run", default="aqb_z64_b8")
    parser.add_argument("--no-aqb-reference", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/results/compression_baselines.csv")
    parser.add_argument("--json-out", type=Path, default=ROOT / "outputs/results/compression_baselines.json")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if "webp" in args.codecs and not features.check("webp"):
        raise RuntimeError("This Pillow build does not support WebP encoding.")

    validate_csv(args.val_csv)
    validate_csv(args.test_csv)
    external_csv = args.external_csv if args.external_csv and args.external_csv.exists() else None
    if external_csv is not None:
        validate_csv(external_csv)
    else:
        print(f"[WARN] External CSV not found; LCC columns will be omitted: {args.external_csv}")

    rows = []
    conditions = []
    if args.include_raw:
        conditions.append(("raw", 100))
    for codec in args.codecs:
        for quality in args.qualities:
            conditions.append((codec, int(quality)))

    for run_name in args.runs:
        cfg, run_dir, checkpoint_path, model, device = load_model_for_run(
            run_name=run_name,
            configs_dir=args.configs_dir,
            runs_dir=args.runs_dir,
            device_override=args.device,
        )
        print(f"[RUN] {run_name} checkpoint={checkpoint_path} device={device}")

        run_rows = []
        for codec, quality in conditions:
            print(f"[EVAL] {run_name} codec={codec} quality={quality if codec != 'raw' else '-'}")
            row = evaluate_condition(
                run_name=run_name,
                cfg=cfg,
                model=model,
                device=device,
                codec=codec,
                quality=quality,
                val_csv=args.val_csv,
                test_csv=args.test_csv,
                external_csv=external_csv,
                external_source_split=args.external_source_split,
                dataset_name=args.dataset_name,
                target_bpcer=float(args.target_bpcer),
                batch_size_override=args.batch_size,
                num_workers_override=args.num_workers,
                max_samples=args.max_samples,
            )
            rows.append(row)
            run_rows.append(row)
            print(
                f"  threshold={row['threshold']:.8f} avg_bits={row['avg_bits']:.1f} "
                f"celeba_acer={row['celeba_acer']:.4f} lcc_acer={row.get('lcc_acer', float('nan')):.4f}"
            )

        run_out_name = "metrics.json"
        if args.max_samples is not None:
            run_out_name = f"metrics_smoke{args.max_samples}.json"
        run_out = run_dir / "compression_baselines" / run_out_name
        save_json(run_rows, str(run_out))
        print(f"[WRITE] {run_out}")

    if not args.no_aqb_reference and args.aqb_reference_run:
        append_aqb_reference(
            rows,
            aqb_run=args.aqb_reference_run,
            configs_dir=args.configs_dir,
            runs_dir=args.runs_dir,
            policy=f"val_target_bpcer_{args.target_bpcer:.3f}",
            dataset_name=args.dataset_name,
        )

    out_path = args.out
    json_out = args.json_out
    if args.max_samples is not None:
        out_path = out_path.with_name(f"{out_path.stem}_smoke{args.max_samples}{out_path.suffix}")
        json_out = json_out.with_name(f"{json_out.stem}_smoke{args.max_samples}{json_out.suffix}")

    write_rows_csv(out_path, rows)
    save_json(rows, str(json_out))
    print(f"[WRITE] {out_path} rows={len(rows)}")
    print(f"[WRITE] {json_out}")

    preview_columns = [
        "method",
        "tx_payload",
        "avg_bits",
        "celeba_acer",
        "lcc_acer",
        "threshold",
    ]
    print(pd.DataFrame(rows)[preview_columns].to_string(index=False))


if __name__ == "__main__":
    main()
