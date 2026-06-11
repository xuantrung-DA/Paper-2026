from pathlib import Path
import argparse
import csv
import json
import sys
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import latent_bits, raw_image_bits
from src.utils import load_yaml


DEFAULT_RUNS = [
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


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def method_name(run_name: str, cfg: Dict) -> str:
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
    return names.get(run_name, cfg.get("model", {}).get("name", run_name))


def communication_row(config_path: Path) -> Dict:
    cfg = load_yaml(str(config_path))
    run_name = cfg.get("run_name", config_path.stem)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    image_size = int(data_cfg.get("image_size", 224))
    image_bits = raw_image_bits(h=image_size, w=image_size, c=3, bits=8)

    model_name = model_cfg.get("name", "")
    dz = model_cfg.get("dz", "")
    bits = model_cfg.get("bits", "")

    if model_name == "aqb_fas" and dz != "" and bits != "":
        transmitted_bits = latent_bits(int(dz), int(bits))
        payload_type = "quantized_latent"
    else:
        transmitted_bits = image_bits
        payload_type = "raw_image"

    compression_ratio = image_bits / max(float(transmitted_bits), 1.0)
    reduction_percent = 100.0 * (1.0 - float(transmitted_bits) / float(image_bits))

    return {
        "run_name": run_name,
        "method": method_name(run_name, cfg),
        "config": display_path(config_path),
        "model": model_name,
        "payload_type": payload_type,
        "image_size": image_size,
        "image_channels": 3,
        "image_bits_per_channel": 8,
        "raw_image_bits": int(image_bits),
        "raw_image_bytes": image_bits / 8.0,
        "dz": dz,
        "bits": bits,
        "transmitted_bits": int(transmitted_bits),
        "transmitted_bytes": transmitted_bits / 8.0,
        "transmitted_kib": transmitted_bits / 8.0 / 1024.0,
        "compression_ratio": compression_ratio,
        "reduction_percent": reduction_percent,
    }


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
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
        writer.writerows(rows)


def write_json(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(rows), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure communication payload size for paper models.")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--configs-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/results/communication_budget.csv")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rows: List[Dict] = []
    for run_name in args.runs:
        config_path = args.configs_dir / f"{run_name}.yaml"
        if not config_path.exists():
            print(f"[SKIP] Missing config: {config_path}")
            continue

        row = communication_row(config_path)
        rows.append(row)
        print(
            f"[COMM] {row['run_name']}: payload={row['transmitted_bits']} bits "
            f"ratio={row['compression_ratio']:.1f}x "
            f"reduction={row['reduction_percent']:.4f}%"
        )

    write_csv(args.out, rows)
    print(f"[WRITE] {args.out} rows={len(rows)}")

    json_out = args.json_out or args.out.with_suffix(".json")
    write_json(json_out, rows)
    print(f"[WRITE] {json_out}")


if __name__ == "__main__":
    main()
