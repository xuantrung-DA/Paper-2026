from pathlib import Path
import argparse
import csv
import json
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.metrics import latent_bits as compute_latent_bits
from src.metrics import raw_image_bits
from src.models.factory import build_model
from src.utils import get_device, load_yaml


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


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def checkpoint_size_mb(path: Optional[Path]) -> Optional[float]:
    if path is None or not path.exists():
        return None
    return path.stat().st_size / (1024.0 * 1024.0)


def display_path(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_latent_bits(cfg: Dict) -> Optional[int]:
    model_cfg = cfg.get("model", {})
    if model_cfg.get("name") != "aqb_fas":
        return None

    dz = model_cfg.get("dz")
    bits = model_cfg.get("bits")
    if dz is None or bits is None:
        return None

    return compute_latent_bits(int(dz), int(bits))


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[Path], device: torch.device) -> None:
    if checkpoint_path is None:
        return
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])


def build_benchmark_model(
    cfg: Dict,
    checkpoint_path: Optional[Path],
    device: torch.device,
):
    model = build_model(cfg).to(device)

    channels_last = bool(cfg.get("train", {}).get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model, channels_last


def make_input(
    batch_size: int,
    image_size: int,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def timed_forward(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    device_type = device.type
    sync_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            outputs = model(images)
            _ = outputs["pad_logit"]
    sync_device(device)
    return time.perf_counter() - start


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return float(ordered[index])


def benchmark_model(
    model: torch.nn.Module,
    cfg: Dict,
    device: torch.device,
    batch_size: int = 1,
    warmup: int = 20,
    repeats: int = 100,
    amp: Optional[bool] = None,
) -> Dict[str, float]:
    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    channels_last = bool(cfg.get("train", {}).get("channels_last", False))
    amp_enabled = bool(cfg.get("train", {}).get("amp", False)) if amp is None else bool(amp)
    amp_enabled = amp_enabled and device.type in {"cuda", "xpu"}

    images = make_input(batch_size, image_size, device, channels_last)

    for _ in range(max(warmup, 0)):
        timed_forward(model, images, device, amp_enabled)

    times_ms = [
        timed_forward(model, images, device, amp_enabled) * 1000.0
        for _ in range(max(repeats, 1))
    ]

    mean_ms = float(statistics.mean(times_ms))
    median_ms = float(statistics.median(times_ms))
    std_ms = float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0
    p95_ms = percentile(times_ms, 0.95)
    per_image_ms = mean_ms / max(batch_size, 1)
    throughput_fps = 1000.0 / per_image_ms if per_image_ms > 0 else float("nan")

    return {
        "batch_size": int(batch_size),
        "image_size": int(image_size),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "amp": bool(amp_enabled),
        "latency_ms_mean": mean_ms,
        "latency_ms_median": median_ms,
        "latency_ms_p95": p95_ms,
        "latency_ms_std": std_ms,
        "latency_ms_per_image": per_image_ms,
        "throughput_fps": throughput_fps,
    }


def row_for_config(
    config_path: Path,
    checkpoint_path: Optional[Path],
    device_name: str,
    batch_size: int,
    warmup: int,
    repeats: int,
    amp: Optional[bool],
) -> Dict:
    cfg = load_yaml(str(config_path))
    device = get_device(device_name if device_name else str(cfg.get("device", "auto")))
    model, _ = build_benchmark_model(cfg, checkpoint_path, device)
    bench = benchmark_model(
        model=model,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        warmup=warmup,
        repeats=repeats,
        amp=amp,
    )

    run_name = cfg.get("run_name", config_path.stem)
    model_cfg = cfg.get("model", {})
    latent = model_latent_bits(cfg)
    raw_bits = raw_image_bits(
        h=int(cfg.get("data", {}).get("image_size", 224)),
        w=int(cfg.get("data", {}).get("image_size", 224)),
        c=3,
        bits=8,
    )

    row = {
        "run_name": run_name,
        "config": display_path(config_path),
        "checkpoint": display_path(checkpoint_path),
        "device": str(device),
        "model": model_cfg.get("name", ""),
        "dz": model_cfg.get("dz", ""),
        "bits": model_cfg.get("bits", ""),
        "latent_bits": latent if latent is not None else "",
        "raw_image_bits": int(raw_bits),
        "compression_ratio": (float(raw_bits) / float(latent)) if latent else "",
        "params": int(count_parameters(model)),
        "checkpoint_mb": checkpoint_size_mb(checkpoint_path),
    }
    row.update(bench)
    return row


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


def config_paths_from_runs(runs: List[str], configs_dir: Path) -> List[Path]:
    return [configs_dir / f"{run}.yaml" for run in runs]


def checkpoint_for_config(config_path: Path, runs_dir: Path) -> Path:
    cfg = load_yaml(str(config_path))
    run_dir = Path(cfg.get("output", {}).get("run_dir", runs_dir / config_path.stem))
    return run_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure inference latency for trained FAS models.")
    parser.add_argument("--configs", nargs="*", type=Path, default=None)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--configs-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Only for single-config runs.")
    parser.add_argument("--device", default="", help="Override config device, e.g. xpu, cuda, cpu, auto.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--amp", choices=["config", "on", "off"], default="config")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/results/latency.csv")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.configs:
        config_paths = args.configs
    else:
        runs = args.runs if args.runs else DEFAULT_RUNS
        config_paths = config_paths_from_runs(runs, args.configs_dir)

    if args.checkpoint is not None and len(config_paths) != 1:
        raise ValueError("--checkpoint can only be used with exactly one config")

    amp = None
    if args.amp == "on":
        amp = True
    elif args.amp == "off":
        amp = False

    rows = []
    for config_path in config_paths:
        checkpoint_path = args.checkpoint or checkpoint_for_config(config_path, args.runs_dir)
        if not config_path.exists():
            print(f"[SKIP] Missing config: {config_path}")
            continue
        if not checkpoint_path.exists():
            print(f"[SKIP] Missing checkpoint: {checkpoint_path}")
            continue

        print(f"[LATENCY] {config_path.stem}")
        row = row_for_config(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            amp=amp,
        )
        rows.append(row)
        print(
            f"  device={row['device']} batch={row['batch_size']} "
            f"mean={row['latency_ms_mean']:.3f}ms "
            f"p95={row['latency_ms_p95']:.3f}ms "
            f"fps={row['throughput_fps']:.2f}"
        )

    write_csv(args.out, rows)
    print(f"[WRITE] {args.out} rows={len(rows)}")

    json_out = args.json_out or args.out.with_suffix(".json")
    write_json(json_out, rows)
    print(f"[WRITE] {json_out}")


if __name__ == "__main__":
    main()
