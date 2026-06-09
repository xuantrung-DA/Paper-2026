from pathlib import Path
import argparse
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets.celeba_spoof import CelebASpoofDataset
from src.eval import evaluate_loader
from src.losses import compute_loss
from src.models.factory import build_model
from src.utils import ensure_dir, get_device, load_yaml, save_json, set_seed


def make_transforms(image_size: int, train: bool):
    ops = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(ops)


def make_loader(
    csv_path: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    dataset = CelebASpoofDataset(
        csv_path,
        transform=make_transforms(image_size=image_size, train=train),
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def move_targets(targets, device):
    return {k: v.to(device, non_blocking=True) for k, v in targets.items()}


def write_history_csv(rows, path: Path):
    if not rows:
        return

    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_history_csv(path: Path):
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def truncate_history(rows, epoch: int):
    kept = []
    for row in rows:
        try:
            row_epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if row_epoch <= epoch:
            kept.append(row)
    return kept


def save_checkpoint(path: Path, model, optimizer, cfg, epoch: int, metrics: dict):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg,
            "metrics": metrics,
        },
        path,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_cfg,
    model_cfg,
    amp_enabled: bool,
    channels_last: bool,
):
    model.train()
    total_loss = 0.0
    steps = 0
    device_type = device.type

    progress = tqdm(loader, desc="train", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        targets = move_targets(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            if model_cfg.get("name") == "aqb_fas":
                outputs = model(
                    images,
                    noise_std=float(model_cfg.get("noise_std", 0.0)),
                    drop_prob=float(model_cfg.get("drop_prob", 0.0)),
                )
            else:
                outputs = model(images)
            loss, logs = compute_loss(outputs, targets, loss_cfg)

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        total_loss += loss_value
        steps += 1
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return total_loss / max(steps, 1), logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", choices=["last", "best"], default="last")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    output_cfg = cfg["output"]

    device = get_device(str(cfg.get("device", "auto")))
    print(f"[DEVICE] {device}")

    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    persistent_workers = bool(data_cfg.get("persistent_workers", False))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))

    train_loader = make_loader(
        data_cfg["train_csv"],
        image_size,
        batch_size,
        num_workers,
        train=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = make_loader(
        data_cfg["val_csv"],
        image_size,
        batch_size,
        num_workers,
        train=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model = build_model(cfg).to(device)
    channels_last = bool(train_cfg.get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    run_dir = ensure_dir(output_cfg["run_dir"])
    amp_enabled = bool(train_cfg.get("amp", False)) and device.type in {"cuda", "xpu"}
    save_json(cfg, str(run_dir / "config.json"))

    best_acer = float("inf")
    best_metrics = {}
    best_epoch = -1
    patience = int(train_cfg.get("early_stop_patience", 0))
    stale_epochs = 0
    history = []
    start_epoch = 0

    if args.resume:
        resume_path = run_dir / f"{args.resume_from}.pt"
        if not resume_path.exists() and args.resume_from == "last":
            fallback_path = run_dir / "best.pt"
            if fallback_path.exists():
                print(f"[RESUME] {resume_path} not found; falling back to {fallback_path}")
                resume_path = fallback_path

        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])

            resumed_epoch = int(checkpoint.get("epoch", -1))
            start_epoch = resumed_epoch + 1
            history = truncate_history(read_history_csv(run_dir / "history.csv"), resumed_epoch)
            write_history_csv(history, run_dir / "history.csv")
            print(f"[RESUME] loaded {resume_path} at epoch={resumed_epoch}")
        else:
            print("[RESUME] no checkpoint found; starting from scratch")

        best_metrics_path = run_dir / "best_metrics.json"
        if best_metrics_path.exists():
            import json

            with open(best_metrics_path, "r", encoding="utf-8") as f:
                best_metrics = json.load(f)
            best_epoch = int(best_metrics.get("epoch", -1))
            best_acer = float(best_metrics.get("val_metrics", {}).get("acer", best_acer))

    epochs = int(train_cfg.get("epochs", 1))
    if start_epoch >= epochs:
        print(f"[DONE] checkpoint already reached epochs={epochs}")
        return

    for epoch in range(start_epoch, epochs):
        print(f"[EPOCH {epoch + 1}/{epochs}]")
        train_loss, train_logs = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=cfg.get("loss", {}),
            model_cfg=cfg.get("model", {}),
            amp_enabled=amp_enabled,
            channels_last=channels_last,
        )
        val_metrics = evaluate_loader(model, val_loader, device, choose_threshold=True)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_logs": train_logs,
            "val_metrics": val_metrics,
        }
        save_json(metrics, str(run_dir / "last_metrics.json"))

        history_row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
        }
        for key, value in train_logs.items():
            history_row[f"train_{key}"] = value
        for key, value in val_metrics.items():
            history_row[f"val_{key}"] = value
        history.append(history_row)
        write_history_csv(history, run_dir / "history.csv")

        val_acer = float(val_metrics["acer"])
        print(
            f"[VAL] loss={train_loss:.4f} "
            f"acer={val_acer:.4f} auc={val_metrics['auc']:.4f} "
            f"threshold={val_metrics['threshold']:.4f}"
        )

        if val_acer < best_acer:
            best_acer = val_acer
            best_metrics = metrics
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg,
                    "metrics": metrics,
                },
                run_dir / "best.pt",
            )
            save_json(best_metrics, str(run_dir / "best_metrics.json"))
            print(f"[SAVE] best checkpoint -> {run_dir / 'best.pt'}")
        else:
            stale_epochs += 1

        save_checkpoint(run_dir / "last.pt", model, optimizer, cfg, epoch, metrics)

        if patience > 0 and stale_epochs >= patience:
            print(f"[EARLY STOP] no improvement for {patience} epochs")
            break

    print(f"[DONE] best_epoch={best_epoch} best_val_acer={best_acer:.4f}")


if __name__ == "__main__":
    main()
