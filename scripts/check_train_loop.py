from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Dataset, DataLoader

from src.models.factory import build_model
from src.losses import compute_loss
from src.eval import evaluate_loader
from src.utils import set_seed, get_device


class FakeFASDataset(Dataset):
    """
    Fake dataset để test train loop.
    Không dùng để báo kết quả.
    """

    def __init__(self, n: int = 64, image_size: int = 224):
        self.n = n
        self.image_size = image_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)

        label = idx % 2

        if label == 0:
            spoof_type = 0
            illumination = 0
            environment = 0
        else:
            spoof_type = (idx % 10) + 1
            illumination = (idx % 4) + 1
            environment = (idx % 2) + 1

        target = {
            "label": torch.tensor(label, dtype=torch.long),
            "spoof_type": torch.tensor(spoof_type, dtype=torch.long),
            "illumination": torch.tensor(illumination, dtype=torch.long),
            "environment": torch.tensor(environment, dtype=torch.long),
        }

        return image, target


def make_cfg(model_name: str):
    if model_name == "baseline_mbv3":
        return {
            "model": {
                "name": "baseline_mbv3",
                "pretrained": False,
                "dropout": 0.2,
            },
            "loss": {
                "lambda_spoof": 0.0,
                "lambda_illum": 0.0,
                "lambda_env": 0.0,
                "gamma_proto": 0.0,
            },
        }

    if model_name == "aqb_fas":
        return {
            "model": {
                "name": "aqb_fas",
                "pretrained": False,
                "dz": 64,
                "bits": 8,
                "dropout": 0.2,
            },
            "loss": {
                "lambda_spoof": 0.3,
                "lambda_illum": 0.1,
                "lambda_env": 0.1,
                "gamma_proto": 0.05,
            },
        }

    raise ValueError(model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline_mbv3",
                        choices=["baseline_mbv3", "aqb_fas"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    set_seed(42)
    device = get_device(args.device)

    cfg = make_cfg(args.model)

    train_ds = FakeFASDataset(n=64)
    val_ds = FakeFASDataset(n=32)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print("=" * 80)
    print(f"[CHECK TRAIN LOOP] model={args.model}, device={device}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss, logs = compute_loss(outputs, targets, cfg["loss"])

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())

            if step == 0:
                print("[STEP 0 LOGS]", logs)

        avg_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate_loader(model, val_loader, device, choose_threshold=True)

        print(f"[EPOCH {epoch}] loss={avg_loss:.4f}, val_acer={val_metrics['acer']:.4f}")
        print("[VAL METRICS]", val_metrics)

    print("[DONE] Train loop smoke test passed.")


if __name__ == "__main__":
    main()