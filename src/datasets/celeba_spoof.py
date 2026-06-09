from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CelebASpoofDataset(Dataset):
    """CelebA-Spoof dataset backed by a processed CSV index.

    Expected CSV columns:
      image_path,label,spoof_type,illumination,environment
    """

    def __init__(self, csv_file, transform=None, root_dir: Optional[str] = None):
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else None
        self.df = pd.read_csv(self.csv_file)

        required = {"image_path", "label", "spoof_type", "illumination", "environment"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {self.csv_file}: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        if self.root_dir is not None:
            return self.root_dir / path
        return path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self._resolve_path(str(row["image_path"]))

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = {
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "spoof_type": torch.tensor(int(row["spoof_type"]), dtype=torch.long),
            "illumination": torch.tensor(int(row["illumination"]), dtype=torch.long),
            "environment": torch.tensor(int(row["environment"]), dtype=torch.long),
        }

        return image, target
