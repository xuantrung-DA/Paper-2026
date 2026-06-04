import os
from torch.utils.data import Dataset

class CelebASpoofDataset(Dataset):
    """Placeholder dataset for CelebA-Spoof processed CSVs."""
    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.transform = transform
        # load paths/labels lazily or implement CSV parsing

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError
