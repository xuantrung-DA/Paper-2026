import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define a simple backbone stub
        self.backbone = nn.Identity()
        self.head = nn.Linear(1,1)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
