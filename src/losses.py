import torch.nn as nn

class FASLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets)
