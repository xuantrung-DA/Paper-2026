import torch
import torch.nn as nn

class AQBFAS(nn.Module):
    def __init__(self, z=64, bitrate=8):
        super().__init__()
        self.z = z
        self.bitrate = bitrate
        self.encoder = nn.Identity()
        self.classifier = nn.Linear(1,1)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)
