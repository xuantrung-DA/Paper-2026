import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class BaselineMobileNetV3(nn.Module):
    """
    Baseline PAD model:
      image -> MobileNetV3-Small -> PAD logit

    Quy ước:
      label = 0: live
      label = 1: spoof
      pad_logit > 0 nghĩa là model nghiêng về spoof.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        mb = mobilenet_v3_small(weights=weights)

        self.backbone = mb.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 576

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        h = self.pool(h).flatten(1)
        pad_logit = self.classifier(h).squeeze(1)

        return {
            "pad_logit": pad_logit,
            "feat": h,
        }