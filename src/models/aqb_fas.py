import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class UniformQuantizerSTE(nn.Module):
    """
    Uniform b-bit quantization với Straight-Through Estimator.

    Input z được tanh về [-1, 1], sau đó lượng tử hóa b-bit.
    """

    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = torch.tanh(z)

        if self.bits is None or self.bits <= 0:
            return z_norm

        levels = 2 ** self.bits - 1

        q_int = torch.round((z_norm + 1.0) * 0.5 * levels)
        z_deq = 2.0 * q_int / levels - 1.0

        # Straight-through estimator
        z_q = z_norm + (z_deq - z_norm).detach()

        return z_q


class AQBFAS(nn.Module):
    """
    AQB-FAS:
      image
      -> MobileNetV3-Small encoder
      -> compact latent z
      -> b-bit quantizer
      -> receiver MLP
      -> PAD head + semantic attribute heads

    Output chính:
      pad_logit: binary live/spoof
      spoof_logits: spoof type classification
      illum_logits: illumination classification
      env_logits: environment classification
      proto_logits: binary prototype logits
    """

    def __init__(
        self,
        dz: int = 64, # sau này có thể đổi thành 16/32/64/128
        bits: int = 8, # sau này có thể đổi thành 4/8/16 hoặc None (không quantization)
        dropout: float = 0.2,
        pretrained: bool = True,
        num_spoof_types: int = 11,
        num_illum: int = 5,
        num_env: int = 3,
    ):
        super().__init__()

        self.dz = dz
        self.bits = bits

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        mb = mobilenet_v3_small(weights=weights)

        self.backbone = mb.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 576

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, dz),
        )

        self.quantizer = UniformQuantizerSTE(bits=bits)

        self.receiver = nn.Sequential(
            nn.Linear(dz, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.pad_head = nn.Linear(128, 1)
        self.spoof_head = nn.Linear(128, num_spoof_types)
        self.illum_head = nn.Linear(128, num_illum)
        self.env_head = nn.Linear(128, num_env)

        # 2 prototypes: live / spoof
        self.proto = nn.Parameter(torch.randn(2, dz) * 0.02)

    def forward(
        self,
        x: torch.Tensor,
        noise_std: float = 0.0,
        drop_prob: float = 0.0,
    ):
        h = self.backbone(x)
        h = self.pool(h).flatten(1)

        z = self.projector(h)
        zq = self.quantizer(z)

        if noise_std > 0:
            zq = zq + torch.randn_like(zq) * noise_std

        if drop_prob > 0:
            mask = (torch.rand_like(zq) > drop_prob).float()
            zq = zq * mask

        r = self.receiver(zq)

        pad_logit = self.pad_head(r).squeeze(1)

        # Prototype logits: càng gần prototype thì logit càng lớn
        proto_logits = -torch.cdist(torch.tanh(z), self.proto, p=2) ** 2

        return {
            "pad_logit": pad_logit,
            "spoof_logits": self.spoof_head(r),
            "illum_logits": self.illum_head(r),
            "env_logits": self.env_head(r),
            "proto_logits": proto_logits,
            "z": z,
            "zq": zq,
        }

    def latent_bits(self) -> int:
        if self.bits is None or self.bits <= 0:
            return self.dz * 32
        return self.dz * self.bits