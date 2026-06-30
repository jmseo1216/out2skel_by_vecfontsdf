"""간단한 CNN encoder + MLP segment decoder."""
from __future__ import annotations

import torch
from torch import nn


class FontSkeletonModel(nn.Module):
    """[B,1,128,128] outline 이미지를 K개 선분으로 디코딩한다."""

    def __init__(self, num_segments: int = 48, image_size: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_segments * 5),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        out = self.decoder(feat).view(x.shape[0], self.num_segments, 5)
        pred_exist_logits = out[..., 0]
        # sigmoid로 [0,1] 정규화 후 image_size 픽셀 좌표계 [0,128]로 변환한다.
        pred_segments = torch.sigmoid(out[..., 1:]) * float(self.image_size)
        return pred_exist_logits, pred_segments
