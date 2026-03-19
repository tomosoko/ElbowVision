"""
ElbowVision ConvNeXt-Small ポジショニングズレ回帰モデル定義

出力: [rotation_error_deg, flexion_deg]
  index 0 — rotation_error_deg : 理想位からの前腕回旋ズレ量（°）
             AP像では回旋ズレ推定、LAT像では不使用（マスク=0）
  index 1 — flexion_deg        : 肘屈曲角（°）
             LAT像では屈曲角推定、AP像では不使用（マスク=0）

訓練・推論で同じ定義を使うため、このファイルを唯一のソース（Single Source of Truth）とする。
このファイルは API(main.py) と 訓練スクリプト(train_angle_predictor.py) で共有する。
"""

import torch
import torch.nn as nn
from torchvision import models

OUTPUT_DIM = 2  # [rotation_error_deg, flexion_deg]
OUTPUT_NAMES = ["rotation_error_deg", "flexion_deg"]


class ElbowConvNeXt(nn.Module):
    """
    ConvNeXt-Small バックボーンによるポジショニングズレ回帰モデル。
    ImageNet事前学習重みを使ったファインチューニングを想定。
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.convnext_small(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    model = ElbowConvNeXt(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"出力shape: {out.shape}")   # → (2, 2)
    print(f"出力例:    {out[0].detach().numpy()}")
