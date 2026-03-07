"""
ElbowVision ConvNeXt-Small 角度回帰モデル定義

OsteoVision の ResNet を ConvNeXt-Small に置き換えた肘用モデル。
出力: [carrying_angle, flexion, pronation_sup, varus_valgus]
  - AP像では carrying_angle のみ有効
  - LAT像では flexion のみ有効

このファイルは API(main.py) と 訓練スクリプト(train_angle_predictor.py) で共有する。
"""

import torch
import torch.nn as nn
from torchvision import models

OUTPUT_DIM = 4  # [carrying_angle, flexion, pronation_sup, varus_valgus]


class ElbowConvNeXt(nn.Module):
    """
    ConvNeXt-Small バックボーンによる肘関節角度回帰モデル。
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
    print(f"出力shape: {out.shape}")   # → (2, 4)
    print(f"出力例:    {out[0].detach().numpy()}")
