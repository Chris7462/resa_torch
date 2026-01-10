import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ExistHead(nn.Module):
    """
    Lane existence prediction head.

    Predicts whether each lane exists in the image.

    Architecture:
        Dropout → Conv(in_channels→num_classes, 1x1) → Softmax →
        AdaptiveAvgPool(pool_size) → Flatten →
        FC(num_classes*pool_h*pool_w → 128) → ReLU → FC(128 → num_lanes)

    Args:
        in_channels: Number of input channels from aggregator
        pool_size: Output size for adaptive pooling (H, W)
                   CULane: (18, 50), TuSimple: (23, 40)
        num_classes: Number of segmentation classes (including background)

    Output:
        Existence logits of shape (B, num_lanes) where num_lanes = num_classes - 1
        Use BCEWithLogitsLoss for training.
    """

    def __init__(
        self,
        in_channels: int = 128,
        pool_size: tuple[int, int] = (18, 50),
        num_classes: int = 5
    ) -> None:
        super().__init__()

        num_lanes = num_classes - 1

        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        self.pool = nn.Sequential(
            nn.Softmax(dim=1),
            nn.AdaptiveAvgPool2d(pool_size),
        )

        fc_input_features = num_classes * pool_size[0] * pool_size[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_lanes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, H, W) from aggregator

        Returns:
            Existence logits of shape (B, num_lanes)
        """
        x = self.dropout(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x)

        return x
