import torch.nn as nn
from torch import Tensor

from ..backbone import ResNetBackbone
from ..aggregator import RESAAggregator
from ..decoder import DECODERS
from ..head import ExistHead


class RESA(nn.Module):
    """
    RESA network for lane detection.

    Architecture:
        Input (B, 3, H, W)
            │
            ▼
        ResNetBackbone ────────── (B, 128, H/8, W/8)
            │
            ▼
        RESAAggregator ────────── (B, 128, H/8, W/8)
            │
            ├──────────────────────────────────┐
            ▼                                  ▼
        Decoder                           ExistHead
            │                                  │
            ▼                                  ▼
        seg_pred (B, C, H, W)            exist_pred (B, num_lanes)

    Args:
        backbone: ResNet architecture name
        pretrained: Whether to load pretrained backbone weights
        num_classes: Number of segmentation classes (including background)
        aggregator_channels: Number of channels in aggregator
        aggregator_iters: Number of shift iterations in aggregator
        aggregator_kernel_size: Kernel size for aggregator convolutions
        aggregator_alpha: Scaling factor for aggregated features
        decoder_type: Decoder type ('PlainDecoder' or 'BUSDDecoder')

    Reference:
        "RESA: Recurrent Feature-Shift Aggregator for Lane Detection"
        https://arxiv.org/abs/2008.13719
    """

    def __init__(
        self,
        backbone: str = 'resnet34',
        pretrained: bool = True,
        num_classes: int = 5,
        aggregator_channels: int = 128,
        aggregator_iters: int = 4,
        aggregator_kernel_size: int = 9,
        aggregator_alpha: float = 2.0,
        decoder_type: str = 'PlainDecoder',
    ) -> None:
        super().__init__()

        self.backbone = ResNetBackbone(
            arch=backbone,
            pretrained=pretrained,
            out_channels=aggregator_channels,
        )

        self.aggregator = RESAAggregator(
            channels=aggregator_channels,
            num_iters=aggregator_iters,
            kernel_size=aggregator_kernel_size,
            alpha=aggregator_alpha,
        )

        decoder_cls = DECODERS.get(decoder_type)
        if decoder_cls is None:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        self.decoder = decoder_cls(
            in_channels=aggregator_channels,
            num_classes=num_classes,
        )

        self.exist_head = ExistHead(
            in_channels=aggregator_channels,
            num_classes=num_classes,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
               H and W must be divisible by 8.

        Returns:
            seg_pred: Segmentation logits of shape (B, num_classes, H, W)
            exist_pred: Existence logits of shape (B, num_lanes)
        """
        input_size = x.shape[2:4]  # (H, W)

        x = self.backbone(x)
        x = self.aggregator(x)

        seg_pred = self.decoder(x, output_size=input_size)
        exist_pred = self.exist_head(x)

        return seg_pred, exist_pred
