import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dice_loss import dice_loss


class RESALoss(nn.Module):
    """
    Combined loss for RESA.

    Combines:
        1. Segmentation loss (CrossEntropy or Dice)
        2. Lane existence loss (BCEWithLogits)

    Total loss = seg_loss * seg_weight + exist_loss * exist_weight

    Args:
        num_classes: Number of segmentation classes (including background)
        loss_type: Segmentation loss type ('cross_entropy' or 'dice')
        seg_weight: Weight for segmentation loss
        exist_weight: Weight for existence loss
        background_weight: Weight for background class in cross entropy
        ignore_label: Label to ignore in segmentation loss
    """

    def __init__(
        self,
        num_classes: int = 5,
        loss_type: str = 'cross_entropy',
        seg_weight: float = 1.0,
        exist_weight: float = 0.1,
        background_weight: float = 0.4,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.loss_type = loss_type
        self.seg_weight = seg_weight
        self.exist_weight = exist_weight

        if loss_type == 'cross_entropy':
            weights = torch.ones(num_classes)
            weights[0] = background_weight
            self.seg_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
        elif loss_type == 'dice':
            self.seg_loss = None  # Computed in forward
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'cross_entropy' or 'dice'")

        self.exist_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        seg_pred: Tensor,
        exist_pred: Tensor,
        seg_gt: Tensor,
        exist_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            seg_pred: Segmentation logits of shape (B, num_classes, H, W)
            exist_pred: Existence logits of shape (B, num_lanes)
            seg_gt: Segmentation ground truth of shape (B, H, W)
            exist_gt: Existence ground truth of shape (B, num_lanes)

        Returns:
            loss: Total combined loss
            loss_seg: Segmentation loss (for logging)
            loss_exist: Existence loss (for logging)
        """
        if self.loss_type == 'cross_entropy':
            loss_seg = self.seg_loss(seg_pred, seg_gt)
        else:  # dice
            target = F.one_hot(seg_gt, num_classes=self.num_classes).permute(0, 3, 1, 2)
            pred = F.softmax(seg_pred, dim=1)
            loss_seg = dice_loss(pred[:, 1:], target[:, 1:])  # Exclude background

        loss_exist = self.exist_loss(exist_pred, exist_gt)

        loss = loss_seg * self.seg_weight + loss_exist * self.exist_weight

        return loss, loss_seg, loss_exist
