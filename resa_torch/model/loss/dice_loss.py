import torch
from torch import Tensor


def dice_loss(input: Tensor, target: Tensor) -> Tensor:
    """
    Compute Dice loss.

    Args:
        input: Predictions of shape (B, C, H, W), after softmax
        target: One-hot encoded targets of shape (B, C, H, W)

    Returns:
        Scalar dice loss
    """
    input = input.contiguous().view(input.size(0), -1)
    target = target.contiguous().view(target.size(0), -1).float()

    intersection = torch.sum(input * target, dim=1)
    input_sum = torch.sum(input * input, dim=1) + 0.001
    target_sum = torch.sum(target * target, dim=1) + 0.001

    dice = (2 * intersection) / (input_sum + target_sum)

    return (1 - dice).mean()
