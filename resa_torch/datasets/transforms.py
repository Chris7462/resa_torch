import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    resize_shape: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    rotation: float = 2.0
) -> A.Compose:
    """
    Build transforms for training.

    Args:
        resize_shape: Target size as (height, width)
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel
        rotation: Rotation angle in degrees (samples from [-rotation, rotation])

    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(
            height=resize_shape[0],
            width=resize_shape[1],
            interpolation=cv2.INTER_CUBIC
        ),
        A.Rotate(limit=rotation, border_mode=cv2.BORDER_CONSTANT, p=0.5,
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    resize_shape: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> A.Compose:
    """
    Build transforms for validation and testing.

    Args:
        resize_shape: Target size as (height, width)
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel

    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(
            height=resize_shape[0],
            width=resize_shape[1],
            interpolation=cv2.INTER_CUBIC
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
