import cv2
import numpy as np


# Lane colors: distinct colors for each lane (RGB format)
LANE_COLORS = np.array([
    [255, 125, 0],    # Lane 1: Orange
    [0, 255, 0],      # Lane 2: Green
    [255, 0, 0],      # Lane 3: Red
    [255, 255, 0],    # Lane 4: Yellow
    [0, 255, 255],    # Lane 5: Cyan
    [255, 0, 255],    # Lane 6: Magenta
], dtype=np.uint8)


def visualize_lanes(
    img: np.ndarray,
    seg_pred: np.ndarray,
    exist_pred: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Visualize lane predictions on image.

    Args:
        img: Original image (H, W, 3) in RGB format
        seg_pred: Segmentation prediction (C, H, W) probabilities or (H, W) as argmax
        exist_pred: Existence prediction (num_lanes,) probabilities
        threshold: Threshold for existence prediction

    Returns:
        img_overlay: Image with lane overlay (H, W, 3) in RGB format
        lane_img: Lane mask image (H, W, 3) in RGB format
    """
    img = img.copy()
    lane_img = np.zeros_like(img)

    # Get lane mask from segmentation prediction
    coord_mask = np.argmax(seg_pred, axis=0)

    # Draw each lane if it exists
    num_lanes = len(exist_pred)
    for i in range(num_lanes):
        if exist_pred[i] > threshold:
            lane_img[coord_mask == (i + 1)] = LANE_COLORS[i]

    # Create overlay
    img_overlay = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1.0, gamma=0.0)

    return img_overlay, lane_img
