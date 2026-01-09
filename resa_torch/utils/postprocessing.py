"""
Postprocessing utilities for lane detection.

Converts probability maps to lane coordinates for both CULane and TuSimple formats.
"""

from pathlib import Path

import cv2
import numpy as np


def resize_seg_pred(seg_pred: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize segmentation prediction to target size.

    Args:
        seg_pred: Segmentation probabilities (C, H, W)
        target_size: Target size (H, W)

    Returns:
        Resized segmentation probabilities (C, target_H, target_W)
    """
    target_h, target_w = target_size
    num_classes = seg_pred.shape[0]
    resized = np.zeros((num_classes, target_h, target_w), dtype=seg_pred.dtype)

    for i in range(num_classes):
        resized[i] = cv2.resize(seg_pred[i], (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    return resized


def get_lane_coords(prob_map: np.ndarray, y_px_gap: int, pts: int, thresh: float) -> np.ndarray:
    """
    Extract lane coordinates from probability map for CULane format.

    Args:
        prob_map: Probability map for single lane (H, W) at original image size
        y_px_gap: Y pixel gap for sampling
        pts: Number of points to sample per lane
        thresh: Probability threshold

    Returns:
        X coordinates bottom up every y_px_gap px, 0 for non-exist
    """
    H, W = prob_map.shape

    coords = np.zeros(pts)
    for i in range(pts):
        y = H - 1 - i * y_px_gap
        if y < 0:
            break
        line = prob_map[y, :]
        idx = np.argmax(line)
        if line[idx] > thresh:
            coords[i] = idx

    if (coords > 0).sum() < 2:
        coords = np.zeros(pts)

    return coords


def prob2lines(
    seg_pred: np.ndarray,
    exist: np.ndarray,
    smooth: bool = True,
    y_px_gap: int = 20,
    pts: int = 18,
    thresh: float = 0.3,
    exist_thresh: float = 0.5
) -> list[list[tuple[int, int]]]:
    """
    Convert probability map to lane coordinates for CULane format.

    Args:
        seg_pred: Segmentation prediction (num_classes, H, W) at original image size
        exist: Lane existence probabilities (num_lanes,)
        smooth: Whether to smooth the probability map
        y_px_gap: Y pixel gap for sampling
        pts: Number of points per lane
        thresh: Probability threshold for lane detection
        exist_thresh: Probability threshold for lane existence

    Returns:
        List of lane coordinates, each lane is list of (x, y) tuples
    """
    num_classes, H, W = seg_pred.shape
    num_lanes = num_classes - 1  # Exclude background
    coordinates = []

    # Transpose to (H, W, C) for easier processing
    seg_pred = np.ascontiguousarray(np.transpose(seg_pred, (1, 2, 0)))

    for i in range(num_lanes):
        if exist[i] < exist_thresh:
            continue

        prob_map = seg_pred[..., i + 1]  # Skip background class

        if smooth:
            prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)

        coords = get_lane_coords(prob_map, y_px_gap, pts, thresh)

        if (coords > 0).sum() < 2:
            continue

        # Convert to (x, y) tuples, only include valid points
        lane_coords = [
            (int(coords[j]), H - 1 - j * y_px_gap)
            for j in range(pts)
            if coords[j] > 0
        ]
        coordinates.append(lane_coords)

    return coordinates


def get_save_path(img_name: str, base_dir: Path, suffix: str) -> Path:
    """
    Get save path for output file.

    Preserves the CULane directory structure:
        base_dir/driver_xxx/xxx/xxx{suffix}

    Args:
        img_name: Original image path
        base_dir: Base directory for saving
        suffix: File suffix (e.g., '.lines.txt' or '.jpg')

    Returns:
        Path to save file
    """
    img_path = Path(img_name)

    # Build output path: base_dir/driver_xxx/xxx/xxx{suffix}
    save_name = img_path.stem + suffix
    save_path = base_dir / img_path.parts[-3] / img_path.parts[-2] / save_name

    return save_path


# TuSimple-specific functions

def get_lane_coords_tusimple(
    prob_map: np.ndarray,
    h_samples: list[int],
    thresh: float
) -> list[int]:
    """
    Extract lane coordinates from probability map for TuSimple format.

    Args:
        prob_map: Probability map for single lane (H, W) at original image size
        h_samples: List of y-coordinates to sample
        thresh: Probability threshold

    Returns:
        X coordinates at each h_sample, -2 for non-exist
    """
    H, W = prob_map.shape
    coords = []

    for y in h_samples:
        if y < 0 or y >= H:
            coords.append(-2)
            continue

        line = prob_map[y, :]
        idx = np.argmax(line)
        if line[idx] > thresh:
            coords.append(int(idx))
        else:
            coords.append(-2)

    return coords


def prob2lines_tusimple(
    seg_pred: np.ndarray,
    exist: np.ndarray,
    h_samples: list[int],
    smooth: bool = True,
    thresh: float = 0.6,
    exist_thresh: float = 0.5
) -> list[list[int]]:
    """
    Convert probability map to lane coordinates for TuSimple format.

    Args:
        seg_pred: Segmentation prediction (num_classes, H, W) at original image size
        exist: Lane existence probabilities (num_lanes,)
        h_samples: List of y-coordinates to sample (from ground truth)
        smooth: Whether to smooth the probability map
        thresh: Probability threshold for lane detection
        exist_thresh: Probability threshold for lane existence

    Returns:
        List of lanes, each lane is list of x-coordinates at h_samples
    """
    num_lanes = len(exist)
    lanes = []

    # Transpose to (H, W, C) for easier processing
    seg_pred = np.ascontiguousarray(np.transpose(seg_pred, (1, 2, 0)))

    for i in range(num_lanes):
        if exist[i] < exist_thresh:
            continue

        prob_map = seg_pred[..., i + 1]  # Skip background class

        if smooth:
            prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)

        coords = get_lane_coords_tusimple(prob_map, h_samples, thresh)

        # Check if lane has enough valid points
        valid_count = sum(1 for x in coords if x >= 0)
        if valid_count < 2:
            continue

        lanes.append(coords)

    return lanes


# TuSimple h_samples (fixed for the dataset)
TUSIMPLE_H_SAMPLES = list(range(160, 720, 10))  # 160 to 710, step 10
