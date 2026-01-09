"""
TuSimple Evaluation in Python

Implements the official TuSimple evaluation metrics:
1. Accuracy: Average of per-lane accuracies
2. FP: False positive rate
3. FN: False negative rate

Based on the official TuSimple benchmark evaluation code.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression


class TuSimpleEvaluator:
    """
    TuSimple lane detection evaluator.

    Args:
        pixel_thresh: Pixel distance threshold for point matching (default: 20)
        pt_thresh: Threshold for lane accuracy to count as match (default: 0.85)
    """

    def __init__(
        self,
        pixel_thresh: int = 20,
        pt_thresh: float = 0.85,
    ):
        self.pixel_thresh = pixel_thresh
        self.pt_thresh = pt_thresh
        self.lr = LinearRegression()

    def get_angle(self, xs: np.ndarray, y_samples: np.ndarray) -> float:
        """
        Get the angle of a lane line.

        Args:
            xs: X coordinates of lane points
            y_samples: Y coordinates (h_samples)

        Returns:
            Angle in radians
        """
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            self.lr.fit(ys[:, None], xs)
            k = self.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    def line_accuracy(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        thresh: float
    ) -> float:
        """
        Compute accuracy between predicted and ground truth lane.

        Args:
            pred: Predicted x coordinates
            gt: Ground truth x coordinates
            thresh: Pixel threshold for matching

        Returns:
            Accuracy score (0 to 1)
        """
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1.0, 0.0)) / len(gt)

    def bench(
        self,
        pred_lanes: list[list[int]],
        gt_lanes: list[list[int]],
        y_samples: list[int],
        run_time: float
    ) -> tuple[float, float, float]:
        """
        Benchmark a single image prediction.

        Args:
            pred_lanes: List of predicted lanes (each lane is list of x coords)
            gt_lanes: List of ground truth lanes
            y_samples: Y coordinates (h_samples)
            run_time: Inference time in ms

        Returns:
            accuracy: Average lane accuracy
            fp: False positive rate
            fn: False negative rate
        """
        if any(len(p) != len(y_samples) for p in pred_lanes):
            raise ValueError('Format of lanes error.')

        if run_time > 200 or len(gt_lanes) + 2 < len(pred_lanes):
            return 0.0, 0.0, 1.0

        y_samples = np.array(y_samples)

        # Compute angle-adjusted thresholds for each GT lane
        angles = [
            self.get_angle(np.array(x_gts), y_samples)
            for x_gts in gt_lanes
        ]
        threshs = [self.pixel_thresh / np.cos(angle) for angle in angles]

        line_accs = []
        matched = 0.0

        for x_gts, thresh in zip(gt_lanes, threshs):
            accs = [
                self.line_accuracy(np.array(x_preds), np.array(x_gts), thresh)
                for x_preds in pred_lanes
            ]
            max_acc = np.max(accs) if len(accs) > 0 else 0.0

            if max_acc < self.pt_thresh:
                pass  # False negative
            else:
                matched += 1

            line_accs.append(max_acc)

        fp = len(pred_lanes) - matched
        fn = len(gt_lanes) - matched

        # Handle case with more than 4 lanes
        if len(gt_lanes) > 4 and fn > 0:
            fn -= 1

        s = sum(line_accs)
        if len(gt_lanes) > 4:
            s -= min(line_accs)

        accuracy = s / max(min(4.0, len(gt_lanes)), 1.0)
        fp_rate = fp / len(pred_lanes) if len(pred_lanes) > 0 else 0.0
        fn_rate = fn / max(min(len(gt_lanes), 4.0), 1.0)

        return accuracy, fp_rate, fn_rate


def evaluate_tusimple(
    pred_file: str | Path,
    gt_file: str | Path,
    pixel_thresh: int = 20,
    pt_thresh: float = 0.85,
) -> dict:
    """
    Evaluate TuSimple predictions.

    Args:
        pred_file: Path to prediction JSON file
        gt_file: Path to ground truth JSON file
        pixel_thresh: Pixel distance threshold for point matching
        pt_thresh: Threshold for lane accuracy to count as match

    Returns:
        Dictionary with accuracy, fp, fn
    """
    pred_file = Path(pred_file)
    gt_file = Path(gt_file)

    # Load predictions
    try:
        with open(pred_file, 'r') as f:
            json_pred = [json.loads(line) for line in f.readlines()]
    except Exception as e:
        raise ValueError(f'Failed to load prediction file: {e}')

    # Load ground truth
    with open(gt_file, 'r') as f:
        json_gt = [json.loads(line) for line in f.readlines()]

    if len(json_gt) != len(json_pred):
        raise ValueError(
            f'Number of predictions ({len(json_pred)}) does not match '
            f'number of ground truth samples ({len(json_gt)})'
        )

    # Build GT lookup by raw_file
    gts = {entry['raw_file']: entry for entry in json_gt}

    evaluator = TuSimpleEvaluator(
        pixel_thresh=pixel_thresh,
        pt_thresh=pt_thresh,
    )

    total_accuracy = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for pred in json_pred:
        # Validate prediction format
        if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
            raise ValueError('raw_file, lanes, or run_time not in prediction')

        raw_file = pred['raw_file']
        pred_lanes = pred['lanes']
        run_time = pred['run_time']

        if raw_file not in gts:
            raise ValueError(f'Prediction file {raw_file} not in ground truth')

        gt = gts[raw_file]
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']

        try:
            acc, fp, fn = evaluator.bench(pred_lanes, gt_lanes, y_samples, run_time)
        except Exception as e:
            raise ValueError(f'Format of lanes error: {e}')

        total_accuracy += acc
        total_fp += fp
        total_fn += fn

    num_samples = len(gts)

    return {
        'accuracy': total_accuracy / num_samples,
        'fp': total_fp / num_samples,
        'fn': total_fn / num_samples,
    }
