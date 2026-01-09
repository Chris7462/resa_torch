"""
Lane Detection Evaluation Script

Evaluates lane detection predictions against ground truth.
Supports CULane and TuSimple datasets.

Usage:
    python tools/evaluate.py --config configs/resa_culane.yaml --pred_dir outputs/predictions
    python tools/evaluate.py --config configs/resa_tusimple.yaml --pred_dir outputs/predictions
"""

import argparse
from pathlib import Path

from resa_torch.utils import load_config
from resa_torch.utils.culane_eval import evaluate_culane
from resa_torch.utils.tusimple_eval import evaluate_tusimple


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate lane detection predictions')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing prediction files')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset root (default: from config)')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluate',
                        help='Directory to save evaluation results')
    parser.add_argument('--iou', type=float, default=None,
                        help='IoU threshold (default: from config)')
    parser.add_argument('--lane_width', type=int, default=None,
                        help='Lane width for drawing (default: from config)')
    return parser.parse_args()


def evaluate_culane_dataset(args, config: dict) -> None:
    """Run CULane evaluation."""
    data_dir = Path(args.data_dir if args.data_dir is not None else config['dataset']['root'])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config['evaluation']
    iou_thresh = args.iou if args.iou is not None else eval_cfg['iou_threshold']
    lane_width = args.lane_width if args.lane_width is not None else eval_cfg['lane_width']

    pred_dir = Path(args.pred_dir)
    list_dir = data_dir / 'list' / 'test_split'

    # Check paths
    if not pred_dir.exists():
        print(f"Error: Prediction directory not found: {pred_dir}")
        return

    if not list_dir.exists():
        print(f"Error: Test split list directory not found: {list_dir}")
        return

    print("=" * 60)
    print("CULane Evaluation")
    print("=" * 60)
    print(f"Predictions:  {pred_dir}")
    print(f"Data dir:     {data_dir}")
    print(f"IoU thresh:   {iou_thresh}")
    print(f"Lane width:   {lane_width}")
    print("=" * 60)

    # CULane test categories
    categories = {
        'Normal': 'test0_normal.txt',
        'Crowd': 'test1_crowd.txt',
        'Hlight': 'test2_hlight.txt',
        'Shadow': 'test3_shadow.txt',
        'Noline': 'test4_noline.txt',
        'Arrow': 'test5_arrow.txt',
        'Curve': 'test6_curve.txt',
        'Cross': 'test7_cross.txt',
        'Night': 'test8_night.txt',
    }

    results = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for category, list_file in categories.items():
        list_path = list_dir / list_file

        if not list_path.exists():
            print(f"\nSkipping {category}: list file not found")
            continue

        print(f"\nEvaluating {category}...")

        result = evaluate_culane(
            pred_dir=pred_dir,
            anno_dir=data_dir,
            list_file=list_path,
            iou_thresh=iou_thresh,
            lane_width=lane_width,
            verbose=False,
        )

        results[category] = result

        # Cross category only counts FP (no ground truth lanes)
        if category == 'Cross':
            print(f"  FP: {result['fp']}")
        else:
            print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall:    {result['recall']:.4f}")
            print(f"  F1:        {result['f1']:.4f}")

            total_tp += result['tp']
            total_fp += result['fp']
            total_fn += result['fn']

        # Save individual result
        out_file = output_dir / f"out_{category}.txt"
        with open(out_file, 'w') as f:
            f.write(f"Category: {category}\n")
            f.write(f"TP: {result['tp']} FP: {result['fp']} FN: {result['fn']}\n")
            f.write(f"Precision: {result['precision']:.6f}\n")
            f.write(f"Recall: {result['recall']:.6f}\n")
            f.write(f"F1: {result['f1']:.6f}\n")

    # Compute overall metrics (excluding cross)
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Category':<12} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 60)

    for category, result in results.items():
        if category == 'Cross':
            print(f"{category:<12} {'N/A':<10} {'N/A':<12} {'N/A':<10} (FP: {result['fp']})")
        else:
            print(f"{category:<12} {result['f1']:<10.4f} {result['precision']:<12.4f} {result['recall']:<10.4f}")

    print("-" * 60)
    print(f"{'Overall':<12} {overall_f1:<10.4f} {overall_precision:<12.4f} {overall_recall:<10.4f}")
    print("=" * 60)

    # Save summary
    summary_file = output_dir / f"summary_iou{iou_thresh}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"CULane Evaluation Results\n")
        f.write(f"IoU threshold: {iou_thresh}\n")
        f.write(f"Lane width: {lane_width}\n\n")

        f.write(f"{'Category':<12} {'F1':<10} {'Precision':<12} {'Recall':<10} {'TP':<8} {'FP':<8} {'FN':<8}\n")
        f.write("-" * 80 + "\n")

        for category, result in results.items():
            if category == 'Cross':
                f.write(f"{category:<12} {'N/A':<10} {'N/A':<12} {'N/A':<10} {result['tp']:<8} {result['fp']:<8} {result['fn']:<8}\n")
            else:
                f.write(f"{category:<12} {result['f1']:<10.4f} {result['precision']:<12.4f} {result['recall']:<10.4f} {result['tp']:<8} {result['fp']:<8} {result['fn']:<8}\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'Overall':<12} {overall_f1:<10.4f} {overall_precision:<12.4f} {overall_recall:<10.4f} {total_tp:<8} {total_fp:<8} {total_fn:<8}\n")

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_file}")


def evaluate_tusimple_dataset(args, config: dict) -> None:
    """Run TuSimple evaluation."""
    data_dir = Path(args.data_dir if args.data_dir is not None else config['dataset']['root'])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.pred_dir)
    pred_file = pred_dir / 'predict_test.json'
    gt_file = data_dir / 'test_label.json'

    # Check paths
    if not pred_file.exists():
        print(f"Error: Prediction file not found: {pred_file}")
        return

    if not gt_file.exists():
        print(f"Error: Ground truth file not found: {gt_file}")
        return

    print("=" * 60)
    print("TuSimple Evaluation")
    print("=" * 60)
    print(f"Predictions:  {pred_file}")
    print(f"Ground truth: {gt_file}")
    print("=" * 60)

    result = evaluate_tusimple(pred_file, gt_file)

    print(f"\nAccuracy: {result['accuracy']:.4f}")
    print(f"FP:       {result['fp']:.4f}")
    print(f"FN:       {result['fn']:.4f}")

    # Save results
    summary_file = output_dir / "summary_tusimple.txt"
    with open(summary_file, 'w') as f:
        f.write("TuSimple Evaluation Results\n\n")
        f.write(f"Accuracy: {result['accuracy']:.6f}\n")
        f.write(f"FP: {result['fp']:.6f}\n")
        f.write(f"FN: {result['fn']:.6f}\n")

    print(f"\nResults saved to: {summary_file}")


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Determine dataset type
    dataset_type = config['dataset']['type']

    if dataset_type == 'CULane':
        evaluate_culane_dataset(args, config)
    elif dataset_type == 'TuSimple':
        evaluate_tusimple_dataset(args, config)
    else:
        print(f"Error: Unknown dataset type: {dataset_type}")


if __name__ == '__main__':
    main()
