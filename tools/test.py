import argparse

import torch
from torch.utils.data import DataLoader

from resa_torch.datasets import DATASETS
from resa_torch.datasets.transforms import get_val_transforms
from resa_torch.model import RESA
from resa_torch.engine import Evaluator
from resa_torch.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test RESA for lane detection')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--num_visualize', type=int, default=20,
                        help='Number of images to visualize (default: 20)')
    return parser.parse_args()


def build_transforms(config: dict):
    """Build transforms for testing."""
    resize_shape = tuple(config['dataset']['resize_shape'])
    mean = tuple(config['normalize']['mean'])
    std = tuple(config['normalize']['std'])

    return get_val_transforms(resize_shape, mean, std)


def build_dataloader(config: dict, transforms):
    """Build test dataloader."""
    dataset_cfg = config['dataset']
    dataset_cls = DATASETS.get(dataset_cfg['type'])

    dataset = dataset_cls(
        root=dataset_cfg['root'],
        image_set='test',
        transforms=transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        collate_fn=dataset.collate,
        pin_memory=True
    )

    return dataloader


def build_model(config: dict):
    """Build RESA model."""
    model_cfg = config['model']

    model = RESA(
        backbone=model_cfg['backbone'],
        pretrained=False,
        num_classes=model_cfg['num_classes'],
        aggregator_channels=model_cfg['aggregator_channels'],
        aggregator_iters=model_cfg['aggregator_iters'],
        aggregator_kernel_size=model_cfg['aggregator_kernel_size'],
        aggregator_alpha=model_cfg['aggregator_alpha'],
        decoder_type=model_cfg['decoder_type'],
    )

    return model


def load_checkpoint(model, checkpoint_path: str):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['net'])

    print(f"  Loaded from iteration {checkpoint['iteration']}")
    return model


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Add output_dir to config
    config['output_dir'] = args.output_dir

    # Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build transforms
    transforms = build_transforms(config)

    # Build dataloader
    print("Building dataloader...")
    test_loader = build_dataloader(config, transforms)
    print(f"  Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    # Build model
    print("Building model...")
    model = build_model(config).to(device)

    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint)

    # Build evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device,
        visualize=args.visualize,
        num_visualize=args.num_visualize
    )

    # Run evaluation
    print("\nRunning evaluation...")
    if args.visualize:
        print(f"  Visualizing first {args.num_visualize} images")
    output_dir = evaluator.evaluate()

    print(f"\nEvaluation complete!")
    print(f"Predictions saved to: {output_dir}/predictions")
    if args.visualize:
        print(f"Visualizations saved to: {output_dir}/visualizations")
    print(f"\nTo run official CULane evaluation:")
    print(f"python tools/evaluate.py --config {args.config} --pred_dir {output_dir}/predictions")


if __name__ == '__main__':
    main()
