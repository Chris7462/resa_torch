import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from resa_torch.datasets import build_dataset
from resa_torch.datasets.transforms import get_train_transforms, get_val_transforms
from resa_torch.model import RESA
from resa_torch.model.loss import RESALoss
from resa_torch.engine import Trainer, PolyLR
from resa_torch.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train RESA for lane detection')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def build_transforms(config: dict, is_train: bool = True):
    """Build transforms for training or validation."""
    resize_shape = tuple(config['preprocessing']['resize_shape'])
    mean = tuple(config['normalize']['mean'])
    std = tuple(config['normalize']['std'])

    if is_train:
        rotation = config['augmentation']['rotation']
        return get_train_transforms(resize_shape, mean, std, rotation=rotation)
    else:
        return get_val_transforms(resize_shape, mean, std)


def build_dataloader(config: dict, image_set: str, transforms):
    """Build dataloader for given image set."""
    dataset = build_dataset(
        config['dataset'],
        image_set=image_set,
        transforms=transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=(image_set == 'train'),
        num_workers=config['dataloader']['num_workers'],
        collate_fn=dataset.collate,
        pin_memory=True,
        drop_last=(image_set == 'train')
    )

    return dataloader


def build_model(config: dict):
    """Build RESA model."""
    model_cfg = config['model']

    model = RESA(
        backbone=model_cfg['backbone'],
        pretrained=model_cfg['pretrained'],
        num_classes=model_cfg['num_classes'],
        aggregator_channels=model_cfg['aggregator_channels'],
        aggregator_iters=model_cfg['aggregator_iters'],
        aggregator_kernel_size=model_cfg['aggregator_kernel_size'],
        aggregator_alpha=model_cfg['aggregator_alpha'],
        decoder_type=model_cfg['decoder_type'],
        exist_pool_size=tuple(model_cfg['exist_pool_size']),
    )

    return model


def build_optimizer(config: dict, model):
    """Build optimizer."""
    optimizer_cfg = config['optimizer']

    optimizer = optim.SGD(
        model.parameters(),
        lr=optimizer_cfg['lr'],
        momentum=optimizer_cfg['momentum'],
        weight_decay=optimizer_cfg['weight_decay'],
        nesterov=optimizer_cfg['nesterov']
    )

    return optimizer


def build_lr_scheduler(config: dict, optimizer):
    """Build PolyLR scheduler with warmup."""
    max_iter = config['train']['max_iter']
    lr_cfg = config['lr_scheduler']

    scheduler = PolyLR(
        optimizer,
        max_iter=max_iter,
        power=lr_cfg['power'],
        warmup=lr_cfg['warmup'],
        min_lr=lr_cfg['min_lr']
    )

    return scheduler


def build_criterion(config: dict):
    """Build loss function."""
    loss_cfg = config['loss']
    model_cfg = config['model']

    criterion = RESALoss(
        num_classes=model_cfg['num_classes'],
        loss_type=loss_cfg['type'],
        seg_weight=loss_cfg['seg_weight'],
        exist_weight=loss_cfg['exist_weight'],
        background_weight=loss_cfg['background_weight'],
        ignore_index=loss_cfg['ignore_index']
    )

    return criterion


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Set random seed for reproducibility
    seed = config['train']['seed']
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build transforms
    train_transforms = build_transforms(config, is_train=True)
    val_transforms = build_transforms(config, is_train=False)

    # Build dataloaders
    print("Building dataloaders...")
    train_loader = build_dataloader(config, 'train', train_transforms)
    val_loader = build_dataloader(config, 'val', val_transforms)
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # Build model
    print("Building model...")
    model = build_model(config).to(device)

    # Build optimizer
    optimizer = build_optimizer(config, model)

    # Build lr scheduler
    print("Building PolyLR scheduler...")
    lr_scheduler = build_lr_scheduler(config, optimizer)
    lr_cfg = config['lr_scheduler']
    print(f"  Power: {lr_cfg['power']}")
    print(f"  Warmup: {lr_cfg['warmup']} iterations")
    print(f"  Initial LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Build criterion
    criterion = build_criterion(config).to(device)

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    print(f"  Max iterations: {config['train']['max_iter']}")
    print(f"  Checkpoint interval: {config['checkpoint']['interval']}")
    print("=" * 60)
    trainer.train()


if __name__ == '__main__':
    main()
