# RESA Lane Detection

PyTorch implementation of [RESA: Recurrent Feature-Shift Aggregator for Lane Detection](https://arxiv.org/abs/2008.13719) (AAAI 2021).

## Key Features

- **Recurrent Feature-Shift Aggregator**: Shifts sliced feature maps recurrently in 4 directions, enabling each pixel to gather global information
- **Multi-dataset Support**: CULane and TuSimple with dataset-specific decoders
- **Resolution-agnostic**: Works with any input size divisible by 8

## Installation
```bash
cd resa_torch
pip install -e .
```

For training:
```bash
pip install -e .[train]
```

## Dataset

Download [CULane](https://xingangpan.github.io/projects/CULane.html) and/or [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3) and create symlinks:
```bash
mkdir -p data
ln -s /path/to/CULane data/CULane
ln -s /path/to/tusimple data/tusimple
```

Expected structure:
```
data/CULane/
├── driver_*/
├── laneseg_label_w16/
└── list/
    ├── train_gt.txt
    ├── val_gt.txt
    ├── test.txt
    └── test_split/

data/tusimple/
├── clips/
├── label_data_*.json
├── test_label.json
└── seg_label/          # Generated
```

For TuSimple, generate segmentation labels:
```bash
python tools/generate_seg_tusimple.py --root data/tusimple
```

## Training
```bash
python tools/train.py --config configs/resa_culane.yaml
python tools/train.py --config configs/resa_tusimple.yaml
```

Resume from checkpoint:
```bash
python tools/train.py --config configs/resa_culane.yaml --resume checkpoints/latest.pth
```

## Testing
```bash
python tools/test.py --config configs/resa_culane.yaml --checkpoint checkpoints/best.pth
python tools/test.py --config configs/resa_tusimple.yaml --checkpoint checkpoints/best.pth
```

With visualization:
```bash
python tools/test.py --config configs/resa_culane.yaml --checkpoint checkpoints/best.pth --visualize
```

## Evaluation
```bash
python tools/evaluate.py --config configs/resa_culane.yaml --pred_dir outputs/predictions
python tools/evaluate.py --config configs/resa_tusimple.yaml --pred_dir outputs/predictions
```

## Model Architecture
```
Input (B, 3, H, W)
    │
    ▼
ResNet Backbone ─────────── (B, 512, H/8, W/8)
    │
    ▼
Channel Reduction ───────── (B, 128, H/8, W/8)
    │
    ▼
RESA Aggregator ─────────── (B, 128, H/8, W/8)
    │                       4-direction recurrent feature shifting
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Decoder                           ExistHead
    │                                  │
    ▼                                  ▼
seg_pred (B, C, H, W)            exist_pred (B, num_lanes)
```

**Decoders:**
- CULane: PlainDecoder (bilinear upsample)
- TuSimple: BUSDDecoder (3-stage learnable upsample)

## Project Structure
```
resa_torch/
├── configs/
│   ├── resa_culane.yaml
│   └── resa_tusimple.yaml
├── tools/
│   ├── train.py
│   ├── test.py
│   └── evaluate.py
└── resa_torch/
    ├── datasets/
    │   ├── culane.py
    │   └── tusimple.py
    ├── model/
    │   ├── backbone/resnet.py
    │   ├── aggregator/resa_aggregator.py
    │   ├── decoder/{plain,busd}_decoder.py
    │   ├── head/exist_head.py
    │   ├── loss/resa_loss.py
    │   └── net/resa.py
    ├── engine/
    │   ├── trainer.py
    │   └── evaluator.py
    └── utils/
```

## Reference
```bibtex
@inproceedings{zheng2021resa,
  title={RESA: Recurrent Feature-Shift Aggregator for Lane Detection},
  author={Zheng, Tu and Fang, Hao and Zhang, Yi and Tang, Wenjian and Yang, Zheng and Liu, Haifeng and Cai, Deng},
  booktitle={AAAI},
  year={2021}
}
```
