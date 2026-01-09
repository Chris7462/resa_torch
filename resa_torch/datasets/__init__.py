from .registry import build_dataset

# These imports trigger @DATASETS.register decorator
from .culane import CULane
from .tusimple import TuSimple


__all__ = [
    "build_dataset",
]
