from resa_torch.utils import Registry, build_from_cfg


DATASETS = Registry("dataset")


def build_dataset(cfg: dict, **kwargs):
    """
    Build dataset from config.

    Args:
        cfg: Config dict with 'type' key (e.g., 'CULane', 'TuSimple')
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        Dataset instance
    """
    return build_from_cfg(cfg, DATASETS, **kwargs)
