import inspect
from typing import Any


class Registry:
    """
    Registry for managing component classes.

    Example:
        >>> DATASETS = Registry('dataset')
        >>> @DATASETS.register
        ... class CULane:
        ...     pass
        >>> dataset_cls = DATASETS.get('CULane')
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: dict[str, type] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> dict[str, type]:
        return self._module_dict

    def get(self, key: str) -> type | None:
        return self._module_dict.get(key)

    def register(self, cls: type) -> type:
        """Register a class using its class name."""
        if not inspect.isclass(cls):
            raise TypeError(f"Expected class, got {type(cls)}")

        name = cls.__name__
        if name in self._module_dict:
            raise KeyError(f"'{name}' already registered in {self._name}")

        self._module_dict[name] = cls
        return cls

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"


def build_from_cfg(cfg: dict, registry: Registry, **default_args) -> Any:
    """
    Build an instance from config dict.

    Args:
        cfg: Config dict with 'type' key
        registry: Registry to look up class
        **default_args: Default arguments to pass to constructor

    Returns:
        Instantiated object
    """
    if "type" not in cfg:
        raise KeyError(f"Config must have 'type' key: {cfg}")

    obj_type = cfg["type"]
    obj_cls = registry.get(obj_type)

    if obj_cls is None:
        raise KeyError(f"'{obj_type}' not found in {registry.name} registry")

    # Merge config args with default args
    args = {k: v for k, v in cfg.items() if k != "type"}
    args.update(default_args)

    return obj_cls(**args)
