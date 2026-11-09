"""3D calcium-wave segmentation of (time, height, width) movies."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import inference
    from .inference import load_segmentation_model, predict_binary, predict_labels


__all__ = ["inference", "load_segmentation_model", "predict_binary", "predict_labels"]


def __getattr__(name):
    if name == "inference":
        return import_module(f"{__name__}.inference")
    if name in __all__:
        return getattr(import_module(f"{__name__}.inference"), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
