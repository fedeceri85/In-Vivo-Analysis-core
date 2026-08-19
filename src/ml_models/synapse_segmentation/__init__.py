from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import inference
    from .inference import predict_labels


__all__ = ["inference", "predict_labels"]


def __getattr__(name):
    if name == "inference":
        return import_module(f"{__name__}.inference")
    if name == "predict_labels":
        return import_module(f"{__name__}.inference").predict_labels

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
