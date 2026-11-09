"""Inference adapted from the calcium-wave model's Hugging Face release."""

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage as ndi

from .config import load_yaml_config
from .model import build_model
from .weights import resolve_model_files


def _select_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_segmentation_model(device="auto"):
    """Load the configured 3D U-Net and return (model, config)."""
    files = resolve_model_files()
    config = load_yaml_config(files.config_path)
    model = build_model(config)
    state_dict = torch.load(files.weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(_select_device(device)).eval(), config


def _predict(model, movie, config, threshold=None, roi_size=None, sw_batch_size=None):
    inference = config["inference"]
    image = torch.as_tensor(movie, dtype=torch.float32, device="cpu")[None, None]
    minimum = image.amin()
    intensity_range = image.amax() - minimum
    if intensity_range > 0:
        image = (image - minimum) / intensity_range
    else:
        image = torch.zeros_like(image)

    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=image,
            roi_size=tuple(inference["roi_size"] if roi_size is None else roi_size),
            sw_batch_size=(
                inference["sw_batch_size"] if sw_batch_size is None else sw_batch_size
            ),
            predictor=model,
            sw_device=next(model.parameters()).device,
            device="cpu",
            overlap=inference["overlap"],
            mode=inference["blend_mode"],
            sigma_scale=inference["sigma_scale"],
            buffer_steps=inference["buffer_steps"],
            buffer_dim=0,
            progress=True,
        )
        probabilities = torch.sigmoid(logits)[0, 0].numpy()

    threshold = inference["threshold"] if threshold is None else threshold
    return probabilities >= threshold, probabilities


def binary_to_instance_labels(
    mask, config, *, min_size=None, closing_size=None, connectivity=None
):
    """Close and label connected waves, then number them by start time."""
    postprocessing = config["postprocessing"]
    min_size = postprocessing["min_size"] if min_size is None else min_size
    closing_size = postprocessing["closing_size"] if closing_size is None else closing_size
    connectivity = postprocessing["connectivity"] if connectivity is None else connectivity

    mask = ndi.binary_closing(mask, structure=np.ones(tuple(closing_size), dtype=bool))
    labels, _ = ndi.label(mask, structure=ndi.generate_binary_structure(3, connectivity))
    sizes = np.bincount(labels.ravel())
    remove = sizes < min_size
    remove[0] = False
    labels[remove[labels]] = 0

    objects = ndi.find_objects(labels)
    existing = [label_id for label_id, slices in enumerate(objects, 1) if slices is not None]
    ordered = sorted(
        existing,
        key=lambda label_id: tuple(s.start for s in objects[label_id - 1]),
    )
    lookup = np.zeros(labels.max() + 1, dtype=np.uint32)
    lookup[ordered] = np.arange(1, len(ordered) + 1, dtype=np.uint32)
    return lookup[labels]


def predict_binary(
    movie,
    *,
    device="auto",
    threshold=None,
    roi_size=None,
    sw_batch_size=None,
    return_probabilities=False,
):
    """Return a (T, H, W) boolean mask, optionally with float32 probabilities."""
    model, config = load_segmentation_model(device)
    mask, probabilities = _predict(model, movie, config, threshold, roi_size, sw_batch_size)
    return (mask, probabilities) if return_probabilities else mask


def predict_labels(
    movie,
    *,
    device="auto",
    threshold=None,
    roi_size=None,
    sw_batch_size=None,
    min_size=None,
    closing_size=None,
    connectivity=None,
    return_probabilities=False,
):
    """Return uint32 wave labels, optionally with float32 probabilities."""
    model, config = load_segmentation_model(device)
    mask, probabilities = _predict(model, movie, config, threshold, roi_size, sw_batch_size)
    labels = binary_to_instance_labels(
        mask, config, min_size=min_size, closing_size=closing_size, connectivity=connectivity
    )
    return (labels, probabilities) if return_probabilities else labels
