from .config import load_yaml_config
from .weights import resolve_model_files


_MISSING = object()


def _select_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _config_value(config, section_name, key, default=_MISSING):
    section = config.get(section_name, {})

    if isinstance(section, dict) and key in section:
        return section[key]
    if key in config:
        return config[key]
    if default is not _MISSING:
        return default

    raise KeyError(f"Missing required config value: {section_name}.{key}")


def _config_value_from_sections(config, section_names, key, default=_MISSING):
    for section_name in section_names:
        section = config.get(section_name, {})
        if isinstance(section, dict) and key in section:
            return section[key]

    if key in config:
        return config[key]
    if default is not _MISSING:
        return default

    section_list = ", ".join(section_names)
    raise KeyError(f"Missing required config value: {section_list}.{key}")


def _model_kwargs_from_config(config):
    return {
        "architecture": _config_value(config, "model", "architecture"),
        "spatial_dims": _config_value(config, "model", "spatial_dims"),
        "in_channels": _config_value(config, "model", "in_channels"),
        "out_channels": _config_value(config, "model", "out_channels"),
        "channels": _config_value(config, "model", "channels"),
        "strides": _config_value(config, "model", "strides"),
        "num_res_units": _config_value(config, "model", "num_res_units", None),
    }


def _inference_value(config, key, default=_MISSING):
    return _config_value_from_sections(config, ("inference", "validation"), key, default)


def _postprocessing_value(config, key, override=None, default=_MISSING):
    if override is not None:
        return override
    return _config_value(config, "postprocessing", key, default)


def _load_checkpoint(weights_path):
    import torch

    suffix = weights_path.suffix.lower()

    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "safetensors is required to load .safetensors model weights. "
                "Install it with `pip install safetensors`."
            ) from exc

        return load_file(weights_path, device="cpu")

    return torch.load(weights_path, map_location="cpu")


def _extract_state_dict(checkpoint):
    import torch

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    for key in ("model_state_dict", "state_dict", "network", "model"):
        value = checkpoint.get(key)
        if isinstance(value, torch.nn.Module):
            return value
        if isinstance(value, dict):
            return value

    return checkpoint


def _normalise_state_dict_keys(state_dict):
    if not all(key.startswith("module.") for key in state_dict):
        return state_dict

    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def _prepare_image(image):
    import numpy as np
    import torch

    if torch.is_tensor(image):
        tensor = image.detach().clone()
    else:
        tensor = torch.as_tensor(np.asarray(image))

    tensor = tensor.float()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == 1:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(
            "Expected image with shape (H, W), (C, H, W), (N, H, W), "
            f"or (N, C, H, W); got {tuple(tensor.shape)}"
        )

    min_value = tensor.min()
    max_value = tensor.max()
    if max_value > min_value:
        tensor = (tensor - min_value) / (max_value - min_value)

    return tensor


def load_segmentation_model():
    import torch

    from .model import build_binary_model

    device = _select_device()
    files = resolve_model_files()
    config = load_yaml_config(files.config_path)
    checkpoint = _load_checkpoint(files.weights_path)
    state_or_model = _extract_state_dict(checkpoint)

    if isinstance(state_or_model, torch.nn.Module):
        model = state_or_model
    else:
        model = build_binary_model(**_model_kwargs_from_config(config))
        model.load_state_dict(_normalise_state_dict_keys(state_or_model))

    model.to(device)
    model.eval()

    return model, config


def predict_binary(model, image, device, roi_size, sw_batch_size,
                   prediction_sigmoid, prediction_threshold):
    import torch

    from monai.inferers import sliding_window_inference
    from monai.transforms import Activations, AsDiscrete, Compose

    post_pred = Compose([
        Activations(sigmoid=prediction_sigmoid),
        AsDiscrete(threshold=prediction_threshold),
    ])

    model.eval()
    image = _prepare_image(image)

    with torch.no_grad():

        output = sliding_window_inference(
            image.to(device),
            roi_size=tuple(roi_size),
            sw_batch_size=sw_batch_size,
            predictor=model,
        )

        prediction = post_pred(output)

    return prediction


def binary_to_instance_labels(binary_mask, min_size=20, min_distance=5):
    import numpy as np
    from scipy import ndimage as ndi
    from skimage import measure
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    # 1. Ensure boolean mask
    mask = binary_mask > 0

    # # 2. Optional cleanup
    # mask = remove_small_objects(mask, min_size=min_size)

    # 3. Distance transform: high values near object centers
    distance = ndi.distance_transform_edt(mask)

    # 4. Find likely object centers
    coords = peak_local_max(
        distance,
        labels=mask,
        min_distance=min_distance,
        exclude_border=False,
    )

    # 5. Convert center points into marker image
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    # 6. Expand markers using watershed
    labels = watershed(
        -distance,
        markers,
        mask=mask,
    )
    # Remove small objects from the labeled image
    labels = remove_small_objects(labels, min_size=min_size)

    # Reset label numbers and order from left to right
    labels = measure.label(labels)
    rp = measure.regionprops(labels)
    centroids = [c.centroid[1] for c in rp]
    sorted_indices = np.argsort(centroids)
    new_labels = np.zeros_like(labels)
    for new_label, old_index in enumerate(sorted_indices, start=1):
        new_labels[labels == rp[old_index].label] = new_label

    return new_labels


def predict_labels(
    image,
    min_size=None,
    min_distance=None,
):
    model, config = load_segmentation_model()

    device = next(model.parameters()).device
    roi_size = _inference_value(config, "roi_size")
    sw_batch_size = _inference_value(config, "sw_batch_size", 1)
    prediction_sigmoid = _inference_value(
        config,
        "prediction_sigmoid",
        True,
    )
    prediction_threshold = _inference_value(
        config,
        "prediction_threshold",
        0.5,
    )
    min_size = _postprocessing_value(config, "min_size", min_size, 20)
    min_distance = _postprocessing_value(config, "min_distance", min_distance, 5)

    pred = predict_binary(model, image, device, roi_size, sw_batch_size,
                          prediction_sigmoid, prediction_threshold)

    labels = binary_to_instance_labels(
        pred[0][0].cpu().numpy(),
        min_size=min_size,
        min_distance=min_distance,
    )

    return labels
