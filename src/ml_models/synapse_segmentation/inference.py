from .config import load_yaml_config
from .weights import resolve_model_files


_MISSING = object()
DEFAULT_POSTPROCESSING = {
    "min_size": 20,
    "min_distance": 5,
    "heatmap_threshold": 0.3,
}


def _select_device(device_name="auto"):
    import torch

    if device_name is None or device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")

    return torch.device(device_name)


def _config_value(config, section_name, key, default=_MISSING):
    section = config.get(section_name, {})

    if isinstance(section, dict) and key in section:
        return section[key]
    if key in config:
        return config[key]
    if default is not _MISSING:
        return default

    raise KeyError(f"Missing required config value: {section_name}.{key}")


def _model_kwargs_from_config(model_config):
    return {
        "architecture": model_config["architecture"],
        "spatial_dims": model_config["spatial_dims"],
        "in_channels": model_config["in_channels"],
        "out_channels": model_config["out_channels"],
        "channels": model_config["channels"],
        "strides": model_config["strides"],
        "num_res_units": model_config.get("num_res_units"),
    }


def _load_checkpoint(weights_path, map_location):
    import torch

    try:
        return torch.load(weights_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(weights_path, map_location=map_location)


def _branch_state_dict(checkpoint, branch_name):
    weights_key = f"{branch_name}_state_dict"
    if weights_key in checkpoint:
        return checkpoint[weights_key]

    branch = checkpoint.get(branch_name)
    if isinstance(branch, dict) and isinstance(branch.get("state_dict"), dict):
        return branch["state_dict"]

    raise KeyError(f"Combined checkpoint does not contain {weights_key}.")


def _branch_config(checkpoint, branch_name, key, fallback):
    branch = checkpoint.get(branch_name)
    if isinstance(branch, dict) and isinstance(branch.get(key), dict):
        return branch[key]
    return fallback


def _normalise_state_dict_keys(state_dict):
    if not all(key.startswith("module.") for key in state_dict):
        return state_dict

    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def _load_branch(checkpoint, branch_name, model_config, device):
    from .model import build_model

    model = build_model(**_model_kwargs_from_config(model_config))
    model.load_state_dict(
        _normalise_state_dict_keys(_branch_state_dict(checkpoint, branch_name))
    )
    model.to(device)
    model.eval()
    return model


def _prepare_image(image):
    import numpy as np

    image = np.asarray(image).squeeze().astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected one 2D image, got shape {image.shape}.")

    min_value = image.min()
    max_value = image.max()
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)

    return image


def _predict_binary(
        model,
        image,
        validation_config,
        device,
        roi_size=None,
        sw_batch_size=None,
        threshold=None,
):
    import torch

    from monai.inferers import sliding_window_inference
    from monai.transforms import Activations, AsDiscrete, Compose

    image_tensor = torch.from_numpy(image)[None, None].to(device)
    threshold = (
        validation_config["prediction_threshold"]
        if threshold is None
        else threshold
    )
    post_pred = Compose([
        Activations(sigmoid=validation_config["prediction_sigmoid"]),
        AsDiscrete(threshold=threshold),
    ])

    with torch.no_grad():
        output = sliding_window_inference(
            image_tensor,
            roi_size=tuple(roi_size or validation_config["roi_size"]),
            sw_batch_size=sw_batch_size or validation_config["sw_batch_size"],
            predictor=model,
        )
        prediction = post_pred(output)

    return prediction[0, 0].cpu().numpy().astype("uint8")


def _predict_heatmap(
        model,
        image,
        validation_config,
        device,
        roi_size=None,
        sw_batch_size=None,
):
    import torch

    from monai.inferers import sliding_window_inference
    from monai.transforms import Activations, Compose

    image_tensor = torch.from_numpy(image)[None, None].to(device)
    post_pred = Compose([
        Activations(sigmoid=validation_config["prediction_sigmoid"]),
    ])

    with torch.no_grad():
        output = sliding_window_inference(
            image_tensor,
            roi_size=tuple(roi_size or validation_config["roi_size"]),
            sw_batch_size=sw_batch_size or validation_config["sw_batch_size"],
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )
        prediction = post_pred(output)

    return prediction[0, 0].cpu().numpy().astype("float32")


def _heatmap_to_marker_image(heatmap, min_distance=5, threshold_abs=0.3):
    import numpy as np

    from skimage.feature import peak_local_max

    centers = peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
    )
    markers = np.zeros_like(heatmap, dtype=np.int32)
    for marker_id, (y, x) in enumerate(centers, start=1):
        markers[y, x] = marker_id

    return markers


def _remove_small_components(labels, min_size):
    from skimage.morphology import remove_small_objects

    if min_size <= 0:
        return labels

    return remove_small_objects(labels, min_size=min_size)


def _binary_to_instance_labels(
        binary_mask,
        min_size=20,
        min_distance=5,
        markers=None,
):
    import numpy as np
    from scipy import ndimage as ndi
    from skimage import measure
    from skimage.feature import peak_local_max
    from skimage.morphology import closing
    from skimage.segmentation import watershed

    mask = binary_mask > 0
    mask = _remove_small_components(mask, min_size=min_size)
    mask = closing(mask)
    distance = ndi.distance_transform_edt(mask)

    if markers is None:
        coords = peak_local_max(
            distance,
            labels=mask,
            min_distance=min_distance,
            exclude_border=False,
        )
        markers = np.zeros_like(mask, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    labels = watershed(-distance, markers, mask=mask)
    labels = _remove_small_components(labels, min_size=min_size)
    labels = measure.label(labels)

    props = measure.regionprops(labels)
    sorted_props = sorted(props, key=lambda prop: prop.centroid[1])
    relabeled = np.zeros_like(labels, dtype=np.uint32)
    for new_label, prop in enumerate(sorted_props, start=1):
        relabeled[labels == prop.label] = new_label

    return relabeled


def _postprocessing_value(postprocessing, key, override=None):
    if override is not None:
        return override
    return postprocessing.get(key, DEFAULT_POSTPROCESSING[key])


def load_segmentation_models():
    files = resolve_model_files()
    binary_config = load_yaml_config(files.binary_config_path)
    heatmap_config = load_yaml_config(files.heatmap_config_path)

    device_name = _config_value(binary_config, "training", "device", "auto")
    device = _select_device(device_name)
    checkpoint = _load_checkpoint(files.weights_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Combined model checkpoint must be a dictionary of state dicts.")

    binary_model_config = _branch_config(
        checkpoint,
        "binary",
        "model_config",
        binary_config["model"],
    )
    heatmap_model_config = _branch_config(
        checkpoint,
        "heatmap",
        "model_config",
        heatmap_config["model"],
    )
    binary_validation_config = _branch_config(
        checkpoint,
        "binary",
        "validation_config",
        binary_config["validation"],
    )
    heatmap_validation_config = _branch_config(
        checkpoint,
        "heatmap",
        "validation_config",
        heatmap_config["validation"],
    )

    binary_model = _load_branch(
        checkpoint,
        "binary",
        binary_model_config,
        device,
    )
    heatmap_model = _load_branch(
        checkpoint,
        "heatmap",
        heatmap_model_config,
        device,
    )

    postprocessing = dict(DEFAULT_POSTPROCESSING)
    postprocessing.update(binary_config.get("postprocessing", {}))
    postprocessing.update(heatmap_config.get("postprocessing", {}))
    if isinstance(checkpoint.get("postprocessing"), dict):
        postprocessing.update(checkpoint["postprocessing"])

    return {
        "binary_model": binary_model,
        "heatmap_model": heatmap_model,
        "binary_validation_config": binary_validation_config,
        "heatmap_validation_config": heatmap_validation_config,
        "postprocessing": postprocessing,
        "device": device,
    }


def predict_labels(
        image,
        min_size=None,
        min_distance=None,
        binary_threshold=None,
        heatmap_threshold=None,
        roi_size=None,
        sw_batch_size=None,
):
    loaded = load_segmentation_models()
    image = _prepare_image(image)

    postprocessing = loaded["postprocessing"]
    min_size = _postprocessing_value(postprocessing, "min_size", min_size)
    min_distance = _postprocessing_value(
        postprocessing,
        "min_distance",
        min_distance,
    )
    heatmap_threshold = _postprocessing_value(
        postprocessing,
        "heatmap_threshold",
        heatmap_threshold,
    )

    binary_mask = _predict_binary(
        loaded["binary_model"],
        image,
        loaded["binary_validation_config"],
        loaded["device"],
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        threshold=binary_threshold,
    )
    heatmap = _predict_heatmap(
        loaded["heatmap_model"],
        image,
        loaded["heatmap_validation_config"],
        loaded["device"],
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
    )
    markers = _heatmap_to_marker_image(
        heatmap,
        min_distance=min_distance,
        threshold_abs=heatmap_threshold,
    )

    return _binary_to_instance_labels(
        binary_mask,
        min_size=min_size,
        min_distance=min_distance,
        markers=markers,
    )
