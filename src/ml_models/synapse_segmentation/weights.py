from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve


DEFAULT_LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[3] / "models" / "synapse_segmentation"
)
DEFAULT_WEIGHTS_FILENAME = "best_model_binary_heatmap_v5_synaptic.pth"
DEFAULT_BINARY_CONFIG_FILENAME = "config/binary_model_v5_synaptic.yml"
DEFAULT_HEATMAP_CONFIG_FILENAME = "config/heatmap_model_v5_synaptic.yml"
DEFAULT_MODEL_BASE_URL = (
    "https://huggingface.co/fedeceri/"
    "cochlea-synapses-segmentation-unet2d/resolve/main"
)
DEFAULT_WEIGHTS_URL = f"{DEFAULT_MODEL_BASE_URL}/{DEFAULT_WEIGHTS_FILENAME}"
DEFAULT_BINARY_CONFIG_URL = f"{DEFAULT_MODEL_BASE_URL}/{DEFAULT_BINARY_CONFIG_FILENAME}"
DEFAULT_HEATMAP_CONFIG_URL = f"{DEFAULT_MODEL_BASE_URL}/{DEFAULT_HEATMAP_CONFIG_FILENAME}"


@dataclass(frozen=True)
class ModelFiles:
    weights_path: Path
    binary_config_path: Path
    heatmap_config_path: Path
    source: str


def resolve_model_files():
    """Return local model files, downloading them from public URLs if needed."""
    model_dir = DEFAULT_LOCAL_MODEL_DIR
    weights_path = model_dir / DEFAULT_WEIGHTS_FILENAME
    binary_config_path = model_dir / DEFAULT_BINARY_CONFIG_FILENAME
    heatmap_config_path = model_dir / DEFAULT_HEATMAP_CONFIG_FILENAME

    source = "local"
    for path, url, label in (
        (weights_path, DEFAULT_WEIGHTS_URL, "model weights"),
        (binary_config_path, DEFAULT_BINARY_CONFIG_URL, "binary model config"),
        (heatmap_config_path, DEFAULT_HEATMAP_CONFIG_URL, "heatmap model config"),
    ):
        if path.exists():
            continue
        if url is None:
            raise FileNotFoundError(
                f"The {label} file was not found locally and no URL was configured. "
                f"Missing file: {path}."
            )

        _download_file(
            url=url,
            destination=path,
        )
        source = "url"

    return ModelFiles(
        weights_path=weights_path,
        binary_config_path=binary_config_path,
        heatmap_config_path=heatmap_config_path,
        source=source,
    )


def _download_file(url, destination):
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported model download URL: {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    urlretrieve(url, temp_path)
    temp_path.replace(destination)
