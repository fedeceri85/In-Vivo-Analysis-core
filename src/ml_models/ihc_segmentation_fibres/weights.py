from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from .config import load_yaml_config


DEFAULT_LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[3] / "models" / "ihc_segmentation_synaptic_hcs"
)
DEFAULT_CONFIG_FILENAME = "config.yml"
DEFAULT_WEIGHTS_FILENAME = "best_model_binary_v4_synaptic_hcs.pth"
DEFAULT_CONFIG_URL = (
    "https://huggingface.co/fedeceri/"
    "cochlea-hair-cell-synapses-binary-segmentation-unet2d/resolve/main/config.yml"
)
DEFAULT_WEIGHTS_URL = (
    "https://huggingface.co/fedeceri/"
    "cochlea-hair-cell-synapses-binary-segmentation-unet2d/resolve/main/"
    "best_model_binary_v4_synaptic_hcs.pth"
)


@dataclass(frozen=True)
class ModelFiles:
    weights_path: Path
    config_path: Path
    source: str


def resolve_model_files():
    """Return local model files, downloading them from public URLs if needed."""
    model_dir = DEFAULT_LOCAL_MODEL_DIR
    config_path = model_dir / DEFAULT_CONFIG_FILENAME

    if not config_path.exists():
        if DEFAULT_CONFIG_URL is None:
            raise FileNotFoundError(
                "The model config was not found locally and no config URL "
                "was configured. "
                f"Missing file: {config_path}. "
                "Set DEFAULT_CONFIG_URL in weights.py."
            )

        _download_file(
            url=DEFAULT_CONFIG_URL,
            destination=config_path,
        )

    config = load_yaml_config(config_path)
    weights_filename = _first_config_value(
        config,
        (
            ("weights", "filename"),
            ("model", "best_model_path"),
        ),
        DEFAULT_WEIGHTS_FILENAME,
    )
    weights_url = _config_value(config, "weights", "url", DEFAULT_WEIGHTS_URL)
    weights_path = model_dir / weights_filename

    if weights_path.exists():
        return ModelFiles(
            weights_path=weights_path,
            config_path=config_path,
            source="local",
        )

    if weights_url is None:
        raise FileNotFoundError(
            "The model weights were not found locally and no weights URL "
            "was configured. "
            f"Missing file: {weights_path}. "
            "Set DEFAULT_WEIGHTS_URL in weights.py or weights.url in config.yml."
        )

    _download_file(
        url=weights_url,
        destination=weights_path,
    )

    return ModelFiles(
        weights_path=weights_path,
        config_path=config_path,
        source="url",
    )


def _config_value(config, section_name, key, default=None):
    section = config.get(section_name, {})

    if isinstance(section, dict) and key in section:
        return section[key]
    if key in config:
        return config[key]

    return default


def _first_config_value(config, candidates, default=None):
    for section_name, key in candidates:
        value = _config_value(config, section_name, key)
        if value is not None:
            return value

    return default


def _download_file(url, destination):
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported model download URL: {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    urlretrieve(url, temp_path)
    temp_path.replace(destination)
