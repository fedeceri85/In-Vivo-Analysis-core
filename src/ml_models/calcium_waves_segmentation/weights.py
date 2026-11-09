from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from .config import load_yaml_config


DEFAULT_LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[3] / "models" / "calcium_waves_segmentation"
)
DEFAULT_CONFIG_FILENAME = "config.yml"
DEFAULT_MODEL_BASE_URL = (
    "https://huggingface.co/fedeceri/"
    "cochlea-calcium-waves-binary-segmentation-unet3d/resolve/main"
)


@dataclass(frozen=True)
class ModelFiles:
    weights_path: Path
    config_path: Path
    source: str


def resolve_config_path():
    """Return the cached config, downloading it if needed."""
    config_path = DEFAULT_LOCAL_MODEL_DIR / DEFAULT_CONFIG_FILENAME
    if not config_path.exists():
        _download_file(DEFAULT_CONFIG_FILENAME, config_path)
    return config_path


def resolve_model_files():
    """Return cached model files, downloading missing files from Hugging Face."""
    source = (
        "local" if (DEFAULT_LOCAL_MODEL_DIR / DEFAULT_CONFIG_FILENAME).exists() else "url"
    )
    config_path = resolve_config_path()

    config = load_yaml_config(config_path)
    weights_filename = config["model"]["weights"]
    weights_path = DEFAULT_LOCAL_MODEL_DIR / weights_filename
    if not weights_path.exists():
        _download_file(weights_filename, weights_path)
        source = "url"

    return ModelFiles(weights_path, config_path, source)


def _download_file(filename, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    urlretrieve(f"{DEFAULT_MODEL_BASE_URL}/{filename}", temp_path)
    temp_path.replace(destination)
