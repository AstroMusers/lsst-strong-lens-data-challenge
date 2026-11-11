"""Resolve repository inputs and outputs from the configured checkout root."""

import os
from pathlib import Path


PATH_ENV_VAR = "LSST_STRONG_LENS_DATA_CHALLENGE_PATH"
DEFAULT_REPOSITORY_PATH = Path(__file__).resolve().parent


def get_repository_path() -> Path:
    os.environ.setdefault(PATH_ENV_VAR, str(DEFAULT_REPOSITORY_PATH))
    return Path(os.environ[PATH_ENV_VAR]).expanduser().resolve()


def resolve_repository_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else get_repository_path() / path


def get_model_output_path(filename: str) -> Path:
    return get_repository_path() / "data" / "models" / filename