from pathlib import Path

from repository_paths import (
    get_model_output_path,
    get_repository_path,
    resolve_repository_path,
)


def test_repository_paths_follow_environment_and_preserve_absolute_data(tmp_path, monkeypatch):
    monkeypatch.setenv("LSST_STRONG_LENS_DATA_CHALLENGE_PATH", str(tmp_path))
    external_data = tmp_path.parent / "challenge-data"

    assert get_repository_path() == tmp_path.resolve()
    assert resolve_repository_path("data/dataset.npz") == tmp_path / "data" / "dataset.npz"
    assert resolve_repository_path(external_data) == external_data
    assert get_model_output_path("v4_retrain.keras") == (
        tmp_path / "data" / "models" / "v4_retrain.keras"
    )