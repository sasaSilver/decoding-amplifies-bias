from pathlib import Path

import pytest

from app.exai import ExAIDataConfig, ExAIPaths, ExAITrainingConfig


def test_exai_paths_live_under_outputs_namespace(tmp_path: Path) -> None:
    paths = ExAIPaths(root=tmp_path / "outputs" / "exai")

    assert paths.root == tmp_path / "outputs" / "exai"
    assert paths.manifests_dir == paths.root / "manifests"
    assert paths.models_dir == paths.root / "models"
    assert paths.explanations_dir == paths.root / "explanations"


def test_exai_paths_create_expected_directories(tmp_path: Path) -> None:
    paths = ExAIPaths(root=tmp_path / "outputs" / "exai").ensure_dirs()

    assert paths.manifests_dir.is_dir()
    assert paths.metadata_dir.is_dir()
    assert paths.models_dir.is_dir()
    assert paths.benchmark_dir.is_dir()


def test_exai_data_config_validates_split_fraction_sum() -> None:
    config = ExAIDataConfig(train_fraction=0.7, validation_fraction=0.2)

    assert config.test_fraction() == pytest.approx(0.1)


def test_exai_training_defaults_match_project_brief() -> None:
    config = ExAITrainingConfig()

    assert config.model_name == "bert-base-uncased"
    assert config.max_length == 128
    assert config.batch_size == 8
    assert config.learning_rate == 2e-5
