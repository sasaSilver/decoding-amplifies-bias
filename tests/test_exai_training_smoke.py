import json
from pathlib import Path

from exai_test_utils import (
    fake_model_loader,
    fake_tokenizer_loader,
    write_training_dataset_fixture,
)

from app.exai.config import ExAIDataConfig, ExAIPaths, ExAITrainingConfig
from app.exai.modeling import load_classifier_bundle
from app.exai.train import train_exai_classifier


def test_train_exai_classifier_smoke(tmp_path: Path) -> None:
    dataset_dir = write_training_dataset_fixture(tmp_path)
    output_paths = ExAIPaths(root=tmp_path / "outputs" / "exai")
    data_config = ExAIDataConfig(
        dataset_path=dataset_dir,
        split_seed=3,
        train_fraction=0.5,
        validation_fraction=0.25,
        use_masking=True,
        output_paths=output_paths,
    )
    training_config = ExAITrainingConfig(
        batch_size=4,
        learning_rate=1e-2,
        epochs=2,
        early_stopping_patience=2,
        seed=7,
        device="cpu",
        output_paths=output_paths,
    )

    result = train_exai_classifier(
        data_config=data_config,
        training_config=training_config,
        tokenizer_loader=fake_tokenizer_loader,
        model_loader=fake_model_loader,
    )

    assert result.checkpoint_dir.exists()
    assert result.manifest_path.exists()
    assert result.metrics_path.exists()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["best_epoch"] >= 1
    bundle = load_classifier_bundle(
        result.checkpoint_dir,
        tokenizer_loader=fake_tokenizer_loader,
        model_loader=fake_model_loader,
    )
    assert bundle.model is not None
