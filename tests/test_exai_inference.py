from pathlib import Path

import pytest
from exai_test_utils import (
    fake_model_loader,
    fake_tokenizer_loader,
    write_training_dataset_fixture,
)

from app.exai.config import ExAIDataConfig, ExAIPaths, ExAITrainingConfig
from app.exai.inference import ExAIInferenceRunner
from app.exai.train import train_exai_classifier


def test_exai_inference_returns_stable_tokens_and_logits(tmp_path: Path) -> None:
    dataset_dir = write_training_dataset_fixture(tmp_path)
    output_paths = ExAIPaths(root=tmp_path / "outputs" / "exai")
    result = train_exai_classifier(
        data_config=ExAIDataConfig(
            dataset_path=dataset_dir,
            split_seed=2,
            train_fraction=0.5,
            validation_fraction=0.25,
            output_paths=output_paths,
        ),
        training_config=ExAITrainingConfig(
            batch_size=4,
            learning_rate=1e-2,
            epochs=1,
            early_stopping_patience=1,
            seed=2,
            device="cpu",
            output_paths=output_paths,
        ),
        tokenizer_loader=fake_tokenizer_loader,
        model_loader=fake_model_loader,
    )

    runner = ExAIInferenceRunner(
        result.checkpoint_dir,
        max_length=16,
        device="cpu",
        tokenizer_loader=fake_tokenizer_loader,
        model_loader=fake_model_loader,
    )
    inference = runner.predict_text("The nurse helped the patient.", target_label="negative")

    assert len(inference.tokens) == len(inference.token_ids) == len(inference.attention_mask)
    assert inference.logits.shape[0] == 4
    assert torch_sum(inference.probabilities.tolist()) == pytest.approx(1.0)
    assert inference.target_label == "negative"
    assert inference.target_label_id == inference.negative_label_id
    assert inference.hidden_states


def torch_sum(values: list[float]) -> float:
    return sum(values)
