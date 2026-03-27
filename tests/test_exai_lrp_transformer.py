from pathlib import Path

import torch
from exai_test_utils import (
    fake_model_loader,
    fake_tokenizer_loader,
    write_training_dataset_fixture,
)

from app.exai.config import ExAIDataConfig, ExAIPaths, ExAITrainingConfig
from app.exai.inference import ExAIInferenceRunner
from app.exai.lrp_transformer import TransformerLRPExplainer
from app.exai.train import train_exai_classifier


def test_transformer_lrp_returns_deterministic_finite_scores(tmp_path: Path) -> None:
    dataset_dir = write_training_dataset_fixture(tmp_path)
    output_paths = ExAIPaths(root=tmp_path / "outputs" / "exai")
    training_result = train_exai_classifier(
        data_config=ExAIDataConfig(
            dataset_path=dataset_dir,
            split_seed=6,
            train_fraction=0.5,
            validation_fraction=0.25,
            output_paths=output_paths,
        ),
        training_config=ExAITrainingConfig(
            batch_size=4,
            learning_rate=1e-2,
            epochs=1,
            early_stopping_patience=1,
            seed=6,
            device="cpu",
            output_paths=output_paths,
        ),
        tokenizer_loader=fake_tokenizer_loader,
        model_loader=fake_model_loader,
    )
    runner = ExAIInferenceRunner(
        training_result.checkpoint_dir,
        max_length=16,
        device="cpu",
        tokenizer_loader=fake_tokenizer_loader,
        model_loader=fake_model_loader,
    )
    explainer = TransformerLRPExplainer(runner)

    first = explainer.explain_text("The doctor was rude.")
    second = explainer.explain_text("The doctor was rude.")

    assert first.token_relevance.shape == second.token_relevance.shape
    assert torch.allclose(first.token_relevance, second.token_relevance)
    assert torch.isfinite(first.token_relevance).all()
    assert "residual connections" in first.approximation_note
