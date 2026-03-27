from pathlib import Path

import torch
from exai_test_utils import (
    fake_model_loader,
    fake_tokenizer_loader,
    write_training_dataset_fixture,
)

from app.exai.config import ExAIDataConfig, ExAIPaths, ExAITrainingConfig
from app.exai.inference import ExAIInferenceRunner
from app.exai.lrp_core import epsilon_lrp_linear
from app.exai.lrp_linear import LinearLRPExplainer
from app.exai.train import train_exai_classifier


def test_epsilon_lrp_linear_returns_finite_input_relevance() -> None:
    inputs = torch.tensor([0.4, -0.2, 0.8], dtype=torch.float32)
    weight = torch.tensor(
        [
            [0.3, 0.5, -0.4],
            [0.2, -0.1, 0.6],
        ],
        dtype=torch.float32,
    )
    relevance = torch.tensor([0.9, 0.1], dtype=torch.float32)

    propagated = epsilon_lrp_linear(
        inputs=inputs,
        weight=weight,
        relevance=relevance,
        epsilon=1e-5,
    )

    assert propagated.shape == inputs.shape
    assert torch.isfinite(propagated).all()


def test_linear_lrp_explainer_returns_token_scores_for_classifier_head(tmp_path: Path) -> None:
    dataset_dir = write_training_dataset_fixture(tmp_path)
    output_paths = ExAIPaths(root=tmp_path / "outputs" / "exai")
    training_result = train_exai_classifier(
        data_config=ExAIDataConfig(
            dataset_path=dataset_dir,
            split_seed=4,
            train_fraction=0.5,
            validation_fraction=0.25,
            output_paths=output_paths,
        ),
        training_config=ExAITrainingConfig(
            batch_size=4,
            learning_rate=1e-2,
            epochs=1,
            early_stopping_patience=1,
            seed=4,
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

    explanation = LinearLRPExplainer(runner).explain_text(
        "The nurse helped the patient.",
        target_label="negative",
    )

    assert explanation.token_relevance.ndim == 1
    assert explanation.token_relevance.shape[0] == len(
        runner.predict_text("The nurse helped the patient.").tokens
    )
    assert torch.isfinite(explanation.token_relevance).all()
