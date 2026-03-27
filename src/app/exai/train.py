from __future__ import annotations

from typing import Any

from .config import ExAIDataConfig, ExAITrainingConfig
from .splits import prepare_regard_dataset
from .trainer import TrainingRunResult, train_classifier


def train_exai_classifier(
    *,
    data_config: ExAIDataConfig | None = None,
    training_config: ExAITrainingConfig | None = None,
    tokenizer_loader: Any | None = None,
    model_loader: Any | None = None,
) -> TrainingRunResult:
    resolved_data_config = data_config or ExAIDataConfig()
    resolved_training_config = training_config or ExAITrainingConfig(
        output_paths=resolved_data_config.output_paths
    )
    dataset = prepare_regard_dataset(resolved_data_config)
    return train_classifier(
        train_records=dataset.splits["train"],
        validation_records=dataset.splits["validation"],
        split_manifest_path=dataset.split_manifest_path,
        config=resolved_training_config,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
