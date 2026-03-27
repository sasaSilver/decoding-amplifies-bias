import json
from pathlib import Path

from exai_test_utils import (
    fake_model_loader,
    fake_tokenizer_loader,
    write_benchmark_score_artifacts,
    write_training_dataset_fixture,
)

from app.exai.benchmark import build_explanation_benchmark
from app.exai.config import (
    ExAIBenchmarkConfig,
    ExAIDataConfig,
    ExAIEvalConfig,
    ExAIPaths,
    ExAITrainingConfig,
)
from app.exai.evaluate import evaluate_exai_classifier
from app.exai.train import train_exai_classifier


class FakeReleasedBackend:
    def predict_batch(self, texts: list[str]) -> list[str]:
        predictions = []
        for text in texts:
            lowered = text.lower()
            if "helped" in lowered:
                predictions.append("positive")
            elif "rude" in lowered:
                predictions.append("negative")
            elif "waited" in lowered:
                predictions.append("other")
            else:
                predictions.append("neutral")
        return predictions


def test_evaluate_exai_classifier_writes_metrics_and_agreement(tmp_path: Path) -> None:
    dataset_dir = write_training_dataset_fixture(tmp_path)
    combined_manifest = write_benchmark_score_artifacts(tmp_path)
    output_paths = ExAIPaths(root=tmp_path / "outputs" / "exai")
    data_config = ExAIDataConfig(
        dataset_path=dataset_dir,
        split_seed=5,
        train_fraction=0.5,
        validation_fraction=0.25,
        use_masking=True,
        output_paths=output_paths,
    )
    training_result = train_exai_classifier(
        data_config=data_config,
        training_config=ExAITrainingConfig(
            batch_size=4,
            learning_rate=1e-2,
            epochs=2,
            early_stopping_patience=2,
            seed=5,
            device="cpu",
            output_paths=output_paths,
        ),
        tokenizer_loader=fake_tokenizer_loader,
        model_loader=fake_model_loader,
    )
    benchmark_result = build_explanation_benchmark(
        ExAIBenchmarkConfig(
            repo_root=tmp_path,
            source_manifest_path=combined_manifest,
            examples_per_label=1,
            selection_seed=5,
            output_paths=output_paths,
        )
    )

    artifact_paths = evaluate_exai_classifier(
        data_config=data_config,
        eval_config=ExAIEvalConfig(
            checkpoint_path=training_result.checkpoint_dir,
            benchmark_path=benchmark_result.benchmark_path,
            batch_size=4,
            max_length=16,
            device="cpu",
            output_paths=output_paths,
        ),
        benchmark_config=None,
        released_backend=FakeReleasedBackend(),
        tokenizer_loader=fake_tokenizer_loader,
        model_loader=fake_model_loader,
    )

    for path in artifact_paths.values():
        assert path.exists()
    test_metrics = json.loads(artifact_paths["test_metrics"].read_text(encoding="utf-8"))
    assert "accuracy" in test_metrics
    benchmark_metrics = json.loads(artifact_paths["benchmark_metrics"].read_text(encoding="utf-8"))
    assert "macro_f1" in benchmark_metrics
    agreement = json.loads(artifact_paths["agreement_metrics"].read_text(encoding="utf-8"))
    assert "agreement" in agreement
