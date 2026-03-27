from __future__ import annotations

from pathlib import Path

from .benchmark import build_explanation_benchmark
from .config import (
    ExAIBenchmarkConfig,
    ExAIDataConfig,
    ExAIEvalConfig,
    ExAIPaths,
    ExAITrainingConfig,
)
from .evaluate import evaluate_exai_classifier
from .faithfulness import run_faithfulness_benchmark
from .inference import ExAIInferenceRunner
from .lrp_transformer import TransformerLRPExplainer
from .render import render_benchmark_explanations, render_text_explanation
from .sensitivity import run_sensitivity_benchmark
from .train import train_exai_classifier


def _paths(output_root: Path | None) -> ExAIPaths:
    if output_root is None:
        return ExAIPaths().ensure_dirs()
    return ExAIPaths(root=output_root).ensure_dirs()


def build_exai_benchmark_cmd(
    *,
    output_root: Path | None = None,
    repo_root: Path | None = None,
    source_manifest_path: Path | None = None,
    examples_per_label: int = 3,
    selection_seed: int = 13,
) -> dict[str, Path]:
    paths = _paths(output_root)
    result = build_explanation_benchmark(
        ExAIBenchmarkConfig(
            repo_root=repo_root or paths.repo_root,
            source_manifest_path=source_manifest_path,
            examples_per_label=examples_per_label,
            selection_seed=selection_seed,
            output_paths=paths,
        )
    )
    return {
        "benchmark": result.benchmark_path,
        "manifest": result.manifest_path,
    }


def train_exai_classifier_cmd(
    *,
    dataset_path: Path,
    output_root: Path | None = None,
    split_seed: int = 13,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    use_masking: bool = True,
    model_name: str = "bert-base-uncased",
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    early_stopping_patience: int = 2,
    seed: int = 13,
    device: str = "auto",
) -> dict[str, Path]:
    paths = _paths(output_root)
    result = train_exai_classifier(
        data_config=ExAIDataConfig(
            dataset_path=dataset_path,
            split_seed=split_seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            use_masking=use_masking,
            output_paths=paths,
        ),
        training_config=ExAITrainingConfig(
            model_name=model_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            seed=seed,
            device=device,
            output_paths=paths,
        ),
    )
    return {
        "checkpoint_dir": result.checkpoint_dir,
        "manifest": result.manifest_path,
        "metrics": result.metrics_path,
    }


def eval_exai_classifier_cmd(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    benchmark_path: Path | None = None,
    output_root: Path | None = None,
    split_seed: int = 13,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    use_masking: bool = True,
    batch_size: int = 8,
    max_length: int = 128,
    device: str = "auto",
) -> dict[str, Path]:
    paths = _paths(output_root)
    return evaluate_exai_classifier(
        data_config=ExAIDataConfig(
            dataset_path=dataset_path,
            split_seed=split_seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            use_masking=use_masking,
            output_paths=paths,
        ),
        eval_config=ExAIEvalConfig(
            checkpoint_path=checkpoint_path,
            benchmark_path=benchmark_path,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            output_paths=paths,
        ),
    )


def explain_text_cmd(
    *,
    checkpoint_path: Path,
    text: str,
    output_root: Path | None = None,
    target_label: str | None = None,
    device: str = "auto",
    max_length: int = 128,
) -> dict[str, Path]:
    paths = _paths(output_root)
    return render_text_explanation(
        checkpoint_path=checkpoint_path,
        text=text,
        output_dir=paths.explanations_dir,
        target_label=target_label,
        device=device,
        max_length=max_length,
    )


def explain_benchmark_cmd(
    *,
    checkpoint_path: Path,
    benchmark_path: Path,
    output_root: Path | None = None,
    max_examples: int = 5,
    device: str = "auto",
    max_length: int = 128,
) -> list[Path]:
    paths = _paths(output_root)
    return render_benchmark_explanations(
        checkpoint_path=checkpoint_path,
        benchmark_path=benchmark_path,
        output_dir=paths.explanations_dir,
        max_examples=max_examples,
        device=device,
        max_length=max_length,
    )


def exai_faithfulness_cmd(
    *,
    checkpoint_path: Path,
    benchmark_path: Path,
    output_root: Path | None = None,
    removal_count: int = 1,
    random_seed: int = 13,
    device: str = "auto",
    max_length: int = 128,
) -> dict[str, Path]:
    paths = _paths(output_root)
    runner = ExAIInferenceRunner(checkpoint_path, device=device, max_length=max_length)
    explainer = TransformerLRPExplainer(runner)
    return run_faithfulness_benchmark(
        runner=runner,
        explainer=explainer,
        benchmark_path=benchmark_path,
        output_dir=paths.reports_dir / "faithfulness",
        removal_count=removal_count,
        random_seed=random_seed,
    )


def exai_sensitivity_cmd(
    *,
    checkpoint_path: Path,
    benchmark_path: Path,
    output_root: Path | None = None,
    top_k: int = 3,
    device: str = "auto",
    max_length: int = 128,
) -> dict[str, Path]:
    paths = _paths(output_root)
    runner = ExAIInferenceRunner(checkpoint_path, device=device, max_length=max_length)
    explainer = TransformerLRPExplainer(runner)
    return run_sensitivity_benchmark(
        runner=runner,
        explainer=explainer,
        benchmark_path=benchmark_path,
        output_dir=paths.reports_dir / "sensitivity",
        top_k=top_k,
    )
