from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.scoring import NLGBiasClassifier

from .benchmark import build_explanation_benchmark
from .compare import compare_with_released_scorer, compute_agreement_metrics
from .config import ExAIBenchmarkConfig, ExAIDataConfig, ExAIEvalConfig
from .eval import (
    build_error_analysis,
    evaluate_benchmark_predictions,
    evaluate_records,
)
from .splits import prepare_regard_dataset
from .utils import file_digest, utc_now_iso, write_json


def evaluate_exai_classifier(
    *,
    data_config: ExAIDataConfig,
    eval_config: ExAIEvalConfig,
    benchmark_config: ExAIBenchmarkConfig | None = None,
    released_backend: NLGBiasClassifier | Any | None = None,
    tokenizer_loader: Any | None = None,
    model_loader: Any | None = None,
) -> dict[str, Path]:
    paths = eval_config.output_paths.ensure_dirs()
    dataset = prepare_regard_dataset(data_config)
    benchmark_result = (
        build_explanation_benchmark(benchmark_config) if benchmark_config is not None else None
    )
    benchmark_path = (
        benchmark_result.benchmark_path
        if benchmark_result is not None
        else eval_config.benchmark_path
    )
    if benchmark_path is None:
        benchmark_path = build_explanation_benchmark(
            ExAIBenchmarkConfig(
                output_paths=eval_config.output_paths, repo_root=data_config.repo_root
            )
        ).benchmark_path

    checkpoint_digest = (
        file_digest(eval_config.checkpoint_path / "training_metrics.json")
        if (eval_config.checkpoint_path / "training_metrics.json").exists()
        else file_digest(eval_config.checkpoint_path / "training_manifest.json")
    )
    eval_key = checkpoint_digest[:20]
    test_metrics_path = paths.eval_dir / f"eval_{eval_key}_test_metrics.json"
    benchmark_metrics_path = paths.eval_dir / f"eval_{eval_key}_benchmark_metrics.json"
    agreement_path = paths.eval_dir / f"eval_{eval_key}_agreement.json"
    error_analysis_path = paths.eval_dir / f"eval_{eval_key}_error_analysis.json"

    test_metrics = evaluate_records(
        checkpoint_path=eval_config.checkpoint_path,
        records=dataset.splits["test"],
        batch_size=eval_config.batch_size,
        max_length=eval_config.max_length,
        device=eval_config.device,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    write_json(test_metrics_path, test_metrics)

    benchmark_metrics = evaluate_benchmark_predictions(
        checkpoint_path=eval_config.checkpoint_path,
        benchmark_path=benchmark_path,
        batch_size=eval_config.batch_size,
        max_length=eval_config.max_length,
        device=eval_config.device,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    write_json(benchmark_metrics_path, benchmark_metrics)

    benchmark_df = pd.read_parquet(benchmark_path)
    agreement_payload = compute_agreement_metrics(
        reference_labels=benchmark_df["predicted_label"].tolist(),
        comparison_labels=benchmark_metrics["predicted_labels"],
    )
    if released_backend is not None:
        released_comparison = compare_with_released_scorer(
            texts=[record.active_text for record in dataset.splits["test"]],
            model_predictions=test_metrics["predicted_labels"],
            released_backend=released_backend,
        )
        agreement_payload["released_test_split"] = released_comparison
    agreement_payload["created_at_utc"] = utc_now_iso()
    write_json(agreement_path, agreement_payload)

    error_analysis = build_error_analysis(
        benchmark_path=benchmark_path,
        predicted_labels=benchmark_metrics["predicted_labels"],
    )
    write_json(error_analysis_path, error_analysis)

    return {
        "test_metrics": test_metrics_path,
        "benchmark_metrics": benchmark_metrics_path,
        "agreement_metrics": agreement_path,
        "error_analysis": error_analysis_path,
    }
