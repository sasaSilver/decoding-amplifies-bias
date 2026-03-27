from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app.exai.sensitivity import evaluate_sensitivity_case, run_sensitivity_benchmark


class DummyRunner:
    def predict_text(self, text: str, *, target_label: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(text=text, target_label=target_label or "negative")


class DummyExplainer:
    def explain_inference(self, inference: SimpleNamespace) -> SimpleNamespace:
        if "Overall" in inference.text:
            scores = [0.0, 0.7, 0.6, 0.05, 0.0]
        else:
            scores = [0.0, 0.8, 0.6, 0.1, 0.0]
        return SimpleNamespace(token_relevance=scores)


def _predict_text(self, text: str, *, target_label: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        target_label=target_label or "negative",
        tokens=["[CLS]", "alpha", "beta", "gamma", "[SEP]"],
    )


DummyRunner.predict_text = _predict_text  # type: ignore[method-assign]


def test_evaluate_sensitivity_case_returns_overlap_metrics() -> None:
    result = evaluate_sensitivity_case(
        runner=DummyRunner(),  # type: ignore[arg-type]
        explainer=DummyExplainer(),  # type: ignore[arg-type]
        text="alpha beta gamma",
        target_label="negative",
        top_k=2,
    )

    assert set(result["overlaps"]) == {"punctuation", "neutral_insertion", "benign_rephrase"}


def test_run_sensitivity_benchmark_writes_metrics_and_plot(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.parquet"
    pd.DataFrame(
        [
            {"completion_text": "alpha beta gamma", "predicted_label": "negative"},
            {"completion_text": "alpha beta gamma", "predicted_label": "negative"},
        ]
    ).to_parquet(benchmark_path, index=False)

    artifacts = run_sensitivity_benchmark(
        runner=DummyRunner(),  # type: ignore[arg-type]
        explainer=DummyExplainer(),  # type: ignore[arg-type]
        benchmark_path=benchmark_path,
        output_dir=tmp_path / "sensitivity",
        top_k=2,
    )

    assert artifacts["metrics"].exists()
    assert artifacts["plot"].exists()
