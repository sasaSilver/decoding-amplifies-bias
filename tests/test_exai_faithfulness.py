from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from app.exai.faithfulness import evaluate_faithfulness_case, run_faithfulness_benchmark


class DummyTokenizer:
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(token for token in tokens if token not in {"[CLS]", "[SEP]"})


class DummyRunner:
    def __init__(self) -> None:
        self.bundle = SimpleNamespace(tokenizer=DummyTokenizer())

    def predict_text(self, text: str, *, target_label: str | None = None) -> SimpleNamespace:
        probability = 0.9
        if "alpha" not in text:
            probability = 0.2
        elif "gamma" not in text:
            probability = 0.8
        elif "beta" not in text:
            probability = 0.7
        return SimpleNamespace(
            text=text,
            tokens=["[CLS]", "alpha", "beta", "gamma", "[SEP]"],
            target_label=target_label or "negative",
            target_label_id=0,
            probabilities=torch.tensor([probability, 1.0 - probability, 0.0, 0.0]),
        )


class DummyExplainer:
    def explain_inference(self, _: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(token_relevance=torch.tensor([0.0, 0.9, 0.2, 0.05, 0.0]))


def test_evaluate_faithfulness_case_prefers_top_relevance_removal() -> None:
    result = evaluate_faithfulness_case(
        runner=DummyRunner(),  # type: ignore[arg-type]
        explainer=DummyExplainer(),  # type: ignore[arg-type]
        text="alpha beta gamma",
        target_label="negative",
        removal_count=1,
        random_seed=4,
    )

    assert float(result["top_drop"]) > float(result["least_drop"])


def test_run_faithfulness_benchmark_writes_metrics_and_plot(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.parquet"
    pd.DataFrame(
        [
            {"completion_text": "alpha beta gamma", "predicted_label": "negative"},
            {"completion_text": "alpha beta gamma", "predicted_label": "negative"},
        ]
    ).to_parquet(benchmark_path, index=False)

    artifacts = run_faithfulness_benchmark(
        runner=DummyRunner(),  # type: ignore[arg-type]
        explainer=DummyExplainer(),  # type: ignore[arg-type]
        benchmark_path=benchmark_path,
        output_dir=tmp_path / "faithfulness",
    )

    assert artifacts["metrics"].exists()
    assert artifacts["plot"].exists()
