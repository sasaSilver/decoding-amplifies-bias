from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd

from .inference import ExAIInferenceRunner
from .lrp_transformer import TransformerLRPExplainer
from .utils import runner_checkpoint_path, runner_device_name, utc_now_iso, write_json
from .visualize import SPECIAL_TOKENS, merge_wordpieces


def _select_random_spans(
    spans: list[dict[str, Any]],
    *,
    seed: int,
    count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        spans,
        key=lambda span: sha256(
            f"{seed}:{span['text']}:{span['token_indices']}".encode()
        ).hexdigest(),
    )
    return ordered[:count]


def reconstruct_text_without_indices(
    *,
    tokenizer: Any,
    tokens: Sequence[str],
    removed_indices: set[int],
) -> str:
    remaining_tokens = [
        token
        for index, token in enumerate(tokens)
        if index not in removed_indices and token not in SPECIAL_TOKENS
    ]
    if hasattr(tokenizer, "convert_tokens_to_string"):
        return tokenizer.convert_tokens_to_string(remaining_tokens)
    return " ".join(token.replace("##", "") for token in remaining_tokens)


def evaluate_faithfulness_case(
    *,
    runner: ExAIInferenceRunner,
    explainer: TransformerLRPExplainer,
    text: str,
    target_label: str | None = None,
    removal_count: int = 1,
    random_seed: int = 13,
) -> dict[str, Any]:
    baseline = runner.predict_text(text, target_label=target_label)
    explanation = explainer.explain_inference(baseline)
    spans = merge_wordpieces(baseline.tokens, explanation.token_relevance.tolist())
    if not spans:
        return {
            "text": text,
            "baseline_probability": float(baseline.probabilities[baseline.target_label_id].item()),
            "top_drop": 0.0,
            "least_drop": 0.0,
            "random_drop": 0.0,
        }

    span_count = min(removal_count, len(spans))
    top_spans = sorted(spans, key=lambda span: abs(span["score"]), reverse=True)[:span_count]
    least_spans = sorted(spans, key=lambda span: abs(span["score"]))[:span_count]
    random_spans = _select_random_spans(spans, seed=random_seed, count=span_count)

    def drop_for(selected_spans: list[dict[str, Any]]) -> float:
        removed_indices = {
            token_index for span in selected_spans for token_index in span["token_indices"]
        }
        modified_text = reconstruct_text_without_indices(
            tokenizer=runner.bundle.tokenizer,
            tokens=baseline.tokens,
            removed_indices=removed_indices,
        )
        modified = runner.predict_text(modified_text, target_label=baseline.target_label)
        return float(
            baseline.probabilities[baseline.target_label_id].item()
            - modified.probabilities[modified.target_label_id].item()
        )

    return {
        "text": text,
        "baseline_probability": float(baseline.probabilities[baseline.target_label_id].item()),
        "top_drop": drop_for(top_spans),
        "least_drop": drop_for(least_spans),
        "random_drop": drop_for(random_spans),
    }


def run_faithfulness_benchmark(
    *,
    runner: ExAIInferenceRunner,
    explainer: TransformerLRPExplainer,
    benchmark_path: Path,
    output_dir: Path,
    removal_count: int = 1,
    random_seed: int = 13,
) -> dict[str, Path]:
    benchmark_df = pd.read_parquet(benchmark_path)
    benchmark_records = cast(list[dict[str, Any]], benchmark_df.to_dict(orient="records"))
    cases = [
        evaluate_faithfulness_case(
            runner=runner,
            explainer=explainer,
            text=str(row["completion_text"]),
            target_label=str(row["predicted_label"]),
            removal_count=removal_count,
            random_seed=random_seed,
        )
        for row in benchmark_records
    ]
    metrics_payload = {
        "created_at_utc": utc_now_iso(),
        "benchmark_path": str(benchmark_path.resolve()),
        "checkpoint_path": runner_checkpoint_path(runner),
        "device": runner_device_name(runner),
        "case_count": len(cases),
        "removal_count": removal_count,
        "random_seed": random_seed,
        "top_drop_mean": mean(float(case["top_drop"]) for case in cases) if cases else 0.0,
        "least_drop_mean": mean(float(case["least_drop"]) for case in cases) if cases else 0.0,
        "random_drop_mean": mean(float(case["random_drop"]) for case in cases) if cases else 0.0,
        "cases": cases[:10],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "faithfulness_metrics.json"
    plot_path = output_dir / "faithfulness_plot.png"
    write_json(metrics_path, metrics_payload)

    plt.figure(figsize=(5, 3))
    plt.bar(
        ["top", "random", "least"],
        [
            metrics_payload["top_drop_mean"],
            metrics_payload["random_drop_mean"],
            metrics_payload["least_drop_mean"],
        ],
        color=["#c9402a", "#6b8ba4", "#9aa9b5"],
    )
    plt.ylabel("Average target-probability drop")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return {
        "metrics": metrics_path,
        "plot": plot_path,
    }
