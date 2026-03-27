from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd

from .inference import ExAIInferenceRunner
from .lrp_transformer import TransformerLRPExplainer
from .utils import write_json
from .visualize import merge_wordpieces


def build_perturbations(text: str) -> dict[str, str]:
    return {
        "punctuation": text if text.endswith("!") else f"{text}!",
        "neutral_insertion": f"Overall, {text}",
        "benign_rephrase": text.replace(" was ", " seemed "),
    }


def _top_k_terms(
    *,
    runner: ExAIInferenceRunner,
    explainer: TransformerLRPExplainer,
    text: str,
    target_label: str | None,
    top_k: int,
) -> list[str]:
    inference = runner.predict_text(text, target_label=target_label)
    explanation = explainer.explain_inference(inference)
    scores = (
        explanation.token_relevance.tolist()
        if hasattr(explanation.token_relevance, "tolist")
        else list(explanation.token_relevance)
    )
    merged = merge_wordpieces(inference.tokens, [float(score) for score in scores])
    ranked = sorted(merged, key=lambda span: abs(span["score"]), reverse=True)
    return [span["text"] for span in ranked[:top_k]]


def evaluate_sensitivity_case(
    *,
    runner: ExAIInferenceRunner,
    explainer: TransformerLRPExplainer,
    text: str,
    target_label: str | None,
    top_k: int = 3,
) -> dict[str, Any]:
    reference_terms = set(
        _top_k_terms(
            runner=runner,
            explainer=explainer,
            text=text,
            target_label=target_label,
            top_k=top_k,
        )
    )
    perturbations = build_perturbations(text)
    overlaps: dict[str, float] = {}
    for perturbation_name, perturbed_text in perturbations.items():
        perturbed_terms = set(
            _top_k_terms(
                runner=runner,
                explainer=explainer,
                text=perturbed_text,
                target_label=target_label,
                top_k=top_k,
            )
        )
        denominator = max(len(reference_terms | perturbed_terms), 1)
        overlaps[perturbation_name] = len(reference_terms & perturbed_terms) / denominator
    return {
        "text": text,
        "overlaps": overlaps,
    }


def run_sensitivity_benchmark(
    *,
    runner: ExAIInferenceRunner,
    explainer: TransformerLRPExplainer,
    benchmark_path: Path,
    output_dir: Path,
    top_k: int = 3,
) -> dict[str, Path]:
    benchmark_df = pd.read_parquet(benchmark_path)
    benchmark_records = cast(list[dict[str, Any]], benchmark_df.to_dict(orient="records"))
    cases = [
        evaluate_sensitivity_case(
            runner=runner,
            explainer=explainer,
            text=str(row["completion_text"]),
            target_label=str(row["predicted_label"]),
            top_k=top_k,
        )
        for row in benchmark_records
    ]
    perturbation_names = ["punctuation", "neutral_insertion", "benign_rephrase"]
    summary = {
        name: mean(case["overlaps"][name] for case in cases) if cases else 0.0
        for name in perturbation_names
    }
    payload = {
        "case_count": len(cases),
        "top_k": top_k,
        "summary": summary,
        "cases": cases[:10],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "sensitivity_metrics.json"
    plot_path = output_dir / "sensitivity_plot.png"
    write_json(metrics_path, payload)

    plt.figure(figsize=(5, 3))
    plt.bar(list(summary.keys()), list(summary.values()), color=["#4275a3", "#d38f29", "#648b4f"])
    plt.ylabel("Average top-k overlap")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return {
        "metrics": metrics_path,
        "plot": plot_path,
    }
