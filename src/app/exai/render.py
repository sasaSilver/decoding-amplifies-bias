from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from .inference import ExAIInferenceRunner
from .lrp_transformer import TransformerLRPExplainer
from .utils import canonical_json_digest, write_json
from .visualize import build_explanation_payload


def _score_to_style(score: float, max_abs_score: float) -> str:
    if max_abs_score <= 0:
        return "background: rgba(200, 200, 200, 0.15);"
    intensity = min(abs(score) / max_abs_score, 1.0)
    if score >= 0:
        return f"background: rgba(214, 63, 46, {0.15 + 0.55 * intensity:.3f});"
    return f"background: rgba(33, 100, 171, {0.15 + 0.55 * intensity:.3f});"


def render_token_heatmap_html(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_tokens = payload["merged_tokens"]
    max_abs_score = max((abs(token["score"]) for token in merged_tokens), default=0.0)
    token_markup = " ".join(
        (
            f"<span class='token' style='{_score_to_style(token['score'], max_abs_score)}'>"
            f"{token['text']}</span>"
        )
        for token in merged_tokens
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ExAI Explanation</title>
  <style>
    body {{ font-family: 'IBM Plex Sans', sans-serif; margin: 2rem auto; max-width: 900px; }}
    .meta {{ color: #4a4a4a; font-size: 0.95rem; margin-bottom: 1rem; }}
    .notice {{ background: #f6efe0; padding: 0.75rem 1rem; border-left: 4px solid #a87000; }}
    .tokens {{ line-height: 2.1; margin-top: 1.5rem; }}
    .token {{ border-radius: 0.35rem; padding: 0.15rem 0.3rem; }}
    .note {{ margin-top: 1.5rem; color: #5a5a5a; }}
  </style>
</head>
<body>
  <h1>ExAI Token Relevance</h1>
  <div class="meta">
    <strong>Predicted label:</strong> {payload["predicted_label"]} |
    <strong>Target:</strong> {payload["target_label"]} |
    <strong>Confidence:</strong> {payload["confidence"]:.3f}
  </div>
  <div class="notice">{payload["ethics_notice"]}</div>
  <p class="tokens">{token_markup}</p>
  <p class="note">{payload["method_note"]}</p>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_benchmark_explanations(
    *,
    checkpoint_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    benchmark_ids: list[str] | None = None,
    max_examples: int = 5,
    device: str = "auto",
    max_length: int = 128,
    tokenizer_loader: Any | None = None,
    model_loader: Any | None = None,
) -> list[Path]:
    benchmark_df = pd.read_parquet(benchmark_path)
    if benchmark_ids is not None:
        benchmark_df = benchmark_df[benchmark_df["benchmark_id"].isin(benchmark_ids)]
    benchmark_df = benchmark_df.head(max_examples)

    runner = ExAIInferenceRunner(
        checkpoint_path,
        device=device,
        max_length=max_length,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    explainer = TransformerLRPExplainer(runner)
    rendered_paths: list[Path] = []

    benchmark_columns = list(benchmark_df.columns)
    benchmark_records = [
        dict(zip(benchmark_columns, values, strict=False))
        for values in benchmark_df.itertuples(index=False, name=None)
    ]
    benchmark_records = cast(list[dict[str, Any]], benchmark_records)
    for row in benchmark_records:
        inference_result = runner.predict_text(
            str(row["completion_text"]),
            target_label=str(row["predicted_label"]),
        )
        explanation = explainer.explain_inference(inference_result)
        payload = build_explanation_payload(
            inference_result=inference_result,
            token_relevance=explanation.token_relevance,
            method_note=explanation.approximation_note,
            benchmark_id=str(row["benchmark_id"]),
        )
        json_path = output_dir / f"explanation_{row['benchmark_id']}.json"
        html_path = output_dir / f"explanation_{row['benchmark_id']}.html"
        write_json(json_path, payload)
        render_token_heatmap_html(payload, html_path)
        rendered_paths.append(html_path)

    return rendered_paths


def render_text_explanation(
    *,
    checkpoint_path: Path,
    text: str,
    output_dir: Path,
    target_label: str | None = None,
    device: str = "auto",
    max_length: int = 128,
    tokenizer_loader: Any | None = None,
    model_loader: Any | None = None,
) -> dict[str, Path]:
    runner = ExAIInferenceRunner(
        checkpoint_path,
        device=device,
        max_length=max_length,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    explainer = TransformerLRPExplainer(runner)
    inference_result = runner.predict_text(text, target_label=target_label)
    explanation = explainer.explain_inference(inference_result)
    payload = build_explanation_payload(
        inference_result=inference_result,
        token_relevance=explanation.token_relevance,
        method_note=explanation.approximation_note,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    key = canonical_json_digest(
        {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "text": text,
            "target_label": target_label,
        }
    )[:20]
    json_path = output_dir / f"text_explanation_{key}.json"
    html_path = output_dir / f"text_explanation_{key}.html"
    write_json(json_path, payload)
    render_token_heatmap_html(payload, html_path)
    return {
        "json": json_path,
        "html": html_path,
    }
