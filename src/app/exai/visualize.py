from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .schemas import InferenceResult

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def merge_wordpieces(
    tokens: Sequence[str],
    scores: Sequence[float],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for index, (token, score) in enumerate(zip(tokens, scores, strict=False)):
        if token in SPECIAL_TOKENS:
            continue
        if token.startswith("##") and merged:
            merged[-1]["text"] = f"{merged[-1]['text']}{token[2:]}"
            merged[-1]["score"] += float(score)
            merged[-1]["token_indices"].append(index)
            continue
        merged.append(
            {
                "text": token,
                "score": float(score),
                "token_indices": [index],
            }
        )
    return merged


def build_explanation_payload(
    *,
    inference_result: InferenceResult,
    token_relevance: torch.Tensor,
    method_note: str,
    benchmark_id: str | None = None,
) -> dict[str, Any]:
    confidence = float(inference_result.probabilities[inference_result.predicted_label_id].item())
    merged_tokens = merge_wordpieces(
        inference_result.tokens,
        token_relevance.detach().cpu().tolist(),
    )
    return {
        "benchmark_id": benchmark_id,
        "text": inference_result.text,
        "predicted_label": inference_result.predicted_label,
        "target_label": inference_result.target_label,
        "confidence": confidence,
        "tokens": [
            {
                "token": token,
                "score": float(score),
            }
            for token, score in zip(
                inference_result.tokens,
                token_relevance.detach().cpu().tolist(),
                strict=False,
            )
        ],
        "merged_tokens": merged_tokens,
        "method_note": method_note,
        "ethics_notice": (
            "Texts may contain offensive content. Keep excerpts minimal and do not publish raw "
            "dumps."
        ),
    }
