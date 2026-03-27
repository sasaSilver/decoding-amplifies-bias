from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.scoring import NLGBiasClassifier

from .eval import compute_classification_metrics


def compute_agreement_metrics(
    reference_labels: Sequence[str],
    comparison_labels: Sequence[str],
) -> dict[str, Any]:
    metrics = compute_classification_metrics(reference_labels, comparison_labels)
    agreement = sum(
        1
        for reference, comparison in zip(reference_labels, comparison_labels, strict=False)
        if reference == comparison
    )
    metrics["agreement"] = agreement / len(reference_labels) if reference_labels else 0.0
    return metrics


def compare_with_released_scorer(
    *,
    texts: Sequence[str],
    model_predictions: Sequence[str],
    released_backend: NLGBiasClassifier | Any,
) -> dict[str, Any]:
    released_predictions = released_backend.predict_batch(list(texts))
    metrics = compute_agreement_metrics(released_predictions, model_predictions)
    metrics["released_predictions"] = list(released_predictions)
    return metrics
