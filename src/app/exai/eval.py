from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from app.device import resolve_torch_device

from .constants import EXAI_LABELS
from .data import RegardDatasetRecord
from .modeling import ID_TO_LABEL, load_classifier_bundle


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_classification_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, Any]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length.")

    total = len(true_labels)
    accuracy = _safe_divide(
        sum(
            1
            for true, predicted in zip(true_labels, predicted_labels, strict=False)
            if true == predicted
        ),
        total,
    )

    confusion_matrix: list[list[int]] = []
    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for label in EXAI_LABELS:
        row = []
        tp = fp = fn = 0
        support = sum(1 for true in true_labels if true == label)
        for predicted_label in EXAI_LABELS:
            count = sum(
                1
                for true, predicted in zip(true_labels, predicted_labels, strict=False)
                if true == label and predicted == predicted_label
            )
            row.append(count)
            if predicted_label == label:
                tp = count
        fp = sum(
            1
            for true, predicted in zip(true_labels, predicted_labels, strict=False)
            if true != label and predicted == label
        )
        fn = sum(
            1
            for true, predicted in zip(true_labels, predicted_labels, strict=False)
            if true == label and predicted != label
        )

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = (
            _safe_divide(2 * precision * recall, precision + recall) if precision or recall else 0.0
        )
        confusion_matrix.append(row)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / len(EXAI_LABELS),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(EXAI_LABELS),
            "matrix": confusion_matrix,
        },
    }


def predict_texts(
    *,
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[list[str], list[list[float]]]:
    predictions: list[str] = []
    probabilities: list[list[float]] = []
    for start_index in range(0, len(texts), batch_size):
        batch = list(texts[start_index : start_index + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, return_dict=True)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu()
        predicted_ids = torch.argmax(probs, dim=-1).tolist()
        predictions.extend(ID_TO_LABEL[int(index)] for index in predicted_ids)
        probabilities.extend(probs.tolist())
    return predictions, probabilities


def evaluate_records(
    *,
    checkpoint_path: Path,
    records: Sequence[RegardDatasetRecord],
    batch_size: int,
    max_length: int,
    device: str,
    tokenizer_loader: Any | None = None,
    model_loader: Any | None = None,
) -> dict[str, Any]:
    bundle = load_classifier_bundle(
        checkpoint_path,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    resolved_device = resolve_torch_device(device)
    bundle.model.to(resolved_device)
    bundle.model.eval()

    true_labels = [record.label for record in records]
    texts = [record.active_text for record in records]
    predicted_labels, probabilities = predict_texts(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
        device=resolved_device,
    )
    metrics = compute_classification_metrics(true_labels, predicted_labels)
    metrics["record_count"] = len(records)
    metrics["probabilities"] = probabilities
    metrics["predicted_labels"] = predicted_labels
    return metrics


def evaluate_benchmark_predictions(
    *,
    checkpoint_path: Path,
    benchmark_path: Path,
    batch_size: int,
    max_length: int,
    device: str,
    tokenizer_loader: Any | None = None,
    model_loader: Any | None = None,
) -> dict[str, Any]:
    bundle = load_classifier_bundle(
        checkpoint_path,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    resolved_device = resolve_torch_device(device)
    bundle.model.to(resolved_device)
    bundle.model.eval()

    benchmark_df = pd.read_parquet(benchmark_path)
    texts = benchmark_df["completion_text"].tolist()
    released_labels = benchmark_df["predicted_label"].tolist()
    predicted_labels, probabilities = predict_texts(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
        device=resolved_device,
    )
    metrics = compute_classification_metrics(released_labels, predicted_labels)
    metrics["record_count"] = int(len(benchmark_df))
    metrics["probabilities"] = probabilities
    metrics["predicted_labels"] = predicted_labels
    metrics["benchmark_ids"] = benchmark_df["benchmark_id"].tolist()
    return metrics


def build_error_analysis(
    *,
    benchmark_path: Path,
    predicted_labels: Sequence[str],
    max_examples: int = 10,
) -> list[dict[str, Any]]:
    benchmark_df = pd.read_parquet(benchmark_path)
    mismatches: list[dict[str, Any]] = []
    for (_, row), predicted_label in zip(benchmark_df.iterrows(), predicted_labels, strict=False):
        reference_label = str(row["predicted_label"])
        if reference_label == predicted_label:
            continue
        completion_text = str(row["completion_text"])
        mismatches.append(
            {
                "benchmark_id": str(row["benchmark_id"]),
                "prompt_id": str(row["prompt_id"]),
                "demographic": str(row["demographic"]),
                "reference_label": reference_label,
                "predicted_label": predicted_label,
                "excerpt": completion_text[:120],
            }
        )
        if len(mismatches) >= max_examples:
            break
    return mismatches
