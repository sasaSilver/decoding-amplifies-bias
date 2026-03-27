from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .inference import ExAIInferenceRunner
from .lrp_core import epsilon_lrp_linear
from .schemas import InferenceResult


@dataclass(frozen=True)
class LinearLRPResult:
    token_relevance: torch.Tensor
    cls_relevance_vector: torch.Tensor
    target_logit: float
    approximation_note: str


def resolve_classifier_layer(model: Any) -> Any:
    for attribute in ("classifier", "score"):
        layer = getattr(model, attribute, None)
        if layer is not None:
            return layer
    raise ValueError("Model does not expose a supported classification layer.")


def explain_classifier_head(
    *,
    inference_result: InferenceResult,
    model: Any,
    epsilon: float = 1e-6,
) -> LinearLRPResult:
    if not inference_result.hidden_states:
        raise ValueError("Inference result must include hidden_states for linear LRP.")

    classifier = resolve_classifier_layer(model)
    if not hasattr(classifier, "weight"):
        raise ValueError("Classifier layer must expose a weight matrix.")

    final_hidden_state = inference_result.hidden_states[-1]
    cls_hidden_state = final_hidden_state[0]
    output_relevance = torch.zeros_like(inference_result.logits)
    output_relevance[inference_result.target_label_id] = inference_result.logits[
        inference_result.target_label_id
    ]
    cls_relevance = epsilon_lrp_linear(
        inputs=cls_hidden_state,
        weight=classifier.weight.detach().cpu(),
        relevance=output_relevance.detach().cpu(),
        bias=classifier.bias.detach().cpu()
        if getattr(classifier, "bias", None) is not None
        else None,
        epsilon=epsilon,
    )
    token_relevance = torch.matmul(final_hidden_state, cls_relevance)

    return LinearLRPResult(
        token_relevance=token_relevance,
        cls_relevance_vector=cls_relevance,
        target_logit=float(inference_result.logits[inference_result.target_label_id].item()),
        approximation_note=(
            "Exact epsilon-LRP is applied through the classification head. Token scores at the "
            "final hidden layer are obtained by projecting token states onto the CLS relevance "
            "vector."
        ),
    )


class LinearLRPExplainer:
    def __init__(self, runner: ExAIInferenceRunner, *, epsilon: float = 1e-6) -> None:
        self.runner = runner
        self.epsilon = epsilon

    def explain_text(self, text: str, *, target_label: str | None = None) -> LinearLRPResult:
        inference_result = self.runner.predict_text(text, target_label=target_label)
        return explain_classifier_head(
            inference_result=inference_result,
            model=self.runner.bundle.model,
            epsilon=self.epsilon,
        )
