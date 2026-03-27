from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .inference import ExAIInferenceRunner
from .lrp_linear import explain_classifier_head
from .schemas import InferenceResult

TRANSFORMER_APPROXIMATION_NOTE = (
    "Classifier-head relevance is computed with exact epsilon-LRP. Each encoder block is then "
    "approximated as a relevance-preserving token mixer: feed-forward and layer-norm sublayers "
    "are treated as token-local identity maps, residual connections split relevance 50/50 between "
    "skip and transformed paths, and self-attention redistributes relevance with head-averaged "
    "attention probabilities."
)


@dataclass(frozen=True)
class TransformerLRPResult:
    token_relevance: torch.Tensor
    linear_relevance: torch.Tensor
    target_logit: float
    approximation_note: str


def propagate_attention_relevance(
    token_relevance: torch.Tensor,
    attentions: tuple[torch.Tensor, ...],
    *,
    residual_weight: float = 0.5,
) -> torch.Tensor:
    relevance = token_relevance.detach().cpu()
    if not attentions:
        return relevance

    for attention_tensor in reversed(attentions):
        attention_mean = attention_tensor.mean(dim=0)
        attention_mean = attention_mean / attention_mean.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        relevance = residual_weight * relevance + (1.0 - residual_weight) * torch.matmul(
            attention_mean.transpose(0, 1),
            relevance,
        )
    return relevance


def explain_transformer(
    *,
    inference_result: InferenceResult,
    model: Any,
    epsilon: float = 1e-6,
) -> TransformerLRPResult:
    linear_result = explain_classifier_head(
        inference_result=inference_result,
        model=model,
        epsilon=epsilon,
    )
    propagated_relevance = propagate_attention_relevance(
        linear_result.token_relevance,
        inference_result.attentions,
    )
    return TransformerLRPResult(
        token_relevance=propagated_relevance,
        linear_relevance=linear_result.token_relevance,
        target_logit=linear_result.target_logit,
        approximation_note=TRANSFORMER_APPROXIMATION_NOTE,
    )


class TransformerLRPExplainer:
    def __init__(self, runner: ExAIInferenceRunner, *, epsilon: float = 1e-6) -> None:
        self.runner = runner
        self.epsilon = epsilon

    def explain_inference(self, inference_result: InferenceResult) -> TransformerLRPResult:
        return explain_transformer(
            inference_result=inference_result,
            model=self.runner.bundle.model,
            epsilon=self.epsilon,
        )

    def explain_text(self, text: str, *, target_label: str | None = None) -> TransformerLRPResult:
        inference_result = self.runner.predict_text(text, target_label=target_label)
        return self.explain_inference(inference_result)
