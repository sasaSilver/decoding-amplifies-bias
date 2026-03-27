from __future__ import annotations

import torch


def stabilize_denominator(denominator: torch.Tensor, epsilon: float) -> torch.Tensor:
    sign = torch.where(
        denominator >= 0, torch.ones_like(denominator), -torch.ones_like(denominator)
    )
    return denominator + epsilon * sign


def epsilon_lrp_linear(
    *,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    relevance: torch.Tensor,
    bias: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if inputs.dim() != 1:
        raise ValueError("epsilon_lrp_linear expects a 1D input activation tensor.")
    if weight.dim() != 2:
        raise ValueError("epsilon_lrp_linear expects a 2D weight matrix.")
    if relevance.dim() != 1:
        raise ValueError("epsilon_lrp_linear expects a 1D output relevance tensor.")

    contributions = weight * inputs.unsqueeze(0)
    denominator = contributions.sum(dim=1)
    if bias is not None:
        denominator = denominator + bias
    stabilized = stabilize_denominator(denominator, epsilon).unsqueeze(1)
    redistribution = contributions / stabilized
    return (redistribution * relevance.unsqueeze(1)).sum(dim=0)
