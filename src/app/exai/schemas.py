from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class InferenceResult:
    text: str
    tokens: list[str]
    token_ids: list[int]
    attention_mask: list[int]
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_label: str
    predicted_label_id: int
    target_label: str
    target_label_id: int
    negative_label_id: int
    hidden_states: tuple[torch.Tensor, ...]
    attentions: tuple[torch.Tensor, ...]
    encoded_inputs: dict[str, torch.Tensor]
    metadata: dict[str, Any]
