from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.device import resolve_torch_device

from .constants import NEGATIVE_LABEL
from .modeling import ID_TO_LABEL, LABEL_TO_ID, load_classifier_bundle
from .schemas import InferenceResult


class ExAIInferenceRunner:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        max_length: int = 128,
        device: str | None = None,
        tokenizer_loader: Any | None = None,
        model_loader: Any | None = None,
    ) -> None:
        bundle = load_classifier_bundle(
            checkpoint_path,
            tokenizer_loader=tokenizer_loader,
            model_loader=model_loader,
        )
        resolved_device = resolve_torch_device(device)
        bundle.model.to(resolved_device)
        bundle.model.eval()

        self.bundle = bundle
        self.device = resolved_device
        self.max_length = max_length

    def predict_text(self, text: str, *, target_label: str | None = None) -> InferenceResult:
        encoded = self.bundle.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.bundle.model(
                **encoded,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        logits = outputs.logits[0].detach().cpu()
        probabilities = torch.softmax(logits, dim=-1)
        predicted_label_id = int(torch.argmax(probabilities).item())
        predicted_label = ID_TO_LABEL[predicted_label_id]
        resolved_target_label = target_label or predicted_label
        target_label_id = LABEL_TO_ID[resolved_target_label]
        negative_label_id = LABEL_TO_ID[NEGATIVE_LABEL]

        token_ids_tensor = encoded["input_ids"][0].detach().cpu()
        attention_mask_tensor = encoded["attention_mask"][0].detach().cpu()
        active_length = int(attention_mask_tensor.sum().item())
        token_ids = token_ids_tensor[:active_length].tolist()
        attention_mask = attention_mask_tensor[:active_length].tolist()
        tokens = self.bundle.tokenizer.convert_ids_to_tokens(token_ids)

        hidden_states = tuple(
            state[0, :active_length].detach().cpu() for state in (outputs.hidden_states or ())
        )
        attentions = tuple(
            attention[0, :, :active_length, :active_length].detach().cpu()
            for attention in (outputs.attentions or ())
        )
        encoded_inputs = {
            key: value[0, :active_length].detach().cpu() for key, value in encoded.items()
        }

        return InferenceResult(
            text=text,
            tokens=tokens,
            token_ids=token_ids,
            attention_mask=attention_mask,
            logits=logits,
            probabilities=probabilities,
            predicted_label=predicted_label,
            predicted_label_id=predicted_label_id,
            target_label=resolved_target_label,
            target_label_id=target_label_id,
            negative_label_id=negative_label_id,
            hidden_states=hidden_states,
            attentions=attentions,
            encoded_inputs=encoded_inputs,
            metadata={
                "checkpoint_path": self.bundle.model_name_or_path,
                "device": self.device,
                "max_length": self.max_length,
            },
        )
