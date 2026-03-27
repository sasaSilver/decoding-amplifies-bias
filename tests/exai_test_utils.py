from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch


def write_training_dataset_fixture(root: Path) -> Path:
    dataset_dir = root / "regard"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = ["label\ttext\tdemographic"]
    labels = [
        ("negative", "doctor", "was rude"),
        ("neutral", "teacher", "entered the room"),
        ("positive", "nurse", "helped the patient"),
        ("other", "artist", "waited outside"),
    ]
    for repeat_index in range(4):
        for label, demographic, suffix in labels:
            rows.append(f"{label}\tThe {demographic} {suffix} {repeat_index}.\t{demographic}")
    (dataset_dir / "train.tsv").write_text("\n".join(rows), encoding="utf-8")
    return dataset_dir


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts: str | list[str],
        *,
        padding: str | bool = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        batch = [texts] if isinstance(texts, str) else list(texts)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for text in batch:
            token_values = [101]
            for token in text.split():
                token_values.append(sum(ord(char) for char in token) % 97 + 1)
            token_values.append(102)
            token_values = token_values[:max_length]
            attention_mask = [1] * len(token_values)
            while len(token_values) < max_length:
                token_values.append(self.pad_token_id)
                attention_mask.append(0)
            input_ids.append(token_values)
            attention_masks.append(attention_mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        tokens = []
        for token_id in token_ids:
            if token_id == 101:
                tokens.append("[CLS]")
            elif token_id == 102:
                tokens.append("[SEP]")
            elif token_id == 0:
                tokens.append("[PAD]")
            else:
                tokens.append(f"tok_{token_id}")
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        words = [
            token.replace("tok_", "")
            for token in tokens
            if token not in {"[CLS]", "[SEP]", "[PAD]"}
        ]
        return " ".join(words)

    def save_pretrained(self, path: str | Path) -> None:
        resolved = Path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        (resolved / "tokenizer.json").write_text(
            json.dumps({"kind": "fake-tokenizer"}, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class FakeSequenceClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 12)
        self.classifier = torch.nn.Linear(12, 4)
        self.config = SimpleNamespace(
            num_labels=4,
            id2label={0: "negative", 1: "neutral", 2: "positive", 3: "other"},
            label2id={"negative": 0, "neutral": 1, "positive": 2, "other": 3},
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        embeddings = self.embedding(input_ids)
        masked = embeddings * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        logits = self.classifier(pooled)
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        hidden_states = (embeddings, masked) if output_hidden_states else None
        attention = attention_mask.unsqueeze(1).unsqueeze(2).float()
        attention = attention * attention.transpose(-1, -2)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
        attentions = (attention.repeat(1, 2, 1, 1),) if output_attentions else None
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def save_pretrained(self, path: str | Path) -> None:
        resolved = Path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), resolved / "weights.pt")
        (resolved / "config.json").write_text(
            json.dumps({"num_labels": 4}, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def fake_tokenizer_loader(_: str) -> FakeTokenizer:
    return FakeTokenizer()


def fake_model_loader(model_name_or_path: str, **_: Any) -> FakeSequenceClassifier:
    model = FakeSequenceClassifier()
    weights_path = Path(model_name_or_path) / "weights.pt"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    return model


def write_benchmark_score_artifacts(root: Path) -> Path:
    outputs_dir = root / "outputs"
    scores_dir = outputs_dir / "scores"
    manifests_dir = outputs_dir / "manifests"
    generations_dir = outputs_dir / "generations"
    scores_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(parents=True, exist_ok=True)

    score_path = scores_dir / "score_fixture.parquet"
    frame = pd.DataFrame(
        [
            {
                "cache_key": "score_fixture",
                "model_name": "gpt2",
                "prompt_id": "neg_example",
                "template_id": "tmpl_1",
                "prompt_type": "occupation",
                "demographic": "Group A",
                "prompt_text": "Prompt A",
                "decoding_strategy": "greedy",
                "do_sample": False,
                "seed": 0,
                "max_new_tokens": 40,
                "sample_index": 0,
                "raw_text": "Prompt A negative",
                "completion_text": "The doctor was rude.",
                "regard_label": "negative",
                "scoring_masked": True,
            },
            {
                "cache_key": "score_fixture",
                "model_name": "gpt2",
                "prompt_id": "neu_example",
                "template_id": "tmpl_2",
                "prompt_type": "descriptor",
                "demographic": "Group B",
                "prompt_text": "Prompt B",
                "decoding_strategy": "greedy",
                "do_sample": False,
                "seed": 1,
                "max_new_tokens": 40,
                "sample_index": 1,
                "raw_text": "Prompt B neutral",
                "completion_text": "The teacher entered the room.",
                "regard_label": "neutral",
                "scoring_masked": True,
            },
            {
                "cache_key": "score_fixture",
                "model_name": "gpt2",
                "prompt_id": "pos_example",
                "template_id": "tmpl_3",
                "prompt_type": "occupation",
                "demographic": "Group C",
                "prompt_text": "Prompt C",
                "decoding_strategy": "greedy",
                "do_sample": False,
                "seed": 2,
                "max_new_tokens": 40,
                "sample_index": 2,
                "raw_text": "Prompt C positive",
                "completion_text": "The nurse helped the patient.",
                "regard_label": "positive",
                "scoring_masked": True,
            },
            {
                "cache_key": "score_fixture",
                "model_name": "gpt2",
                "prompt_id": "other_example",
                "template_id": "tmpl_4",
                "prompt_type": "descriptor",
                "demographic": "Group D",
                "prompt_text": "Prompt D",
                "decoding_strategy": "greedy",
                "do_sample": False,
                "seed": 3,
                "max_new_tokens": 40,
                "sample_index": 3,
                "raw_text": "Prompt D other",
                "completion_text": "The artist waited outside.",
                "regard_label": "other",
                "scoring_masked": True,
            },
        ]
    )
    frame.to_parquet(score_path, index=False)
    generation_path = generations_dir / "score_fixture.parquet"
    generation_path.write_text("placeholder", encoding="utf-8")
    (manifests_dir / "score_fixture.json").write_text(
        json.dumps(
            {
                "cache_key": "score_fixture",
                "generations_cache_key": "generation_fixture",
                "generations_path": str(generation_path.resolve()),
                "use_masking": True,
                "artifacts": {"scores_path": str(score_path.resolve())},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    combined_manifest = manifests_dir / "score_fixture_week3_combined.json"
    combined_manifest.write_text(
        json.dumps(
            {
                "cache_key": "score_fixture_combined",
                "created_from_scores": [str(score_path.resolve())],
                "use_masking": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return combined_manifest
