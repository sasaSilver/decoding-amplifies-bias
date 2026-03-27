from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .constants import EXAI_LABELS

LABEL_TO_ID = {label: index for index, label in enumerate(EXAI_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

TokenizerLoader = Callable[[str], Any]
ModelLoader = Callable[..., Any]


@dataclass(frozen=True)
class ClassifierBundle:
    model: Any
    tokenizer: Any
    model_name_or_path: str


def load_classifier_bundle(
    model_name_or_path: str | Path,
    *,
    tokenizer_loader: TokenizerLoader | None = None,
    model_loader: ModelLoader | None = None,
) -> ClassifierBundle:
    resolved_reference = (
        str(Path(model_name_or_path).expanduser())
        if isinstance(model_name_or_path, Path)
        else model_name_or_path
    )
    active_tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
    active_model_loader = model_loader or AutoModelForSequenceClassification.from_pretrained

    tokenizer = active_tokenizer_loader(resolved_reference)
    model = active_model_loader(
        resolved_reference,
        num_labels=len(EXAI_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    if hasattr(model, "config"):
        model.config.num_labels = len(EXAI_LABELS)
        model.config.id2label = ID_TO_LABEL
        model.config.label2id = LABEL_TO_ID

    return ClassifierBundle(
        model=model,
        tokenizer=tokenizer,
        model_name_or_path=resolved_reference,
    )


def save_classifier_bundle(bundle: ClassifierBundle, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(checkpoint_dir)
    bundle.tokenizer.save_pretrained(checkpoint_dir)
