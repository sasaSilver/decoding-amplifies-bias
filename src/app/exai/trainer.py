from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset

from app.device import resolve_torch_device

from .config import ExAITrainingConfig
from .constants import EXAI_LABELS
from .data import RegardDatasetRecord
from .eval import compute_classification_metrics, predict_texts
from .modeling import ID_TO_LABEL, LABEL_TO_ID, load_classifier_bundle, save_classifier_bundle
from .utils import collect_environment_snapshot, file_digest, utc_now_iso, write_json


class EncodedTextDataset(Dataset[Any]):
    def __init__(
        self,
        records: Sequence[RegardDatasetRecord],
        *,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.records = list(records)
        texts = [record.active_text for record in records]
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.encodings = {key: value.clone().detach() for key, value in encodings.items()}
        self.labels = torch.tensor(
            [LABEL_TO_ID[record.label] for record in records],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


class TrainingRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_key: str
    checkpoint_dir: Path
    manifest_path: Path
    metrics_path: Path
    best_epoch: int
    from_cache: bool


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    *,
    model: Any,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: str,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    epoch_losses: list[float] = []
    true_labels: list[str] = []
    predicted_labels: list[str] = []

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if is_training:
            optimizer.zero_grad()

        outputs = model(**batch, return_dict=True)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model forward pass did not return a loss tensor.")
        if is_training:
            loss.backward()
            optimizer.step()

        logits = outputs.logits.detach().cpu()
        predicted_ids = torch.argmax(logits, dim=-1).tolist()
        predicted_labels.extend(ID_TO_LABEL[int(index)] for index in predicted_ids)
        true_labels.extend(
            ID_TO_LABEL[int(index)] for index in batch["labels"].detach().cpu().tolist()
        )
        epoch_losses.append(float(loss.detach().cpu().item()))

    metrics = compute_classification_metrics(true_labels, predicted_labels)
    metrics["loss"] = mean(epoch_losses) if epoch_losses else 0.0
    return metrics


def _checkpoint_artifact_digests(checkpoint_dir: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in sorted(checkpoint_dir.iterdir()):
        if path.is_file():
            digests[path.name] = file_digest(path)
    return digests


def train_classifier(
    *,
    train_records: Sequence[RegardDatasetRecord],
    validation_records: Sequence[RegardDatasetRecord],
    split_manifest_path: Path,
    config: ExAITrainingConfig,
    tokenizer_loader: Any | None = None,
    model_loader: Any | None = None,
) -> TrainingRunResult:
    if not train_records:
        raise ValueError("Training split is empty.")
    if not validation_records:
        raise ValueError("Validation split is empty.")

    paths = config.output_paths.ensure_dirs()
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    run_payload = {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "seed": config.seed,
        "split_membership_digest": split_manifest["split_membership_digest"],
    }
    run_key = (
        file_digest(split_manifest_path)[:10]
        + "_"
        + json.dumps(run_payload, sort_keys=True).encode().hex()[:10]
    )
    checkpoint_dir = paths.models_dir / f"classifier_{run_key}"
    manifest_path = checkpoint_dir / "training_manifest.json"
    metrics_path = checkpoint_dir / "training_metrics.json"

    if checkpoint_dir.exists() and manifest_path.exists() and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return TrainingRunResult(
            run_key=run_key,
            checkpoint_dir=checkpoint_dir,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
            best_epoch=int(metrics["best_epoch"]),
            from_cache=True,
        )

    _set_reproducible_seed(config.seed)
    bundle = load_classifier_bundle(
        config.model_name,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    resolved_device = resolve_torch_device(config.device)
    bundle.model.to(resolved_device)

    train_dataset = EncodedTextDataset(
        train_records,
        tokenizer=bundle.tokenizer,
        max_length=config.max_length,
    )
    validation_dataset = EncodedTextDataset(
        validation_records,
        tokenizer=bundle.tokenizer,
        max_length=config.max_length,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=config.learning_rate)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_epoch = 1
    best_macro_f1 = float("-inf")
    patience_without_improvement = 0

    for epoch_index in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model=bundle.model,
            dataloader=train_loader,
            device=resolved_device,
            optimizer=optimizer,
        )
        validation_metrics = _run_epoch(
            model=bundle.model,
            dataloader=validation_loader,
            device=resolved_device,
            optimizer=None,
        )
        epoch_record = {
            "epoch": epoch_index,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)

        if float(validation_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(validation_metrics["macro_f1"])
            best_epoch = epoch_index
            patience_without_improvement = 0
            save_classifier_bundle(bundle, checkpoint_dir)
        else:
            patience_without_improvement += 1
            if patience_without_improvement >= config.early_stopping_patience:
                break

    reload_bundle = load_classifier_bundle(
        checkpoint_dir,
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    reload_bundle.model.to(resolved_device)
    reload_bundle.model.eval()
    validation_predictions, _ = predict_texts(
        model=reload_bundle.model,
        tokenizer=reload_bundle.tokenizer,
        texts=[record.active_text for record in validation_records],
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=resolved_device,
    )
    best_validation_metrics = compute_classification_metrics(
        [record.label for record in validation_records],
        validation_predictions,
    )
    metrics_payload = {
        "created_at_utc": utc_now_iso(),
        "run_key": run_key,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "history": history,
    }
    write_json(metrics_path, metrics_payload)

    manifest_payload = {
        "created_at_utc": utc_now_iso(),
        "run_key": run_key,
        "model_name": config.model_name,
        "split_manifest_path": str(split_manifest_path.resolve()),
        "split_membership_digest": split_manifest["split_membership_digest"],
        "seed": config.seed,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "best_epoch": best_epoch,
        "device": resolved_device,
        "environment": collect_environment_snapshot(device=resolved_device),
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "artifacts": {
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "checkpoint_digests": _checkpoint_artifact_digests(checkpoint_dir),
        },
        "labels": list(EXAI_LABELS),
    }
    write_json(manifest_path, manifest_payload)

    return TrainingRunResult(
        run_key=run_key,
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        best_epoch=best_epoch,
        from_cache=False,
    )
