from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import ExAIDataConfig
from .constants import EXAI_LABELS
from .data import (
    RegardDatasetRecord,
    load_regard_dataset,
    resolve_dataset_sources,
    summarize_regard_dataset,
)
from .utils import canonical_json_digest, utc_now_iso, write_json


@dataclass(frozen=True)
class DatasetPreparationResult:
    records: tuple[RegardDatasetRecord, ...]
    splits: dict[str, tuple[RegardDatasetRecord, ...]]
    summary_path: Path
    split_manifest_path: Path


def _stable_sort_key(seed: int, label: str, example_id: str) -> str:
    payload = f"{seed}:{label}:{example_id}"
    return sha256(payload.encode("utf-8")).hexdigest()


def generate_deterministic_splits(
    records: list[RegardDatasetRecord],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, tuple[RegardDatasetRecord, ...]]:
    grouped: dict[str, list[RegardDatasetRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    split_records: dict[str, list[RegardDatasetRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for label in EXAI_LABELS:
        label_records = sorted(
            grouped.get(label, []),
            key=lambda item: (_stable_sort_key(seed, label, item.example_id), item.example_id),
        )
        train_count = int(len(label_records) * train_fraction)
        validation_count = int(len(label_records) * validation_fraction)
        validation_end = train_count + validation_count

        split_records["train"].extend(label_records[:train_count])
        split_records["validation"].extend(label_records[train_count:validation_end])
        split_records["test"].extend(label_records[validation_end:])

    return {
        split_name: tuple(sorted(items, key=lambda item: item.example_id))
        for split_name, items in split_records.items()
    }


def _split_membership_payload(
    splits: dict[str, tuple[RegardDatasetRecord, ...]],
) -> list[dict[str, Any]]:
    memberships: list[dict[str, Any]] = []
    for split_name, records in splits.items():
        for record in records:
            memberships.append(
                {
                    "example_id": record.example_id,
                    "split": split_name,
                    "label": record.label,
                    "source_file": record.source_file,
                    "source_row": record.source_row,
                }
            )
    return sorted(memberships, key=lambda item: item["example_id"])


def _split_counts(records: tuple[RegardDatasetRecord, ...]) -> dict[str, int]:
    counts = Counter(record.label for record in records)
    return {label: counts.get(label, 0) for label in EXAI_LABELS}


def build_split_manifest(
    *,
    config: ExAIDataConfig,
    records: list[RegardDatasetRecord],
    splits: dict[str, tuple[RegardDatasetRecord, ...]],
    source_paths: list[Path],
) -> dict[str, Any]:
    memberships = _split_membership_payload(splits)
    summary = summarize_regard_dataset(
        records,
        source_paths=source_paths,
        use_masking=config.use_masking,
    )
    split_counts = {split_name: len(split_records) for split_name, split_records in splits.items()}
    split_label_counts = {
        split_name: _split_counts(split_records) for split_name, split_records in splits.items()
    }

    return {
        "created_at_utc": utc_now_iso(),
        "dataset_path": str(config.dataset_path.expanduser().resolve()),
        "split_seed": config.split_seed,
        "train_fraction": config.train_fraction,
        "validation_fraction": config.validation_fraction,
        "test_fraction": config.test_fraction(),
        "dataset_summary_digest": canonical_json_digest(summary),
        "split_membership_digest": canonical_json_digest(memberships),
        "source_files": summary["source_files"],
        "record_count": len(records),
        "split_counts": split_counts,
        "split_label_counts": split_label_counts,
        "memberships": memberships,
    }


def prepare_regard_dataset(config: ExAIDataConfig) -> DatasetPreparationResult:
    paths = config.output_paths.ensure_dirs()
    source_paths = resolve_dataset_sources(config.dataset_path)
    records = load_regard_dataset(config.dataset_path, use_masking=config.use_masking)
    splits = generate_deterministic_splits(
        records,
        seed=config.split_seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    summary_payload = summarize_regard_dataset(
        records,
        source_paths=source_paths,
        use_masking=config.use_masking,
    )
    summary_payload["created_at_utc"] = utc_now_iso()
    summary_payload["dataset_path"] = str(config.dataset_path.expanduser().resolve())
    summary_payload["split_seed"] = config.split_seed

    split_manifest = build_split_manifest(
        config=config,
        records=records,
        splits=splits,
        source_paths=source_paths,
    )
    dataset_key = split_manifest["split_membership_digest"][:20]
    summary_path = paths.metadata_dir / f"regard_dataset_{dataset_key}_summary.json"
    split_manifest_path = paths.metadata_dir / f"regard_dataset_{dataset_key}_splits.json"
    write_json(summary_path, summary_payload)
    write_json(split_manifest_path, split_manifest)

    return DatasetPreparationResult(
        records=tuple(records),
        splits=splits,
        summary_path=summary_path,
        split_manifest_path=split_manifest_path,
    )
