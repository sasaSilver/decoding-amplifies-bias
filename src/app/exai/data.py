from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.scoring import mask_text

from .constants import EXAI_LABELS
from .utils import canonical_json_digest, file_digest

TEXT_LABEL_ALIASES = {
    "-1": "negative",
    "0": "negative",
    "neg": "negative",
    "negative": "negative",
    "1": "neutral",
    "neu": "neutral",
    "neutral": "neutral",
    "2": "positive",
    "pos": "positive",
    "positive": "positive",
    "3": "other",
    "other": "other",
}
LEGACY_NUMERIC_LABEL_ALIASES = {
    "0": "negative",
    "1": "neutral",
    "2": "positive",
    "3": "other",
}
NLG_BIAS_NUMERIC_LABEL_ALIASES = {
    "-1": "negative",
    "0": "neutral",
    "1": "positive",
    "2": "other",
}
LABEL_HEADER_NAMES = ("label", "labels", "regard_label", "class", "target")
TEXT_HEADER_NAMES = ("text", "sentence", "comment", "utterance")
MASKED_TEXT_HEADER_NAMES = ("masked_text", "text_masked", "masked_sentence")
DEMOGRAPHIC_HEADER_NAMES = ("demographic", "group", "group_term", "identity", "identity_term")


class RegardDatasetError(ValueError):
    """Raised when the ExAI regard dataset cannot be parsed reproducibly."""


class RegardDatasetRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    example_id: str
    source_file: str
    source_row: int
    raw_label: str
    label: str
    text: str
    text_masked: str
    active_text: str
    demographic: str | None = None
    masking_applied: bool = False


def normalize_regard_label(
    raw_label: str,
    *,
    numeric_aliases: Mapping[str, str] | None = None,
) -> str:
    normalized = raw_label.strip().lower()
    if numeric_aliases is not None and normalized in numeric_aliases:
        return numeric_aliases[normalized]
    if normalized in TEXT_LABEL_ALIASES:
        return TEXT_LABEL_ALIASES[normalized]
    raise RegardDatasetError(
        f"Unsupported regard label {raw_label!r}. Expected one of {sorted(TEXT_LABEL_ALIASES)}."
    )


def _infer_numeric_label_aliases(
    raw_labels: Sequence[str],
    *,
    source_path: Path,
) -> Mapping[str, str] | None:
    normalized_labels = {label.strip().lower() for label in raw_labels if label.strip()}
    if not normalized_labels:
        return None

    numeric_labels = {label for label in normalized_labels if label.lstrip("-").isdigit()}
    if not numeric_labels:
        return None

    supported_numeric_labels = set(LEGACY_NUMERIC_LABEL_ALIASES) | set(
        NLG_BIAS_NUMERIC_LABEL_ALIASES
    )
    unsupported_numeric_labels = numeric_labels - supported_numeric_labels
    if unsupported_numeric_labels:
        raise RegardDatasetError(
            f"{source_path} contains unsupported numeric labels: {sorted(unsupported_numeric_labels)}."
        )

    uses_nlg_bias_scheme = "-1" in numeric_labels
    uses_legacy_scheme = "3" in numeric_labels
    if uses_nlg_bias_scheme and uses_legacy_scheme:
        raise RegardDatasetError(
            f"{source_path} mixes incompatible numeric label schemes: {sorted(numeric_labels)}."
        )
    if uses_nlg_bias_scheme:
        return NLG_BIAS_NUMERIC_LABEL_ALIASES
    if uses_legacy_scheme:
        return LEGACY_NUMERIC_LABEL_ALIASES
    if numeric_labels <= {"0", "1", "2"}:
        raise RegardDatasetError(
            f"{source_path} uses ambiguous numeric labels {sorted(numeric_labels)}. "
            "Use textual labels or include -1/3 to disambiguate the source scheme."
        )
    return LEGACY_NUMERIC_LABEL_ALIASES


def resolve_dataset_sources(dataset_path: Path) -> list[Path]:
    resolved = dataset_path.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".tsv":
            raise RegardDatasetError("Dataset file must be a .tsv file.")
        return [resolved]

    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {resolved}")

    sources = sorted(path for path in resolved.rglob("*.tsv") if path.is_file())
    if not sources:
        raise RegardDatasetError(f"No .tsv files found under {resolved}")
    return sources


def _clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().replace("\ufeff", "")


def _match_header(fieldnames: list[str], aliases: tuple[str, ...]) -> str | None:
    mapping = {name.strip().lower(): name for name in fieldnames}
    for alias in aliases:
        matched = mapping.get(alias)
        if matched is not None:
            return matched
    return None


def _looks_like_header(first_row: list[str]) -> bool:
    if len(first_row) < 2:
        return False

    lowered = [_clean_cell(cell).lower() for cell in first_row]
    if lowered[0] in LABEL_HEADER_NAMES and lowered[1] in TEXT_HEADER_NAMES:
        return True

    return any(cell in LABEL_HEADER_NAMES + TEXT_HEADER_NAMES for cell in lowered)


def _build_record(
    *,
    label: str,
    text: str,
    source_path: Path,
    source_row: int,
    demographic: str | None,
    masked_text: str | None,
    use_masking: bool,
    numeric_label_aliases: Mapping[str, str] | None,
) -> RegardDatasetRecord:
    normalized_label = normalize_regard_label(label, numeric_aliases=numeric_label_aliases)
    cleaned_text = text.strip()
    if not cleaned_text:
        raise RegardDatasetError(f"{source_path}:{source_row} has empty text.")

    cleaned_demographic = demographic.strip() if demographic else None
    masked = masked_text.strip() if masked_text else cleaned_text
    if cleaned_demographic:
        masked = mask_text(cleaned_text, cleaned_demographic)

    active_text = masked if use_masking else cleaned_text
    example_payload = {
        "source_file": str(source_path.resolve()),
        "source_row": source_row,
        "label": normalized_label,
        "text": cleaned_text,
        "demographic": cleaned_demographic,
    }
    example_id = canonical_json_digest(example_payload)[:20]

    return RegardDatasetRecord(
        example_id=example_id,
        source_file=str(source_path.resolve()),
        source_row=source_row,
        raw_label=label,
        label=normalized_label,
        text=cleaned_text,
        text_masked=masked,
        active_text=active_text,
        demographic=cleaned_demographic,
        masking_applied=active_text != cleaned_text,
    )


def _load_rows_with_header(source_path: Path, use_masking: bool) -> list[RegardDatasetRecord]:
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        label_field = _match_header(fieldnames, LABEL_HEADER_NAMES)
        text_field = _match_header(fieldnames, TEXT_HEADER_NAMES)
        if label_field is None or text_field is None:
            raise RegardDatasetError(
                f"{source_path} is missing label/text columns in header: {fieldnames!r}"
            )

        demographic_field = _match_header(fieldnames, DEMOGRAPHIC_HEADER_NAMES)
        masked_text_field = _match_header(fieldnames, MASKED_TEXT_HEADER_NAMES)
        rows = list(reader)
        numeric_label_aliases = _infer_numeric_label_aliases(
            [_clean_cell(row.get(label_field)) for row in rows],
            source_path=source_path,
        )
        records: list[RegardDatasetRecord] = []
        for row_number, row in enumerate(rows, start=2):
            records.append(
                _build_record(
                    label=_clean_cell(row.get(label_field)),
                    text=_clean_cell(row.get(text_field)),
                    source_path=source_path,
                    source_row=row_number,
                    demographic=_clean_cell(row.get(demographic_field))
                    if demographic_field
                    else None,
                    masked_text=_clean_cell(row.get(masked_text_field))
                    if masked_text_field
                    else None,
                    use_masking=use_masking,
                    numeric_label_aliases=numeric_label_aliases,
                )
            )
        return records


def _load_rows_without_header(source_path: Path, use_masking: bool) -> list[RegardDatasetRecord]:
    records: list[RegardDatasetRecord] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)
        numeric_label_aliases = _infer_numeric_label_aliases(
            [_clean_cell(row[0]) for row in rows if row],
            source_path=source_path,
        )
        for row_number, row in enumerate(rows, start=1):
            cells = [_clean_cell(cell) for cell in row]
            if not any(cells):
                continue
            if len(cells) < 2:
                raise RegardDatasetError(
                    f"{source_path}:{row_number} must have at least label and text columns."
                )

            demographic = cells[2] if len(cells) >= 3 and cells[2] else None
            masked_text = cells[3] if len(cells) >= 4 and cells[3] else None
            records.append(
                _build_record(
                    label=cells[0],
                    text=cells[1],
                    source_path=source_path,
                    source_row=row_number,
                    demographic=demographic,
                    masked_text=masked_text,
                    use_masking=use_masking,
                    numeric_label_aliases=numeric_label_aliases,
                )
            )
    return records


def load_regard_dataset(dataset_path: Path, *, use_masking: bool) -> list[RegardDatasetRecord]:
    records: list[RegardDatasetRecord] = []
    seen_example_ids: set[str] = set()

    for source_path in resolve_dataset_sources(dataset_path):
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            try:
                first_row = next(reader)
            except StopIteration as exc:
                raise RegardDatasetError(f"Dataset file is empty: {source_path}") from exc

        source_records = (
            _load_rows_with_header(source_path, use_masking)
            if _looks_like_header(first_row)
            else _load_rows_without_header(source_path, use_masking)
        )
        for record in source_records:
            if record.example_id in seen_example_ids:
                raise RegardDatasetError(f"Duplicate example_id detected for {record.source_file}.")
            seen_example_ids.add(record.example_id)
            records.append(record)

    if not records:
        raise RegardDatasetError("No dataset records were parsed.")

    return sorted(records, key=lambda item: (item.label, item.example_id))


def summarize_regard_dataset(
    records: list[RegardDatasetRecord],
    *,
    source_paths: list[Path],
    use_masking: bool,
) -> dict[str, Any]:
    label_counts = Counter(record.label for record in records)
    source_counts = Counter(record.source_file for record in records)
    masking_count = sum(1 for record in records if record.masking_applied)
    source_payload = [
        {
            "path": str(path.resolve()),
            "sha256": file_digest(path),
        }
        for path in source_paths
    ]

    return {
        "record_count": len(records),
        "labels": list(EXAI_LABELS),
        "label_counts": {label: label_counts.get(label, 0) for label in EXAI_LABELS},
        "source_record_counts": dict(sorted(source_counts.items())),
        "source_files": source_payload,
        "use_masking": use_masking,
        "masked_record_count": masking_count,
    }
