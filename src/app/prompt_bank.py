import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

from .models import PromptRecord

REQUIRED_COLUMNS = ("prompt_id", "template_id", "prompt_type", "demographic", "prompt_text")


class PromptBankValidationError(ValueError):
    """Raised when the fixed prompt bank violates reproducibility constraints."""


def _clean_value(row: Mapping[str, str | None], field_name: str, row_number: int) -> str:
    raw_value = row.get(field_name)
    if raw_value is None:
        raise PromptBankValidationError(f"Row {row_number} is missing the {field_name!r} column.")

    value = raw_value.strip()
    if not value:
        raise PromptBankValidationError(f"Row {row_number} has an empty {field_name!r} value.")
    return value


def load_prompt_bank(path: Path) -> list[PromptRecord]:
    prompt_bank_path = Path(path).expanduser().resolve()
    with prompt_bank_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise PromptBankValidationError(
                f"Prompt bank is missing required columns: {', '.join(missing_columns)}."
            )

        records = [
            PromptRecord(
                prompt_id=_clean_value(row, "prompt_id", row_number),
                template_id=_clean_value(row, "template_id", row_number),
                prompt_type=_clean_value(row, "prompt_type", row_number),
                demographic=_clean_value(row, "demographic", row_number),
                prompt_text=_clean_value(row, "prompt_text", row_number),
            )
            for row_number, row in enumerate(reader, start=2)
        ]

    validate_prompt_bank(records)
    return records


def validate_prompt_bank(records: Sequence[PromptRecord]) -> None:
    errors: list[str] = []

    if not 30 <= len(records) <= 80:
        errors.append("Prompt bank must contain between 30 and 80 prompts.")

    prompt_ids = [record.prompt_id for record in records]
    if len(set(prompt_ids)) != len(prompt_ids):
        errors.append("prompt_id values must be unique.")

    demographics = {record.demographic for record in records}
    if len(demographics) < 2:
        errors.append("Prompt bank must contain at least two demographics.")

    template_demographics: defaultdict[str, list[str]] = defaultdict(list)
    template_prompt_types: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()

    for record in records:
        if record.demographic not in record.prompt_text:
            errors.append(f"{record.prompt_id} does not include its demographic in prompt_text.")

        template_demographics[record.template_id].append(record.demographic)
        pair = (record.template_id, record.demographic)
        if pair in seen_pairs:
            errors.append(f"Duplicate template/demographic pair found for {record.prompt_id}.")
        seen_pairs.add(pair)

        previous_prompt_type = template_prompt_types.setdefault(
            record.template_id, record.prompt_type
        )
        if previous_prompt_type != record.prompt_type:
            errors.append(f"{record.template_id} mixes prompt types across rows.")

    expected_demographics: frozenset[str] | None = None
    for template_id, template_members in sorted(template_demographics.items()):
        template_set = frozenset(template_members)
        if len(template_set) != len(template_members):
            errors.append(f"{template_id} repeats the same demographic more than once.")

        if expected_demographics is None:
            expected_demographics = template_set
            continue

        if template_set != expected_demographics:
            errors.append(
                f"{template_id} must cover the same demographic set as the rest of the prompt bank."
            )

    if errors:
        unique_errors = sorted(set(errors))
        raise PromptBankValidationError(" ".join(unique_errors))


def prompt_bank_digest(records: Sequence[PromptRecord]) -> str:
    canonical_records = [
        record.model_dump() for record in sorted(records, key=lambda item: item.prompt_id)
    ]
    payload = json.dumps(canonical_records, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()
