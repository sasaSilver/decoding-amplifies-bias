from __future__ import annotations

from collections import Counter

import pytest

from app.prompt_bank import (
    PromptBankValidationError,
    load_prompt_bank,
    validate_prompt_bank,
)
from app.settings.settings import settings


def test_prompt_bank_meets_week1_constraints() -> None:
    records = load_prompt_bank(settings.generation.prompt_bank_path)

    assert 30 <= len(records) <= 80
    assert Counter(record.demographic for record in records) == {
        "Black woman": 12,
        "Black man": 12,
        "White woman": 12,
        "White man": 12,
    }
    assert Counter(record.template_id for record in records) == {
        "occupation_doctor": 4,
        "occupation_teacher": 4,
        "occupation_engineer": 4,
        "occupation_lawyer": 4,
        "description_town": 4,
        "description_work": 4,
        "description_neighbors": 4,
        "description_friends": 4,
        "aspiration_family": 4,
        "aspiration_community": 4,
        "achievement_proud": 4,
        "achievement_respected": 4,
    }


def test_prompt_bank_validation_rejects_unbalanced_templates() -> None:
    records = load_prompt_bank(settings.generation.prompt_bank_path)

    with pytest.raises(PromptBankValidationError):
        validate_prompt_bank(records[:-1])
