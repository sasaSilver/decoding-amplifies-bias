from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest

from decoding_amplifies_bias.generation import GenerationRunner, GreedyGenerationBackend
from decoding_amplifies_bias.models import GeneratedText


@pytest.fixture
def fake_greedy_backend() -> MagicMock:
    """Fixture providing a mock backend for testing generation without loading real models."""
    backend = create_autospec(GreedyGenerationBackend)
    backend.model_name = "fake-gpt2"
    backend.device = "cpu"

    def generate_side_effect(prompt_text: str, max_new_tokens: int, seed: int) -> GeneratedText:
        completion_text = f" completion-for-seed-{seed}"
        return GeneratedText(
            raw_text=f"{prompt_text}{completion_text}",
            completion_text=completion_text,
        )

    backend.generate.side_effect = generate_side_effect
    return backend


@pytest.fixture
def generation_runner() -> GenerationRunner:
    """Fixture providing a GenerationRunner instance."""
    return GenerationRunner()
