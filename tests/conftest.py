from unittest.mock import MagicMock, create_autospec

import pytest

from app.generation import GenerationBackend, GenerationRunner
from app.models import GeneratedText
from app.settings.generation import DecodingConfig


@pytest.fixture
def fake_greedy_backend() -> MagicMock:
    """Fixture providing a mock backend for testing generation without loading real models."""
    backend = create_autospec(GenerationBackend)
    backend.model_name = "fake-gpt2"
    backend.device = "cpu"

    def generate_batch_side_effect(
        prompt_texts: list[str],
        max_new_tokens: int,
        seed: int,
        decoding: DecodingConfig,
    ) -> list[GeneratedText]:
        strategy_suffix = decoding.strategy
        results = []
        for sample_index, prompt_text in enumerate(prompt_texts):
            completion_text = f" completion-for-seed-{seed}-sample-{sample_index}"
            results.append(
                GeneratedText(
                    raw_text=f"{prompt_text}{completion_text}-{strategy_suffix}",
                    completion_text=f"{completion_text}-{strategy_suffix}",
                )
            )
        return results

    backend.generate_batch.side_effect = generate_batch_side_effect
    return backend


@pytest.fixture
def generation_runner() -> GenerationRunner:
    """Fixture providing a GenerationRunner instance."""
    return GenerationRunner()
