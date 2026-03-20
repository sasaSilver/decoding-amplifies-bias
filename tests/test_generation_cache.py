import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.app.cache import compute_generation_cache_key
from src.app.generation import GenerationRunner
from src.app.prompt_bank import (
    load_prompt_bank,
    prompt_bank_digest,
)
from src.app.settings.settings import GenerationConfig, settings


def test_generation_cache_key_and_outputs_are_stable(
    tmp_path: Path,
    fake_greedy_backend: MagicMock,
    generation_runner: GenerationRunner,
) -> None:
    prompt_records = load_prompt_bank(settings.generation.prompt_bank_path)
    digest = prompt_bank_digest(prompt_records)
    config = GenerationConfig(
        prompt_bank_path=settings.generation.prompt_bank_path,
        output_dir=tmp_path,
        model_name="fake-gpt2",
        max_new_tokens=8,
        n_samples_per_prompt=2,
        seeds=(11, 29),
    )

    key_once = compute_generation_cache_key(config, digest)
    key_twice = compute_generation_cache_key(config, digest)
    assert key_once == key_twice

    first_result = generation_runner.run(config, backend=fake_greedy_backend)

    assert first_result.cache_key == key_once
    assert first_result.from_cache is False
    assert fake_greedy_backend.generate.call_count == len(prompt_records) * len(config.seeds)

    frame = pd.read_parquet(first_result.generations_path)
    assert (
        int(frame.shape[0]) == len(prompt_records) * len(config.seeds) * config.n_samples_per_prompt
    )
    assert set(frame["sample_index"].tolist()) == {0, 1}

    manifest = json.loads(first_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["cache_key"] == key_once
    assert manifest["decoding"] == {"strategy": "greedy", "do_sample": False}
    assert manifest["seeds"] == [11, 29]

    second_result = generation_runner.run(config, backend=fake_greedy_backend)
    assert second_result.cache_key == key_once
    assert second_result.from_cache is True
    assert fake_greedy_backend.generate.call_count == len(prompt_records) * len(config.seeds)
