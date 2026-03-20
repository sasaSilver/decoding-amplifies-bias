import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from app.cache import compute_generation_cache_key
from app.generation import GenerationRunner
from app.prompt_bank import (
    load_prompt_bank,
    prompt_bank_digest,
)
from app.settings.generation import DecodingConfig, build_week3_decoding_grid
from app.settings.settings import GenerationConfig, settings


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
        decoding=DecodingConfig(strategy="temperature", temperature=0.7),
    )

    key_once = compute_generation_cache_key(config, digest)
    key_twice = compute_generation_cache_key(config, digest)
    assert key_once == key_twice

    first_result = generation_runner.run(config, backend=fake_greedy_backend)

    assert first_result.cache_key == key_once
    assert first_result.from_cache is False
    assert fake_greedy_backend.generate_batch.call_count == len(prompt_records) * len(config.seeds)

    frame = pd.read_parquet(first_result.generations_path)
    assert (
        int(frame.shape[0]) == len(prompt_records) * len(config.seeds) * config.n_samples_per_prompt
    )
    assert set(frame["sample_index"].tolist()) == {0, 1}
    assert set(frame["decoding_strategy"].tolist()) == {"temperature"}
    assert set(frame["temperature"].dropna().tolist()) == {0.7}
    assert set(frame["do_sample"].tolist()) == {True}

    manifest = json.loads(first_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["cache_key"] == key_once
    assert manifest["decoding"] == {
        "strategy": "temperature",
        "do_sample": True,
        "temperature": 0.7,
        "top_k": None,
        "top_p": None,
        "no_repeat_ngram_size": 0,
    }
    assert manifest["seeds"] == [11, 29]

    second_result = generation_runner.run(config, backend=fake_greedy_backend)
    assert second_result.cache_key == key_once
    assert second_result.from_cache is True
    assert fake_greedy_backend.generate_batch.call_count == len(prompt_records) * len(config.seeds)


def test_generation_cache_key_changes_with_decoding_config() -> None:
    prompt_records = load_prompt_bank(settings.generation.prompt_bank_path)
    digest = prompt_bank_digest(prompt_records)

    greedy = GenerationConfig(
        prompt_bank_path=settings.generation.prompt_bank_path,
        output_dir=Path("/tmp"),
        model_name="fake-gpt2",
        decoding=DecodingConfig(strategy="greedy"),
    )
    temperature = GenerationConfig(
        prompt_bank_path=settings.generation.prompt_bank_path,
        output_dir=Path("/tmp"),
        model_name="fake-gpt2",
        decoding=DecodingConfig(strategy="temperature", temperature=0.7),
    )

    greedy_key = compute_generation_cache_key(greedy, digest)
    temperature_key = compute_generation_cache_key(temperature, digest)

    assert greedy_key != temperature_key


def test_generation_cache_key_is_backward_compatible_for_greedy() -> None:
    prompt_records = load_prompt_bank(settings.generation.prompt_bank_path)
    digest = prompt_bank_digest(prompt_records)

    config = GenerationConfig(
        prompt_bank_path=settings.generation.prompt_bank_path,
        output_dir=Path("/tmp"),
        model_name="gpt2",
        max_new_tokens=40,
        n_samples_per_prompt=50,
        seeds=(0, 1, 2),
        decoding=DecodingConfig(strategy="greedy"),
    )

    assert compute_generation_cache_key(config, digest) == "e64c237a9d1c5330a8ce"


def test_greedy_generation_reuses_single_output_per_prompt_seed(
    tmp_path: Path,
    fake_greedy_backend: MagicMock,
    generation_runner: GenerationRunner,
) -> None:
    prompt_records = load_prompt_bank(settings.generation.prompt_bank_path)
    config = GenerationConfig(
        prompt_bank_path=settings.generation.prompt_bank_path,
        output_dir=tmp_path,
        model_name="fake-gpt2",
        max_new_tokens=8,
        n_samples_per_prompt=3,
        seeds=(7, 13),
        decoding=DecodingConfig(strategy="greedy"),
    )

    result = generation_runner.run(config, backend=fake_greedy_backend)

    assert result.from_cache is False
    assert fake_greedy_backend.generate_batch.call_count == len(prompt_records) * len(config.seeds)

    frame = pd.read_parquet(result.generations_path)
    assert int(frame.shape[0]) == len(prompt_records) * len(config.seeds) * 3

    grouped = frame.groupby(["prompt_id", "seed"])["completion_text"].nunique()
    assert grouped.eq(1).all()


def test_week3_decoding_grid_matches_requirements() -> None:
    grid = build_week3_decoding_grid()

    assert len(grid) == 10
    assert grid[0].to_dict() == {
        "strategy": "greedy",
        "do_sample": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "no_repeat_ngram_size": 0,
    }

    configs = {(config.strategy, config.temperature, config.top_k, config.top_p) for config in grid}
    assert ("temperature", 0.7, None, None) in configs
    assert ("temperature", 1.0, None, None) in configs
    assert ("temperature", 1.3, None, None) in configs
    assert ("top_k", None, 20, None) in configs
    assert ("top_k", None, 50, None) in configs
    assert ("top_k", None, 100, None) in configs
    assert ("top_p", None, None, 0.8) in configs
    assert ("top_p", None, None, 0.9) in configs
    assert ("top_p", None, None, 0.95) in configs
