from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from decoding_amplifies_bias.cache import (
    build_artifact_paths,
    build_cache_payload,
    compute_generation_cache_key,
)
from decoding_amplifies_bias.models import (
    GeneratedText,
    GenerationConfig,
    GenerationRecord,
    GenerationRunResult,
)
from decoding_amplifies_bias.prompt_bank import load_prompt_bank, prompt_bank_digest


class GreedyGenerationBackend(Protocol):
    model_name: str
    device: str

    def generate(self, prompt_text: str, max_new_tokens: int, seed: int) -> GeneratedText:
        """Generate a single greedy continuation for a prompt."""
        ...


class GPT2GreedyBackend:
    def __init__(self, model_name: str = "gpt2", device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(resolved_device)
        model.eval()

        self.model_name = model_name
        self.device = resolved_device
        self._model: Any = model
        self._tokenizer: Any = tokenizer
        self._set_seed = set_seed
        self._torch = torch

    def generate(self, prompt_text: str, max_new_tokens: int, seed: int) -> GeneratedText:
        self._set_seed(seed)
        encoded = self._tokenizer(prompt_text, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])

        with self._torch.no_grad():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0]
        completion_ids = generated_ids[prompt_length:]
        raw_text = self._tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        completion_text = self._tokenizer.decode(
            completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return GeneratedText(raw_text=raw_text, completion_text=completion_text)


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _collect_environment_snapshot(device: str) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "pandas": _package_version("pandas"),
        "pyarrow": _package_version("pyarrow"),
        "device": device,
    }


def _build_manifest(
    config: GenerationConfig,
    cache_key: str,
    prompt_bank_digest_value: str,
    prompt_count: int,
    generations_path: Path,
    environment: dict[str, str],
    record_count: int,
) -> dict[str, object]:
    return {
        "run_id": cache_key,
        "cache_key": cache_key,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_name": config.model_name,
        "prompt_bank_path": str(config.prompt_bank_path.resolve()),
        "prompt_bank_digest": prompt_bank_digest_value,
        "prompt_count": prompt_count,
        "seeds": list(config.seeds),
        "max_new_tokens": config.max_new_tokens,
        "n_samples_per_prompt": config.n_samples_per_prompt,
        "decoding": config.decoding.model_dump(),
        "cache_payload": build_cache_payload(config, prompt_bank_digest_value),
        "environment": environment,
        "record_count": record_count,
        "artifacts": {
            "generations_path": str(generations_path),
        },
        "ethics_notice": (
            "Generated text may contain offensive content. Do not publish or commit large raw dumps."
        ),
    }


def _count_records(generations_path: Path) -> int:
    frame = pd.read_parquet(generations_path, columns=["prompt_id"])
    return int(frame.shape[0])


class GenerationRunner:
    def run(
        self,
        config: GenerationConfig,
        backend: GreedyGenerationBackend | None = None,
    ) -> GenerationRunResult:
        prompts = load_prompt_bank(config.prompt_bank_path)
        digest = prompt_bank_digest(prompts)
        cache_key = compute_generation_cache_key(config, digest)
        artifact_paths = build_artifact_paths(config.output_dir, cache_key)

        if artifact_paths.generations_path.exists() and artifact_paths.manifest_path.exists():
            return GenerationRunResult(
                cache_key=cache_key,
                generations_path=artifact_paths.generations_path,
                manifest_path=artifact_paths.manifest_path,
                record_count=_count_records(artifact_paths.generations_path),
                from_cache=True,
            )

        active_backend = backend or GPT2GreedyBackend(
            model_name=config.model_name,
            device=config.device,
        )
        if active_backend.model_name != config.model_name:
            raise ValueError("Backend model_name must match GenerationConfig.model_name.")

        environment = _collect_environment_snapshot(active_backend.device)
        records: list[dict[str, str | bool | int]] = []
        for seed in config.seeds:
            for prompt in prompts:
                generated = active_backend.generate(
                    prompt_text=prompt.prompt_text,
                    max_new_tokens=config.max_new_tokens,
                    seed=seed,
                )
                for sample_index in range(config.n_samples_per_prompt):
                    record = GenerationRecord(
                        cache_key=cache_key,
                        model_name=config.model_name,
                        prompt_id=prompt.prompt_id,
                        template_id=prompt.template_id,
                        prompt_type=prompt.prompt_type,
                        demographic=prompt.demographic,
                        prompt_text=prompt.prompt_text,
                        decoding_strategy=config.decoding.strategy,
                        do_sample=config.decoding.do_sample,
                        seed=seed,
                        max_new_tokens=config.max_new_tokens,
                        sample_index=sample_index,
                        raw_text=generated.raw_text,
                        completion_text=generated.completion_text,
                    )
                    records.append(record.model_dump())

        frame = pd.DataFrame.from_records(records)
        artifact_paths.generations_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(artifact_paths.generations_path, index=False)

        manifest = _build_manifest(
            config=config,
            cache_key=cache_key,
            prompt_bank_digest_value=digest,
            prompt_count=len(prompts),
            generations_path=artifact_paths.generations_path,
            environment=environment,
            record_count=len(records),
        )
        artifact_paths.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return GenerationRunResult(
            cache_key=cache_key,
            generations_path=artifact_paths.generations_path,
            manifest_path=artifact_paths.manifest_path,
            record_count=len(records),
            from_cache=False,
        )
