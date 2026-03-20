import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .cache import (
    build_artifact_paths,
    build_cache_payload,
    compute_generation_cache_key,
)
from .device import resolve_torch_device
from .models import (
    GeneratedText,
    GenerationRecord,
    GenerationRunResult,
)
from .prompt_bank import load_prompt_bank, prompt_bank_digest
from .settings.generation import DecodingConfig, GenerationConfig


class GenerationBackend(Protocol):
    model_name: str
    device: str

    def generate_batch(
        self,
        prompt_texts: list[str],
        max_new_tokens: int,
        seed: int,
        decoding: DecodingConfig,
    ) -> list[GeneratedText]:
        """Generate continuations for a batch of prompts."""
        ...


class GPT2GenerationBackend:
    def __init__(self, model_name: str = "gpt2", device: str | None = None) -> None:
        resolved_device = resolve_torch_device(device)
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

    def generate_batch(
        self,
        prompt_texts: list[str],
        max_new_tokens: int,
        seed: int,
        decoding: DecodingConfig,
    ) -> list[GeneratedText]:
        if not prompt_texts:
            return []

        self._set_seed(seed)
        encoded = self._tokenizer(prompt_texts, padding=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        generation_kwargs = decoding.to_generation_kwargs()

        with self._torch.no_grad():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                **generation_kwargs,
            )

        results: list[GeneratedText] = []
        for index, generated_ids in enumerate(output_ids):
            prompt_length = int(prompt_lengths[index])
            completion_ids = generated_ids[prompt_length:]
            raw_text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            completion_text = self._tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.append(GeneratedText(raw_text=raw_text, completion_text=completion_text))

        return results

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int,
        seed: int,
        decoding: DecodingConfig,
    ) -> GeneratedText:
        return self.generate_batch(
            prompt_texts=[prompt_text],
            max_new_tokens=max_new_tokens,
            seed=seed,
            decoding=decoding,
        )[0]


GPT2GreedyBackend = GPT2GenerationBackend
GreedyGenerationBackend = GenerationBackend


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
        "decoding": config.decoding.to_dict(),
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
        backend: GenerationBackend | None = None,
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

        active_backend = backend or GPT2GenerationBackend(
            model_name=config.model_name,
            device=config.device,
        )
        if active_backend.model_name != config.model_name:
            raise ValueError("Backend model_name must match GenerationConfig.model_name.")

        environment = _collect_environment_snapshot(active_backend.device)
        records: list[dict[str, str | bool | int | float | None]] = []
        for seed in config.seeds:
            for prompt in prompts:
                if config.decoding.do_sample:
                    generated_texts = active_backend.generate_batch(
                        prompt_texts=[prompt.prompt_text] * config.n_samples_per_prompt,
                        max_new_tokens=config.max_new_tokens,
                        seed=seed,
                        decoding=config.decoding,
                    )
                    if len(generated_texts) != config.n_samples_per_prompt:
                        raise ValueError(
                            "Backend must return one generated text per requested sample."
                        )
                else:
                    generated_text = active_backend.generate_batch(
                        prompt_texts=[prompt.prompt_text],
                        max_new_tokens=config.max_new_tokens,
                        seed=seed,
                        decoding=config.decoding,
                    )
                    if len(generated_text) != 1:
                        raise ValueError("Greedy decoding must return exactly one generated text.")
                    generated_texts = generated_text * config.n_samples_per_prompt

                for sample_index, generated in enumerate(generated_texts):
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
                        temperature=config.decoding.temperature,
                        top_k=config.decoding.top_k,
                        top_p=config.decoding.top_p,
                        no_repeat_ngram_size=config.decoding.no_repeat_ngram_size,
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
