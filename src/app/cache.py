import json
from hashlib import sha256
from pathlib import Path

from .models import GenerationArtifactPaths
from .settings.generation import GenerationConfig


def _build_cache_decoding_payload(decoding: dict[str, object]) -> dict[str, object]:
    payload = {
        "strategy": decoding["strategy"],
        "do_sample": decoding["do_sample"],
    }

    for field in ("temperature", "top_k", "top_p"):
        value = decoding.get(field)
        if value is not None:
            payload[field] = value

    no_repeat_ngram_size = decoding.get("no_repeat_ngram_size")
    if isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size > 0:
        payload["no_repeat_ngram_size"] = no_repeat_ngram_size

    return payload


def build_cache_payload(config: GenerationConfig, prompt_bank_digest: str) -> dict[str, object]:
    return {
        "model_name": config.model_name,
        "prompt_bank_digest": prompt_bank_digest,
        "decoding": _build_cache_decoding_payload(config.decoding.to_dict()),
        "max_new_tokens": config.max_new_tokens,
        "n_samples_per_prompt": config.n_samples_per_prompt,
        "seeds": list(config.seeds),
    }


def compute_generation_cache_key(config: GenerationConfig, prompt_bank_digest: str) -> str:
    payload = build_cache_payload(config, prompt_bank_digest)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()[:20]


def build_artifact_paths(output_dir: Path, cache_key: str) -> GenerationArtifactPaths:
    root = output_dir.expanduser().resolve()
    return GenerationArtifactPaths(
        generations_path=root / "generations" / f"{cache_key}.parquet",
        manifest_path=root / "manifests" / f"{cache_key}.json",
    )
