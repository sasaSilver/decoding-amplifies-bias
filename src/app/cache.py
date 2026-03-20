import json
from hashlib import sha256
from pathlib import Path

from .models import GenerationArtifactPaths
from .settings.generation import GenerationConfig


def build_cache_payload(config: GenerationConfig, prompt_bank_digest: str) -> dict[str, object]:
    return {
        "model_name": config.model_name,
        "prompt_bank_digest": prompt_bank_digest,
        "decoding": config.decoding.model_dump(),
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
