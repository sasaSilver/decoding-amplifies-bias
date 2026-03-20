from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PromptRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_id: str
    template_id: str
    prompt_type: str
    demographic: str
    prompt_text: str


class GeneratedText(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_text: str
    completion_text: str


class GenerationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_key: str
    model_name: str
    prompt_id: str
    template_id: str
    prompt_type: str
    demographic: str
    prompt_text: str
    decoding_strategy: str
    do_sample: bool
    seed: int
    max_new_tokens: int
    sample_index: int
    raw_text: str
    completion_text: str


class GenerationArtifactPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    generations_path: Path
    manifest_path: Path


class GenerationRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_key: str
    generations_path: Path
    manifest_path: Path
    record_count: int
    from_cache: bool
