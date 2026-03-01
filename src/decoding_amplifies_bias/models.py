from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from decoding_amplifies_bias.paths import DEFAULT_OUTPUT_DIR, DEFAULT_PROMPT_BANK_PATH


class PromptRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_id: str
    template_id: str
    prompt_type: str
    demographic: str
    prompt_text: str


class GreedyDecodingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Literal["greedy"] = "greedy"
    do_sample: bool = False


class GenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_bank_path: Path = DEFAULT_PROMPT_BANK_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    model_name: str = "gpt2"
    max_new_tokens: int = 40
    n_samples_per_prompt: int = 50
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str | None = None
    decoding: GreedyDecodingConfig = GreedyDecodingConfig()

    @field_validator("prompt_bank_path", "output_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Path | str) -> Path:
        return Path(v).expanduser()

    @field_validator("seeds", mode="before")
    @classmethod
    def ensure_tuple(cls, v: list[int] | tuple[int, ...]) -> tuple[int, ...]:
        return tuple(v)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_name must not be blank.")
        return v

    @field_validator("max_new_tokens")
    @classmethod
    def validate_max_new_tokens(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        return v

    @field_validator("n_samples_per_prompt")
    @classmethod
    def validate_n_samples_per_prompt(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_samples_per_prompt must be at least 1.")
        return v

    @model_validator(mode="after")
    def validate_seeds(self) -> GenerationConfig:
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique.")
        return self

    @model_validator(mode="after")
    def validate_decoding(self) -> GenerationConfig:
        if self.decoding.strategy != "greedy" or self.decoding.do_sample:
            raise ValueError("Week 1 only supports greedy decoding.")
        return self


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
