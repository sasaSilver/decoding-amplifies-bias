from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DecodingStrategy = Literal["greedy", "temperature", "top_k", "top_p"]


class DecodingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: DecodingStrategy = "greedy"
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    no_repeat_ngram_size: int = 0

    @property
    def do_sample(self) -> bool:
        return self.strategy != "greedy"

    def to_dict(self) -> dict[str, str | bool | float | int | None]:
        return {
            "strategy": self.strategy,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }

    def to_generation_kwargs(self) -> dict[str, bool | float | int]:
        kwargs: dict[str, bool | float | int] = {
            "do_sample": self.do_sample,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size

        return kwargs

    @model_validator(mode="after")
    def validate_strategy_parameters(self) -> "DecodingConfig":
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative.")

        if self.strategy == "greedy":
            if self.temperature is not None or self.top_k is not None or self.top_p is not None:
                raise ValueError("Greedy decoding must not set temperature, top_k, or top_p.")
            return self

        if self.strategy == "temperature":
            if self.temperature is None or self.temperature <= 0:
                raise ValueError("Temperature decoding requires temperature > 0.")
            if self.top_k is not None or self.top_p is not None:
                raise ValueError("Temperature decoding must not set top_k or top_p.")
            return self

        if self.strategy == "top_k":
            if self.top_k is None or self.top_k < 1:
                raise ValueError("Top-k decoding requires top_k >= 1.")
            if self.temperature is not None or self.top_p is not None:
                raise ValueError("Top-k decoding must not set temperature or top_p.")
            return self

        if self.top_p is None or not 0 < self.top_p <= 1:
            raise ValueError("Top-p decoding requires top_p in (0, 1].")
        if self.temperature is not None or self.top_k is not None:
            raise ValueError("Top-p decoding must not set temperature or top_k.")
        return self


def build_week3_decoding_grid(
    *,
    include_greedy: bool = True,
    no_repeat_ngram_size: int = 0,
) -> list[DecodingConfig]:
    configs: list[DecodingConfig] = []

    if include_greedy:
        configs.append(DecodingConfig(strategy="greedy", no_repeat_ngram_size=no_repeat_ngram_size))

    for temperature in (0.7, 1.0, 1.3):
        configs.append(
            DecodingConfig(
                strategy="temperature",
                temperature=temperature,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        )

    for top_k in (20, 50, 100):
        configs.append(
            DecodingConfig(
                strategy="top_k",
                top_k=top_k,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        )

    for top_p in (0.8, 0.9, 0.95):
        configs.append(
            DecodingConfig(
                strategy="top_p",
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        )

    return configs


class GenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_root: Path = Path(__file__).resolve().parents[3]
    prompt_bank_path: Path = repo_root / "data" / "prompt_bank_v1.csv"
    output_dir: Path = repo_root / "outputs"

    model_name: str = "gpt2"
    max_new_tokens: int = 40
    n_samples_per_prompt: int = 50
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str | None = None
    decoding: DecodingConfig = DecodingConfig()

    @field_validator("prompt_bank_path", "output_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Path | str) -> Path:
        return Path(v).expanduser()

    @field_validator("seeds", mode="before")
    @classmethod
    def ensure_tuple(cls, v: list[int] | tuple[int, ...] | str) -> tuple[int, ...]:
        if isinstance(v, str):
            return tuple(int(s.strip()) for s in v.split(","))
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
    def validate_seeds(self) -> "GenerationConfig":
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique.")
        return self
