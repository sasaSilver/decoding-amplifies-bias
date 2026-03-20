from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class DecodingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Literal["greedy"] = "greedy"
    do_sample: bool = False


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

    @model_validator(mode="after")
    def validate_decoding(self) -> "GenerationConfig":
        if self.decoding.strategy != "greedy" or self.decoding.do_sample:
            raise ValueError("Week 1 only supports greedy decoding.")
        return self