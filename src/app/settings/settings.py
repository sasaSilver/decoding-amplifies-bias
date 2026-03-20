from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.app.settings.generation import DecodingConfig, GenerationConfig
from src.app.settings.scoring import ScoringConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    prompt_bank: Path = Path(__file__).resolve().parents[3] / "data" / "prompt_bank_v1.csv"
    output_dir: Path = Path(__file__).resolve().parents[3] / "outputs"
    model_name: str = "gpt2"
    max_new_tokens: int = 40
    n_samples: int = 50
    seeds: str = "0,1,2"
    device: Literal["cpu", "cuda"] = "cpu"

    generations_path: Path = Path(__file__).resolve().parents[3] / "outputs" / "generations"
    scoring_model: str = "sasha/regardv3"
    use_masking: bool = True
    n_bootstrap: int = 1000
    ci_level: float = 0.95
    n_spot_check: int = 20

    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        exclude=True,
    )
    decoding: DecodingConfig = Field(
        default_factory=DecodingConfig,
        exclude=True,
    )
    scoring: ScoringConfig = Field(
        default_factory=ScoringConfig,
        exclude=True,
    )

    @model_validator(mode="after")
    def populate_generation_config(self) -> "Settings":
        self.generation = GenerationConfig(
            prompt_bank_path=self.prompt_bank,
            output_dir=self.output_dir,
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            n_samples_per_prompt=self.n_samples,
            seeds=tuple(int(s.strip()) for s in self.seeds.split(",")),
            device=self.device,
            decoding=self.decoding,
        )
        return self

    @model_validator(mode="after")
    def populate_scoring_config(self) -> "Settings":
        self.scoring = ScoringConfig(
            generations_path=self.generations_path,
            output_dir=self.output_dir,
            model_name=self.scoring_model,
            use_masking=self.use_masking,
            device=self.device,
        )
        return self


settings = Settings()
