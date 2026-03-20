from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.generation import GenerationRunner
from app.settings.generation import GenerationConfig, DecodingConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_prefix="",
        cli_enforce_required=True,
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    prompt_bank: Path = Field(
        default=Path(__file__).resolve().parents[3] / "data" / "prompt_bank_v1.csv",
        description="Path to the prompt bank CSV file.",
    )
    output_dir: Path = Field(
        default=Path(__file__).resolve().parents[3] / "outputs",
        description="Directory for output files.",
    )
    model_name: str = Field(
        default="gpt2",
        description="HuggingFace model name to use.",
    )
    max_new_tokens: int = Field(
        default=40,
        ge=1,
        description="Maximum number of tokens to generate.",
    )
    n_samples: int = Field(
        default=50,
        ge=1,
        description="Number of samples to generate per prompt.",
    )
    seeds: str = Field(
        default="0,1,2",
        description="Random seeds to use (comma-separated).",
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Device to run on (e.g., 'cuda', 'cpu').",
    )

    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        exclude=True,
    )
    decoding: DecodingConfig = Field(
        default_factory=DecodingConfig,
        exclude=True,
    )

    @model_validator(mode="after")
    def populate_generation_config(self) -> "Settings":
        """Populate generation config from CLI fields."""
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
    
    def cli_cmd(self) -> None:
        print(
        f"""
App settings:
{self.model_dump_json(indent=2)}
        """
        )
        
        result = GenerationRunner().run(self.generation)
        print(
            f"""
App results:
{result.model_dump_json()}
            """
        )


settings = Settings()
