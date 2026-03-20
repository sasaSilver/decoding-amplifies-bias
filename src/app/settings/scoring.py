from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class ScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_root: Path = Path(__file__).resolve().parents[3]
    generations_path: Path = repo_root / "outputs" / "generations"
    output_dir: Path = repo_root / "outputs"

    model_name: str = "sasha/regardv3"
    use_masking: bool = True
    device: str | None = None

    @field_validator("generations_path", "output_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Path | str) -> Path:
        return Path(v).expanduser()

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_name must not be blank.")
        return v
