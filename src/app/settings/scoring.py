from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class ScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_root: Path = Path(__file__).resolve().parents[3]
    generations_path: Path = repo_root / "outputs" / "generations"
    output_dir: Path = repo_root / "outputs"

    model_name: str = "sasha/regardv3"
    model_path: Path | None = None
    use_masking: bool = True
    local_files_only: bool = False
    low_cpu_mem_usage: bool = True
    batch_size: int = 32
    device: str | None = None

    @field_validator("generations_path", "output_dir", "model_path", mode="before")
    @classmethod
    def expand_path(cls, v: Path | str | None) -> Path | None:
        if v is None:
            return None
        return Path(v).expanduser()

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_name must not be blank.")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be at least 1.")
        return v

    def resolved_model_reference(self) -> str:
        if self.model_path is not None:
            return str(self.model_path.resolve())

        model_path = Path(self.model_name).expanduser()
        if model_path.exists():
            return str(model_path.resolve())

        return self.model_name
