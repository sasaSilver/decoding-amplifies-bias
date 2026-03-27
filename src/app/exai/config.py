from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import EXAI_NAMESPACE


class ExAIPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_root: Path = Path(__file__).resolve().parents[3]
    root: Path = repo_root / "outputs" / EXAI_NAMESPACE

    @field_validator(
        "repo_root",
        "root",
        mode="before",
    )
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def benchmark_dir(self) -> Path:
        return self.root / "benchmark"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def explanations_dir(self) -> Path:
        return self.root / "explanations"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def ensure_dirs(self) -> ExAIPaths:
        for path in (
            self.root,
            self.manifests_dir,
            self.metadata_dir,
            self.models_dir,
            self.benchmark_dir,
            self.eval_dir,
            self.explanations_dir,
            self.figures_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


class ExAIDataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_root: Path = Path(__file__).resolve().parents[3]
    dataset_path: Path = repo_root / "data" / "regard"
    split_seed: int = 13
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    use_masking: bool = True
    output_paths: ExAIPaths = Field(default_factory=ExAIPaths)

    @field_validator("repo_root", "dataset_path", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @field_validator("train_fraction", "validation_fraction")
    @classmethod
    def validate_fraction(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("Split fractions must be in (0, 1).")
        return value

    @field_validator("split_seed")
    @classmethod
    def validate_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("split_seed must be non-negative.")
        return value

    @model_validator(mode="after")
    def validate_split_fractions(self) -> ExAIDataConfig:
        self.test_fraction()
        return self

    def test_fraction(self) -> float:
        remainder = 1.0 - self.train_fraction - self.validation_fraction
        if remainder <= 0:
            raise ValueError("train_fraction + validation_fraction must be < 1.")
        return remainder


class ExAITrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 3
    early_stopping_patience: int = 2
    seed: int = 13
    device: str = "auto"
    output_paths: ExAIPaths = Field(default_factory=ExAIPaths)

    @field_validator("max_length", "batch_size", "epochs", "early_stopping_patience")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Training integers must be at least 1.")
        return value

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("learning_rate must be positive.")
        return value


class ExAIBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_root: Path = Path(__file__).resolve().parents[3]
    source_scores_path: Path | None = None
    source_manifest_path: Path | None = None
    examples_per_label: int = 3
    selection_seed: int = 13
    output_paths: ExAIPaths = Field(default_factory=ExAIPaths)

    @field_validator("repo_root", "source_scores_path", "source_manifest_path", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    @field_validator("examples_per_label", "selection_seed")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Benchmark integers must be at least 1.")
        return value


class ExAIEvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_root: Path = Path(__file__).resolve().parents[3]
    checkpoint_path: Path
    benchmark_path: Path | None = None
    batch_size: int = 8
    max_length: int = 128
    device: str = "auto"
    compare_to_released: bool = True
    output_paths: ExAIPaths = Field(default_factory=ExAIPaths)

    @field_validator("repo_root", "checkpoint_path", "benchmark_path", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    @field_validator("batch_size", "max_length")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Evaluation integers must be at least 1.")
        return value
