from __future__ import annotations

import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict

from .config import ExAIBenchmarkConfig
from .constants import EXAI_LABELS
from .utils import canonical_json_digest, file_digest, utc_now_iso, write_json


class BenchmarkBuildError(ValueError):
    """Raised when the explanation benchmark cannot be built reproducibly."""


class BenchmarkExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    source_score_path: str
    source_score_manifest_path: str
    source_generation_path: str | None = None
    source_generations_cache_key: str | None = None
    source_row_index: int
    score_cache_key: str
    prompt_id: str
    prompt_type: str
    demographic: str
    predicted_label: str
    scoring_masked: bool
    decoding_strategy: str
    seed: int
    sample_index: int
    completion_text: str


class BenchmarkBuildResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_path: Path
    manifest_path: Path
    record_count: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_source_manifest(repo_root: Path) -> Path:
    manifests_dir = repo_root / "outputs" / "manifests"
    combined_manifests = sorted(manifests_dir.glob("*_week3_combined.json"))
    if combined_manifests:
        return max(combined_manifests, key=lambda path: path.stat().st_mtime)

    score_manifests = [
        path
        for path in sorted(manifests_dir.glob("*.json"))
        if not path.name.endswith("_week3_combined.json")
    ]
    if score_manifests:
        return max(score_manifests, key=lambda path: path.stat().st_mtime)

    raise BenchmarkBuildError(f"No score manifests found under {manifests_dir}")


def resolve_score_sources(config: ExAIBenchmarkConfig) -> list[Path]:
    manifest_path = config.source_manifest_path
    if manifest_path is None and config.source_scores_path is not None:
        candidate_manifest = (
            config.repo_root / "outputs" / "manifests" / f"{config.source_scores_path.stem}.json"
        )
        manifest_path = candidate_manifest if candidate_manifest.exists() else None

    if manifest_path is None:
        manifest_path = _default_source_manifest(config.repo_root)

    resolved_manifest = manifest_path.expanduser().resolve()
    payload = _read_json(resolved_manifest)
    if "created_from_scores" in payload:
        score_paths = [
            Path(value).expanduser().resolve() for value in payload["created_from_scores"]
        ]
    elif config.source_scores_path is not None:
        score_paths = [config.source_scores_path.expanduser().resolve()]
    elif "artifacts" in payload and "scores_path" in payload["artifacts"]:
        score_paths = [Path(payload["artifacts"]["scores_path"]).expanduser().resolve()]
    else:
        raise BenchmarkBuildError(
            f"Could not resolve score paths from manifest {resolved_manifest}."
        )

    for score_path in score_paths:
        if not score_path.exists():
            raise FileNotFoundError(f"Score file does not exist: {score_path}")
    return score_paths


def _row_selection_hash(row: pd.Series, seed: int) -> str:
    payload = (
        f"{seed}|{row['source_score_path']}|{row['prompt_id']}|{row['demographic']}|"
        f"{row['seed']}|{row['sample_index']}|{row['completion_text']}"
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def load_benchmark_candidates(score_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for score_path in score_paths:
        manifest_path = score_path.parent.parent / "manifests" / f"{score_path.stem}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing score manifest for {score_path}")

        manifest = _read_json(manifest_path)
        frame = pd.read_parquet(score_path).copy()
        frame["source_score_path"] = str(score_path.resolve())
        frame["source_score_manifest_path"] = str(manifest_path.resolve())
        frame["source_generation_path"] = manifest.get("generations_path")
        frame["source_generations_cache_key"] = manifest.get("generations_cache_key")
        frame["source_row_index"] = range(len(frame))
        frame["score_cache_key"] = manifest.get("cache_key", score_path.stem)
        frames.append(frame)

    if not frames:
        raise BenchmarkBuildError("No score frames were loaded.")

    combined = pd.concat(frames, ignore_index=True)
    required_columns = {
        "prompt_id",
        "prompt_type",
        "demographic",
        "regard_label",
        "scoring_masked",
        "decoding_strategy",
        "seed",
        "sample_index",
        "completion_text",
        "source_score_path",
        "source_score_manifest_path",
        "score_cache_key",
        "source_row_index",
    }
    missing = sorted(required_columns - set(combined.columns))
    if missing:
        raise BenchmarkBuildError(f"Benchmark candidates are missing columns: {missing}")
    return combined


def select_benchmark_rows(
    candidates: pd.DataFrame,
    *,
    examples_per_label: int,
    selection_seed: int,
) -> pd.DataFrame:
    if candidates.empty:
        raise BenchmarkBuildError("Candidate frame is empty.")

    working = candidates.copy()
    working["selection_hash"] = working.apply(
        lambda row: _row_selection_hash(row, selection_seed),
        axis=1,
    )
    selected_rows: list[pd.Series] = []
    demographic_counts: Counter[str] = Counter()
    prompt_type_counts: Counter[str] = Counter()

    for label in EXAI_LABELS:
        label_frame = working[working["regard_label"] == label].copy()
        if label_frame.empty:
            continue

        demographic_groups: dict[str, list[int]] = defaultdict(list)
        label_columns = list(label_frame.columns)
        label_records = [
            dict(zip(label_columns, values, strict=False))
            for values in label_frame.itertuples(index=False, name=None)
        ]
        label_records = cast(list[dict[str, Any]], label_records)
        label_records = sorted(label_records, key=lambda row: str(row["selection_hash"]))
        for row_index, row in enumerate(label_records):
            demographic_groups[str(row["demographic"])].append(row_index)

        chosen_indices: set[int] = set()
        while len(chosen_indices) < min(examples_per_label, len(label_frame)):
            available_demographics = [
                demographic
                for demographic, indices in demographic_groups.items()
                if any(index not in chosen_indices for index in indices)
            ]
            if not available_demographics:
                break

            selected_demographic = min(
                available_demographics,
                key=lambda demographic: (
                    demographic_counts[demographic],
                    sha256(f"{selection_seed}:{label}:{demographic}".encode()).hexdigest(),
                ),
            )
            candidate_indices = [
                index
                for index in demographic_groups[selected_demographic]
                if index not in chosen_indices
            ]
            selected_index = min(
                candidate_indices,
                key=lambda index: (
                    prompt_type_counts[str(label_records[index]["prompt_type"])],
                    str(label_records[index]["selection_hash"]),
                ),
            )
            chosen_indices.add(selected_index)

            selected_row = label_records[selected_index]
            selected_rows.append(pd.Series(selected_row))
            demographic_counts[str(selected_row["demographic"])] += 1
            prompt_type_counts[str(selected_row["prompt_type"])] += 1

    if not selected_rows:
        raise BenchmarkBuildError("No benchmark rows were selected.")

    selected = pd.DataFrame(selected_rows).copy()
    return selected.sort_values(["regard_label", "demographic", "selection_hash"]).reset_index(
        drop=True
    )


def _build_benchmark_examples(selected_rows: pd.DataFrame) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    selected_records = cast(list[dict[str, Any]], selected_rows.to_dict(orient="records"))
    for row in selected_records:
        payload = {
            "source_score_path": row["source_score_path"],
            "source_row_index": int(row["source_row_index"]),
            "prompt_id": row["prompt_id"],
            "seed": int(row["seed"]),
            "sample_index": int(row["sample_index"]),
        }
        benchmark_id = canonical_json_digest(payload)[:20]
        examples.append(
            BenchmarkExample(
                benchmark_id=benchmark_id,
                source_score_path=str(row["source_score_path"]),
                source_score_manifest_path=str(row["source_score_manifest_path"]),
                source_generation_path=(
                    str(row["source_generation_path"])
                    if row["source_generation_path"] is not None
                    and pd.notna(row["source_generation_path"])
                    else None
                ),
                source_generations_cache_key=(
                    str(row["source_generations_cache_key"])
                    if row["source_generations_cache_key"] is not None
                    and pd.notna(row["source_generations_cache_key"])
                    else None
                ),
                source_row_index=int(row["source_row_index"]),
                score_cache_key=str(row["score_cache_key"]),
                prompt_id=str(row["prompt_id"]),
                prompt_type=str(row["prompt_type"]),
                demographic=str(row["demographic"]),
                predicted_label=str(row["regard_label"]),
                scoring_masked=bool(row["scoring_masked"]),
                decoding_strategy=str(row["decoding_strategy"]),
                seed=int(row["seed"]),
                sample_index=int(row["sample_index"]),
                completion_text=str(row["completion_text"]),
            )
        )
    return examples


def build_explanation_benchmark(config: ExAIBenchmarkConfig) -> BenchmarkBuildResult:
    paths = config.output_paths.ensure_dirs()
    score_paths = resolve_score_sources(config)
    candidates = load_benchmark_candidates(score_paths)
    selected_rows = select_benchmark_rows(
        candidates,
        examples_per_label=config.examples_per_label,
        selection_seed=config.selection_seed,
    )
    examples = _build_benchmark_examples(selected_rows)
    benchmark_key = canonical_json_digest([example.model_dump() for example in examples])[:20]
    benchmark_path = paths.benchmark_dir / f"benchmark_{benchmark_key}.parquet"
    manifest_path = paths.benchmark_dir / f"benchmark_{benchmark_key}.json"

    benchmark_frame = pd.DataFrame([example.model_dump() for example in examples])
    benchmark_frame.to_parquet(benchmark_path, index=False)

    label_counts = Counter(example.predicted_label for example in examples)
    demographic_counts = Counter(example.demographic for example in examples)
    prompt_type_counts = Counter(example.prompt_type for example in examples)
    manifest_payload = {
        "created_at_utc": utc_now_iso(),
        "benchmark_key": benchmark_key,
        "record_count": len(examples),
        "examples_per_label": config.examples_per_label,
        "selection_seed": config.selection_seed,
        "source_scores": [
            {
                "path": str(path.resolve()),
                "sha256": file_digest(path),
            }
            for path in score_paths
        ],
        "label_counts": {label: label_counts.get(label, 0) for label in EXAI_LABELS},
        "demographic_counts": dict(sorted(demographic_counts.items())),
        "prompt_type_counts": dict(sorted(prompt_type_counts.items())),
        "artifacts": {
            "benchmark_path": str(benchmark_path.resolve()),
        },
        "ethics_notice": (
            "Benchmark texts may contain offensive content. Keep excerpts minimal and do not "
            "publish raw dumps."
        ),
        "selected_example_ids": [example.benchmark_id for example in examples],
    }
    write_json(manifest_path, manifest_payload)
    return BenchmarkBuildResult(
        benchmark_path=benchmark_path,
        manifest_path=manifest_path,
        record_count=len(examples),
    )
