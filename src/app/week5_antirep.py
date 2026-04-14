from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import compute_week3_metrics
from .settings.settings import Settings

IDENTITY_COLUMNS = [
    "decoding_strategy",
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
]
REGARD_COLUMNS = ["negative", "neutral", "positive", "other"]
QUALITY_COLUMNS = [
    "distinct_1",
    "distinct_2",
    "repeated_3gram_rate",
    "longest_repetition_span",
]
QUALITY_CI_COLUMNS = [
    "distinct_1_ci_lower",
    "distinct_1_ci_upper",
    "distinct_2_ci_lower",
    "distinct_2_ci_upper",
    "repeated_3gram_rate_ci_lower",
    "repeated_3gram_rate_ci_upper",
    "longest_repetition_span_ci_lower",
    "longest_repetition_span_ci_upper",
]
KEY_TRACE = {
    "prompt_type": "description",
    "group_a": "Black man",
    "group_b": "White woman",
}
BASELINE_STEM = "week5_baseline_reference"
ANTIREP_STEM = "week5_antirep_combined"


@dataclass(frozen=True)
class ScoreArtifact:
    score_path: Path
    score_manifest_path: Path
    generation_manifest_path: Path
    signature: str
    decoding: dict[str, object]


@dataclass(frozen=True)
class Week3Bundle:
    combined_scores_path: Path
    combined_manifest_path: Path
    regard_metrics_path: Path
    gap_metrics_path: Path
    quality_metrics_path: Path
    summary_path: Path
    score_files: list[Path]
    no_repeat_ngram_size: int


@dataclass(frozen=True)
class Week5ArtifactPaths:
    baseline_bundle: Week3Bundle
    antirep_bundle: Week3Bundle
    gap_comparison_path: Path
    quality_comparison_path: Path
    regard_comparison_path: Path
    key_trace_path: Path
    summary_path: Path
    quality_plot_path: Path
    gap_plot_path: Path
    final_summary_path: Path
    report_path: Path


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _report_date_str() -> str:
    return datetime.now(UTC).astimezone().strftime("%B %d, %Y").replace(" 0", " ")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _normalize_decoding_payload(decoding: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "strategy": decoding["strategy"],
        "do_sample": decoding["do_sample"],
    }
    for field in ("temperature", "top_k", "top_p"):
        value = decoding.get(field)
        if value is not None:
            payload[field] = value

    no_repeat_ngram_size = decoding.get("no_repeat_ngram_size")
    if isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size > 0:
        payload["no_repeat_ngram_size"] = no_repeat_ngram_size
    return payload


def _generation_signature_from_manifest(manifest: Mapping[str, object]) -> str:
    raw_cache_payload = manifest.get("cache_payload")
    cache_payload = raw_cache_payload if isinstance(raw_cache_payload, Mapping) else {}
    raw_decoding = cache_payload.get("decoding", manifest.get("decoding", {}))
    if not isinstance(raw_decoding, Mapping):
        raise ValueError("Generation manifest decoding payload must be a dictionary.")

    payload = {
        "model_name": cache_payload.get("model_name", manifest.get("model_name")),
        "prompt_bank_digest": cache_payload.get(
            "prompt_bank_digest",
            manifest.get("prompt_bank_digest"),
        ),
        "max_new_tokens": cache_payload.get("max_new_tokens", manifest.get("max_new_tokens")),
        "n_samples_per_prompt": cache_payload.get(
            "n_samples_per_prompt",
            manifest.get("n_samples_per_prompt"),
        ),
        "seeds": cache_payload.get("seeds", manifest.get("seeds")),
        "decoding": _normalize_decoding_payload(raw_decoding),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        numeric_value = float(value)
        if math.isnan(numeric_value):
            return None
        return numeric_value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered == "nan":
            return None
        return float(stripped)
    return None


def _coerce_int(value: object, default: int = 0) -> int:
    coerced = _coerce_float(value)
    if coerced is None:
        return default
    return int(coerced)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _require_float(value: object) -> float:
    coerced = _coerce_float(value)
    if coerced is None:
        raise ValueError(f"Expected numeric value, got {value!r}.")
    return coerced


def _decoding_from_generation_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    raw_decoding = manifest.get("decoding", {})
    decoding = raw_decoding if isinstance(raw_decoding, Mapping) else {}
    return {
        "strategy": str(decoding.get("strategy", "greedy")),
        "do_sample": _coerce_bool(decoding.get("do_sample", False)),
        "temperature": _coerce_float(decoding.get("temperature")),
        "top_k": _coerce_int(decoding.get("top_k"), default=0) or None,
        "top_p": _coerce_float(decoding.get("top_p")),
        "no_repeat_ngram_size": _coerce_int(decoding.get("no_repeat_ngram_size"), default=0),
    }


def _strategy_rank(strategy: str) -> int:
    return {
        "greedy": 0,
        "temperature": 1,
        "top_k": 2,
        "top_p": 3,
    }.get(strategy, 99)


def _config_order_from_values(
    strategy: str,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
) -> int:
    base_rank = _strategy_rank(strategy) * 100
    if strategy == "greedy":
        return base_rank
    if strategy == "temperature":
        return base_rank + int(round((temperature or 0.0) * 10))
    if strategy == "top_k":
        return base_rank + int(top_k or 0)
    return base_rank + int(round((top_p or 0.0) * 100))


def format_config_label(
    strategy: str,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
) -> str:
    if strategy == "greedy":
        return "Greedy"
    if strategy == "temperature":
        return f"Temperature {temperature:.1f}"
    if strategy == "top_k":
        return f"Top-k {top_k:d}"
    return f"Top-p {top_p:.2f}".rstrip("0").rstrip(".")


def _format_config_label_from_row(row: Mapping[str, object]) -> str:
    return format_config_label(
        strategy=str(row["decoding_strategy"]),
        temperature=_coerce_float(row.get("temperature")),
        top_k=_coerce_int(row.get("top_k"), default=0) or None,
        top_p=_coerce_float(row.get("top_p")),
    )


def _decoding_key(decoding: Mapping[str, object]) -> str:
    payload = {
        "decoding_strategy": str(decoding["decoding_strategy"]),
        "do_sample": _coerce_bool(decoding["do_sample"]),
        "temperature": _coerce_float(decoding.get("temperature")),
        "top_k": _coerce_int(decoding.get("top_k"), default=0) or None,
        "top_p": _coerce_float(decoding.get("top_p")),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _config_order_from_row(row: Mapping[str, object]) -> int:
    return _config_order_from_values(
        strategy=str(row["decoding_strategy"]),
        temperature=_coerce_float(row.get("temperature")),
        top_k=_coerce_int(row.get("top_k"), default=0) or None,
        top_p=_coerce_float(row.get("top_p")),
    )


def _prepare_decoding_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["decoding_key"] = prepared.apply(
        lambda row: _decoding_key(
            {
                "decoding_strategy": row["decoding_strategy"],
                "do_sample": row["do_sample"],
                "temperature": row["temperature"],
                "top_k": row["top_k"],
                "top_p": row["top_p"],
            }
        ),
        axis=1,
    )
    prepared["config_label"] = prepared.apply(_format_config_label_from_row, axis=1)
    prepared["config_order"] = prepared.apply(_config_order_from_row, axis=1)
    return prepared


def _rename_frame_columns(df: pd.DataFrame, rename_map: Mapping[str, str]) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [rename_map.get(str(column), str(column)) for column in renamed.columns]
    return renamed


def _dataframe_records(df: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in df.index.tolist():
        record: dict[str, object] = {}
        for column in df.columns.tolist():
            record[str(column)] = df.at[index, column]
        records.append(record)
    return records


def _collect_score_files_for_no_repeat(
    settings: Settings,
    no_repeat_ngram_size: int,
) -> list[ScoreArtifact]:
    manifests_dir = settings.output_dir / "manifests"
    scores_dir = settings.output_dir / "scores"
    model_reference = settings.scoring.resolved_model_reference()
    unique_files: dict[str, ScoreArtifact] = {}

    for score_path in sorted(scores_dir.glob("*.parquet")):
        if score_path.name.endswith("_week3_combined.parquet"):
            continue

        score_manifest_path = manifests_dir / f"{score_path.stem}.json"
        if not score_manifest_path.exists():
            continue

        score_manifest = _read_json_object(score_manifest_path)
        generations_cache_key = score_manifest.get("generations_cache_key")
        if not isinstance(generations_cache_key, str):
            continue
        if (
            str(score_manifest.get("model_reference", score_manifest.get("model_name")))
            != model_reference
        ):
            continue
        if _coerce_bool(score_manifest.get("use_masking")) != settings.scoring.use_masking:
            continue

        generation_manifest_path = manifests_dir / f"{generations_cache_key}.json"
        if not generation_manifest_path.exists():
            continue

        generation_manifest = _read_json_object(generation_manifest_path)
        decoding = _decoding_from_generation_manifest(generation_manifest)
        if _coerce_int(decoding.get("no_repeat_ngram_size"), default=0) != no_repeat_ngram_size:
            continue

        signature = _generation_signature_from_manifest(generation_manifest)
        artifact = ScoreArtifact(
            score_path=score_path,
            score_manifest_path=score_manifest_path,
            generation_manifest_path=generation_manifest_path,
            signature=signature,
            decoding=decoding,
        )
        existing = unique_files.get(signature)
        if existing is None or score_path.name > existing.score_path.name:
            unique_files[signature] = artifact

    def sort_key(item: ScoreArtifact) -> tuple[int, float, int, float]:
        strategy = str(item.decoding["strategy"])
        temperature = _coerce_float(item.decoding.get("temperature")) or 0.0
        top_k = _coerce_int(item.decoding.get("top_k"), default=0)
        top_p = _coerce_float(item.decoding.get("top_p")) or 0.0
        return (_strategy_rank(strategy), temperature, top_k, top_p)

    return sorted(unique_files.values(), key=sort_key)


def _build_combined_scores(
    settings: Settings,
    stem: str,
    score_files: list[ScoreArtifact],
    no_repeat_ngram_size: int,
) -> tuple[Path, Path]:
    if not score_files:
        raise ValueError(
            f"No scored files found for no_repeat_ngram_size={no_repeat_ngram_size}. "
            "Run generation and scoring for the target ablation first."
        )

    combined_path = settings.output_dir / "scores" / f"{stem}.parquet"
    manifest_path = settings.output_dir / "manifests" / f"{stem}.json"

    frames = [pd.read_parquet(artifact.score_path) for artifact in score_files]
    combined_df = pd.concat(frames, ignore_index=True)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(combined_path, index=False)

    manifest = {
        "cache_key": stem,
        "created_at_utc": _utc_now_iso(),
        "created_from_scores": [str(artifact.score_path.resolve()) for artifact in score_files],
        "generation_signatures": [artifact.signature for artifact in score_files],
        "model_reference": settings.scoring.resolved_model_reference(),
        "record_count": len(combined_df),
        "use_masking": settings.scoring.use_masking,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "n_decoding_configs": len(score_files),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return combined_path, manifest_path


def _build_week3_bundle(
    settings: Settings,
    stem: str,
    no_repeat_ngram_size: int,
) -> Week3Bundle:
    score_files = _collect_score_files_for_no_repeat(settings, no_repeat_ngram_size)
    if len(score_files) != 10:
        raise ValueError(
            f"Expected 10 unique scored decoding configs for no_repeat_ngram_size={no_repeat_ngram_size}, "
            f"found {len(score_files)}."
        )

    combined_scores_path, combined_manifest_path = _build_combined_scores(
        settings=settings,
        stem=stem,
        score_files=score_files,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    metric_paths = compute_week3_metrics(
        scores_path=combined_scores_path,
        output_dir=settings.output_dir,
        n_bootstrap=settings.n_bootstrap,
        quality_n_bootstrap=settings.quality_n_bootstrap,
        ci_level=settings.ci_level,
    )
    return Week3Bundle(
        combined_scores_path=combined_scores_path,
        combined_manifest_path=combined_manifest_path,
        regard_metrics_path=metric_paths["week3_regard_distributions"],
        gap_metrics_path=metric_paths["week3_negative_gaps_with_ci"],
        quality_metrics_path=metric_paths["week3_quality_metrics_with_ci"],
        summary_path=metric_paths["week3_summary"],
        score_files=[artifact.score_path for artifact in score_files],
        no_repeat_ngram_size=no_repeat_ngram_size,
    )


def compare_quality_metrics(
    baseline_df: pd.DataFrame,
    antirep_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_prepared = _prepare_decoding_frame(baseline_df)
    antirep_prepared = _prepare_decoding_frame(antirep_df)

    baseline_columns = [
        "decoding_key",
        "config_label",
        "config_order",
        *IDENTITY_COLUMNS,
        "no_repeat_ngram_size",
        "n_generations",
        *QUALITY_COLUMNS,
        *QUALITY_CI_COLUMNS,
    ]
    antirep_columns = [
        "decoding_key",
        "no_repeat_ngram_size",
        "n_generations",
        *QUALITY_COLUMNS,
        *QUALITY_CI_COLUMNS,
    ]
    baseline_rename_map = {
        "no_repeat_ngram_size": "baseline_no_repeat_ngram_size",
        "n_generations": "baseline_n_generations",
        **{column: f"baseline_{column}" for column in QUALITY_COLUMNS + QUALITY_CI_COLUMNS},
    }
    antirep_rename_map = {
        "no_repeat_ngram_size": "antirep_no_repeat_ngram_size",
        "n_generations": "antirep_n_generations",
        **{column: f"antirep_{column}" for column in QUALITY_COLUMNS + QUALITY_CI_COLUMNS},
    }
    baseline_subset = cast(pd.DataFrame, baseline_prepared.loc[:, baseline_columns])
    antirep_subset = cast(pd.DataFrame, antirep_prepared.loc[:, antirep_columns])
    baseline_renamed = _rename_frame_columns(baseline_subset, baseline_rename_map)
    antirep_renamed = _rename_frame_columns(antirep_subset, antirep_rename_map)

    merged = baseline_renamed.merge(
        antirep_renamed,
        on="decoding_key",
        how="inner",
        validate="one_to_one",
    )
    for column in QUALITY_COLUMNS:
        merged[f"delta_{column}"] = merged[f"antirep_{column}"] - merged[f"baseline_{column}"]
    return merged.sort_values(["config_order"]).reset_index(drop=True)


def compare_regard_distributions(
    baseline_df: pd.DataFrame,
    antirep_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_prepared = _prepare_decoding_frame(baseline_df)
    antirep_prepared = _prepare_decoding_frame(antirep_df)

    baseline_columns = [
        "decoding_key",
        "config_label",
        "config_order",
        *IDENTITY_COLUMNS,
        "group",
        "no_repeat_ngram_size",
        "total",
        *REGARD_COLUMNS,
    ]
    antirep_columns = ["decoding_key", "group", "no_repeat_ngram_size", "total", *REGARD_COLUMNS]
    baseline_rename_map = {
        "no_repeat_ngram_size": "baseline_no_repeat_ngram_size",
        "total": "baseline_total",
        **{column: f"baseline_{column}" for column in REGARD_COLUMNS},
    }
    antirep_rename_map = {
        "no_repeat_ngram_size": "antirep_no_repeat_ngram_size",
        "total": "antirep_total",
        **{column: f"antirep_{column}" for column in REGARD_COLUMNS},
    }
    baseline_subset = cast(pd.DataFrame, baseline_prepared.loc[:, baseline_columns])
    antirep_subset = cast(pd.DataFrame, antirep_prepared.loc[:, antirep_columns])
    baseline_renamed = _rename_frame_columns(baseline_subset, baseline_rename_map)
    antirep_renamed = _rename_frame_columns(antirep_subset, antirep_rename_map)

    merged = baseline_renamed.merge(
        antirep_renamed,
        on=["decoding_key", "group"],
        how="inner",
        validate="one_to_one",
    )
    for column in REGARD_COLUMNS:
        merged[f"delta_{column}"] = merged[f"antirep_{column}"] - merged[f"baseline_{column}"]
    return merged.sort_values(["config_order", "group"]).reset_index(drop=True)


def _gap_sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def compare_gap_metrics(
    baseline_df: pd.DataFrame,
    antirep_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_prepared = _prepare_decoding_frame(baseline_df)
    antirep_prepared = _prepare_decoding_frame(antirep_df)

    baseline_columns = [
        "decoding_key",
        "config_label",
        "config_order",
        *IDENTITY_COLUMNS,
        "prompt_type",
        "group_a",
        "group_b",
        "no_repeat_ngram_size",
        "gap_neg",
        "ci_lower",
        "ci_upper",
        "p_neg_a",
        "p_neg_b",
        "n_samples_a",
        "n_samples_b",
    ]
    antirep_columns = [
        "decoding_key",
        "prompt_type",
        "group_a",
        "group_b",
        "no_repeat_ngram_size",
        "gap_neg",
        "ci_lower",
        "ci_upper",
        "p_neg_a",
        "p_neg_b",
        "n_samples_a",
        "n_samples_b",
    ]
    baseline_rename_map = {
        "no_repeat_ngram_size": "baseline_no_repeat_ngram_size",
        "gap_neg": "baseline_gap_neg",
        "ci_lower": "baseline_ci_lower",
        "ci_upper": "baseline_ci_upper",
        "p_neg_a": "baseline_p_neg_a",
        "p_neg_b": "baseline_p_neg_b",
        "n_samples_a": "baseline_n_samples_a",
        "n_samples_b": "baseline_n_samples_b",
    }
    antirep_rename_map = {
        "no_repeat_ngram_size": "antirep_no_repeat_ngram_size",
        "gap_neg": "antirep_gap_neg",
        "ci_lower": "antirep_ci_lower",
        "ci_upper": "antirep_ci_upper",
        "p_neg_a": "antirep_p_neg_a",
        "p_neg_b": "antirep_p_neg_b",
        "n_samples_a": "antirep_n_samples_a",
        "n_samples_b": "antirep_n_samples_b",
    }
    baseline_subset = cast(pd.DataFrame, baseline_prepared.loc[:, baseline_columns])
    antirep_subset = cast(pd.DataFrame, antirep_prepared.loc[:, antirep_columns])
    baseline_renamed = _rename_frame_columns(baseline_subset, baseline_rename_map)
    antirep_renamed = _rename_frame_columns(antirep_subset, antirep_rename_map)

    merged = baseline_renamed.merge(
        antirep_renamed,
        on=["decoding_key", "prompt_type", "group_a", "group_b"],
        how="inner",
        validate="one_to_one",
    )
    merged["delta_gap_neg"] = merged["antirep_gap_neg"] - merged["baseline_gap_neg"]
    merged["sign_changed"] = merged.apply(
        lambda row: _gap_sign(float(row["baseline_gap_neg"]))
        != _gap_sign(float(row["antirep_gap_neg"])),
        axis=1,
    )
    return merged.sort_values(["config_order", "prompt_type", "group_a", "group_b"]).reset_index(
        drop=True
    )


def extract_key_trace(gap_comparison_df: pd.DataFrame) -> pd.DataFrame:
    trace = cast(
        pd.DataFrame,
        gap_comparison_df[
            (gap_comparison_df["prompt_type"] == KEY_TRACE["prompt_type"])
            & (gap_comparison_df["group_a"] == KEY_TRACE["group_a"])
            & (gap_comparison_df["group_b"] == KEY_TRACE["group_b"])
        ].copy(),
    )
    return trace.sort_values(by=["config_order"]).reset_index(drop=True)


def summarize_antirepetition(
    quality_comparison_df: pd.DataFrame,
    gap_comparison_df: pd.DataFrame,
    key_trace_df: pd.DataFrame,
    *,
    baseline_bundle: Week3Bundle,
    antirep_bundle: Week3Bundle,
    masking_summary_path: Path | None,
) -> dict[str, Any]:
    sign_flip_count = (
        sum(bool(value) for value in gap_comparison_df["sign_changed"].tolist())
        if not gap_comparison_df.empty
        else 0
    )
    repetition_improved_count = (
        sum(
            float(value) < 0.0
            for value in quality_comparison_df["delta_repeated_3gram_rate"].tolist()
        )
        if not quality_comparison_df.empty
        else 0
    )
    distinct2_improved_count = (
        sum(float(value) > 0.0 for value in quality_comparison_df["delta_distinct_2"].tolist())
        if not quality_comparison_df.empty
        else 0
    )
    key_trace_sign_flip = (
        bool(key_trace_df["sign_changed"].any()) if not key_trace_df.empty else False
    )
    key_trace_positive_all = (
        bool((key_trace_df["antirep_gap_neg"] > 0).all()) if not key_trace_df.empty else False
    )

    max_gap_shift_row: dict[str, Any] | None = None
    if not gap_comparison_df.empty:
        max_gap_shift_series = gap_comparison_df.loc[
            gap_comparison_df["delta_gap_neg"].abs().idxmax()
        ]
        max_gap_shift_row = {
            "config_label": str(max_gap_shift_series["config_label"]),
            "prompt_type": str(max_gap_shift_series["prompt_type"]),
            "group_a": str(max_gap_shift_series["group_a"]),
            "group_b": str(max_gap_shift_series["group_b"]),
            "delta_gap_neg": float(max_gap_shift_series["delta_gap_neg"]),
        }

    max_quality_shift: dict[str, dict[str, Any]] = {}
    for metric_name in QUALITY_COLUMNS:
        if quality_comparison_df.empty:
            break
        shift_series = quality_comparison_df.loc[
            quality_comparison_df[f"delta_{metric_name}"].abs().idxmax()
        ]
        max_quality_shift[metric_name] = {
            "config_label": str(shift_series["config_label"]),
            "delta": float(shift_series[f"delta_{metric_name}"]),
            "baseline": float(shift_series[f"baseline_{metric_name}"]),
            "antirep": float(shift_series[f"antirep_{metric_name}"]),
        }

    top_gap_shifts = (
        gap_comparison_df.assign(abs_delta=gap_comparison_df["delta_gap_neg"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(5)
    )
    top_gap_shift_records = [
        {
            "config_label": str(row["config_label"]),
            "prompt_type": str(row["prompt_type"]),
            "group_a": str(row["group_a"]),
            "group_b": str(row["group_b"]),
            "delta_gap_neg": _require_float(row["delta_gap_neg"]),
            "baseline_gap_neg": _require_float(row["baseline_gap_neg"]),
            "antirep_gap_neg": _require_float(row["antirep_gap_neg"]),
        }
        for row in _dataframe_records(top_gap_shifts)
    ]

    if key_trace_positive_all and not key_trace_sign_flip:
        main_conclusion_changed = False
        main_conclusion_text = (
            "Anti-repetition reduces repetition while the highlighted description / "
            "Black man vs White woman gap stays positive across every decoding configuration. "
            "The main Week 3 conclusion remains unchanged."
        )
    else:
        main_conclusion_changed = True
        main_conclusion_text = (
            "Anti-repetition changes the highlighted Week 3 trace enough to weaken the original "
            "qualitative conclusion, so the main Week 3 takeaway is no longer fully stable."
        )

    masking_status: dict[str, Any]
    if masking_summary_path is not None and masking_summary_path.exists():
        masking_status = {
            "status": "available",
            "path": str(masking_summary_path.resolve()),
            "summary": _read_json_object(masking_summary_path),
        }
    else:
        masking_status = {
            "status": "pending",
            "path": None,
            "summary": None,
        }

    return {
        "created_at_utc": _utc_now_iso(),
        "baseline_bundle": {
            "combined_scores_path": str(baseline_bundle.combined_scores_path.resolve()),
            "gap_metrics_path": str(baseline_bundle.gap_metrics_path.resolve()),
            "quality_metrics_path": str(baseline_bundle.quality_metrics_path.resolve()),
            "regard_metrics_path": str(baseline_bundle.regard_metrics_path.resolve()),
            "summary_path": str(baseline_bundle.summary_path.resolve()),
        },
        "antirep_bundle": {
            "combined_scores_path": str(antirep_bundle.combined_scores_path.resolve()),
            "gap_metrics_path": str(antirep_bundle.gap_metrics_path.resolve()),
            "quality_metrics_path": str(antirep_bundle.quality_metrics_path.resolve()),
            "regard_metrics_path": str(antirep_bundle.regard_metrics_path.resolve()),
            "summary_path": str(antirep_bundle.summary_path.resolve()),
        },
        "total_compared_decoding_configs": int(len(quality_comparison_df)),
        "total_compared_gap_rows": int(len(gap_comparison_df)),
        "sign_flip_count": sign_flip_count,
        "repetition_improved_config_count": repetition_improved_count,
        "distinct2_improved_config_count": distinct2_improved_count,
        "largest_gap_shift": max_gap_shift_row,
        "largest_quality_shifts": max_quality_shift,
        "top_gap_shifts": top_gap_shift_records,
        "key_trace": {
            "prompt_type": KEY_TRACE["prompt_type"],
            "group_a": KEY_TRACE["group_a"],
            "group_b": KEY_TRACE["group_b"],
            "rows": [
                {
                    "config_label": str(row["config_label"]),
                    "baseline_gap_neg": _require_float(row["baseline_gap_neg"]),
                    "antirep_gap_neg": _require_float(row["antirep_gap_neg"]),
                    "delta_gap_neg": _require_float(row["delta_gap_neg"]),
                    "sign_changed": bool(row["sign_changed"]),
                }
                for row in _dataframe_records(key_trace_df)
            ],
            "stays_positive_across_configs": key_trace_positive_all,
            "has_sign_flip": key_trace_sign_flip,
        },
        "main_week3_conclusion_changed": main_conclusion_changed,
        "main_week3_conclusion_text": main_conclusion_text,
        "masking_integration": masking_status,
        "ethics_notice": (
            "Generated and scored text may contain offensive content. "
            "Do not publish or commit large raw dumps."
        ),
    }


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _plot_quality_deltas(quality_df: pd.DataFrame, output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = quality_df["config_label"].tolist()

    axes[0].bar(labels, quality_df["delta_repeated_3gram_rate"], color="#d35400")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Anti-Repetition Delta: Repeated 3-Gram Rate")
    axes[0].set_ylabel("Anti-repetition - baseline")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(labels, quality_df["delta_distinct_2"], color="#1f78b4")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Anti-Repetition Delta: Distinct-2")
    axes[1].set_ylabel("Anti-repetition - baseline")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_gap_deltas(key_trace_df: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = key_trace_df["config_label"].tolist()
    ax.bar(labels, key_trace_df["delta_gap_neg"], color="#8e44ad")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Anti-Repetition Delta: Key Trace Gap\ndescription / Black man vs White woman")
    ax.set_ylabel("Anti-repetition - baseline")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _tex_escape(text: str) -> str:
    escaped = text
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def _render_quality_table_rows(quality_df: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in quality_df.iterrows():
        rows.append(
            "    "
            f"{_tex_escape(str(row['config_label']))} & "
            f"{row['baseline_repeated_3gram_rate']:.3f} & "
            f"{row['antirep_repeated_3gram_rate']:.3f} & "
            f"{row['delta_repeated_3gram_rate']:.3f} & "
            f"{row['baseline_distinct_2']:.3f} & "
            f"{row['antirep_distinct_2']:.3f} & "
            f"{row['delta_distinct_2']:.3f} \\\\"
        )
    return "\n".join(rows)


def _render_key_trace_rows(key_trace_df: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in key_trace_df.iterrows():
        rows.append(
            "    "
            f"{_tex_escape(str(row['config_label']))} & "
            f"{row['baseline_gap_neg']:.3f} & "
            f"{row['antirep_gap_neg']:.3f} & "
            f"{row['delta_gap_neg']:.3f} & "
            f"{'yes' if bool(row['sign_changed']) else 'no'} \\\\"
        )
    return "\n".join(rows)


def _build_final_summary(
    antirep_summary: dict[str, Any],
    comparison_paths: dict[str, str],
) -> dict[str, Any]:
    masking_integration = antirep_summary["masking_integration"]
    antirepetition_done = True
    masking_done = masking_integration["status"] == "available"
    final_conclusion = antirep_summary["main_week3_conclusion_text"]
    summary_artifacts = dict(comparison_paths)
    if masking_done:
        masking_summary = masking_integration["summary"]
        masking_path = masking_integration["path"]
        if isinstance(masking_path, str):
            summary_artifacts["masking_summary"] = masking_path
        final_conclusion += (
            " " + str(masking_summary["main_week3_conclusion_text"])
            if isinstance(masking_summary, dict) and "main_week3_conclusion_text" in masking_summary
            else " Masking-sensitivity artifacts are available and integrated."
        )
    else:
        final_conclusion += " Masking-sensitivity integration is still pending."

    return {
        "created_at_utc": _utc_now_iso(),
        "w1_complete": True,
        "w2_complete": True,
        "w3_complete": True,
        "w4_complete": True,
        "w5_antirepetition_complete": antirepetition_done,
        "w5_masking_complete": masking_done,
        "comparison_artifacts": summary_artifacts,
        "final_conclusion_text": final_conclusion,
        "main_week3_conclusion_changed": antirep_summary["main_week3_conclusion_changed"],
        "key_trace_stays_positive": antirep_summary["key_trace"]["stays_positive_across_configs"],
        "key_trace_has_sign_flip": antirep_summary["key_trace"]["has_sign_flip"],
        "ethics_notice": antirep_summary["ethics_notice"],
    }


def _write_report(
    settings: Settings,
    quality_comparison_df: pd.DataFrame,
    key_trace_df: pd.DataFrame,
    antirep_summary: dict[str, Any],
    paths: Week5ArtifactPaths,
) -> Path:
    report_dir = settings.output_dir.parent / "docs" / "week5"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.tex"
    masking_integration = antirep_summary["masking_integration"]
    masking_status = masking_integration["status"]
    masking_summary = masking_integration["summary"]
    masking_text = (
        "Masking-sensitivity artifacts are available and integrated below."
        if masking_status == "available"
        else "Masking-sensitivity artifacts are not yet available, so this report covers the anti-repetition ablation only."
    )
    masking_section = ""
    masking_artifact_references = ""
    if isinstance(masking_summary, dict):
        masking_artifact_references = f"""
\\item \\texttt{{{_tex_escape(str((settings.output_dir / "metrics" / "week5_masking_gap_comparison.csv").relative_to(settings.output_dir.parent)))}}}
\\item \\texttt{{{_tex_escape(str((settings.output_dir / "metrics" / "week5_masking_distribution_comparison.csv").relative_to(settings.output_dir.parent)))}}}
\\item \\texttt{{{_tex_escape(str((settings.output_dir / "metrics" / "week5_masking_key_trace.csv").relative_to(settings.output_dir.parent)))}}}
\\item \\texttt{{{_tex_escape(str((settings.output_dir / "metrics" / "week5_masking_summary.json").relative_to(settings.output_dir.parent)))}}}
"""
        masking_section = f"""
\\section{{Masking Sensitivity}}
The masking-sensitivity ablation compares the baseline Week 3 masked scores against unmasked scores
for the same baseline decoding grid. Its machine-readable summary concludes:

\\begin{{quote}}
{_tex_escape(str(masking_summary["main_week3_conclusion_text"]))}
\\end{{quote}}

Three points matter most:
\\begin{{itemize}}
\\item Masked vs unmasked comparison rows: {int(masking_summary["total_compared_prompt_type_group_pairs"])}.
\\item Sign flips across all compared gap rows: {int(masking_summary["sign_flip_count"])}.
\\item The highlighted key trace {"stayed positive across all configurations" if bool(masking_summary["key_trace"]["stays_positive_across_configs"]) else "did not stay positive across all configurations"}.
\\end{{itemize}}

The masking artifact summary is stored at
\\texttt{{{_tex_escape(str(Path(str(masking_integration["path"])).relative_to(settings.output_dir.parent)))}}}.
"""
    report = f"""\\documentclass[11pt]{{article}}

\\usepackage[margin=1in]{{geometry}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{booktabs}}
\\usepackage{{tabularx}}
\\usepackage{{array}}
\\usepackage{{enumitem}}
\\usepackage{{hyperref}}

\\setlist[itemize]{{leftmargin=1.2em, itemsep=0.2em, topsep=0.3em}}
\\hypersetup{{colorlinks=true, urlcolor=blue}}

\\title{{Week 5 Report: Anti-Repetition Ablation and Final Integration}}
\\author{{Ivan Chabanov \\\\ Alexander Michailov}}
\\date{{{_report_date_str()}}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
This report completes the Week 5 anti-repetition ablation for the proposal-locked study
\\textit{{Decoding Amplifies Bias}}. The new experiment keeps the GPT-2 checkpoint, prompt bank,
seeds, sample count, and maximum generation length fixed while enabling
\\texttt{{no\\_repeat\\_ngram\\_size = 3}} across the full decoding grid. The purpose is to test
whether reducing repetition changes the study's main bias conclusions or mainly improves generation
quality.
\\end{{abstract}}

\\section{{Status}}
Weeks 1--4 are complete. Week 5 anti-repetition is now complete and backed by generated repository
artifacts. {_tex_escape(masking_text)}

\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}lX@{{}}}}
\\toprule
Item & Value \\\\
\\midrule
Baseline combined scores & \\texttt{{{_tex_escape(str(paths.baseline_bundle.combined_scores_path.relative_to(settings.output_dir.parent)))}}} \\\\
Anti-repetition combined scores & \\texttt{{{_tex_escape(str(paths.antirep_bundle.combined_scores_path.relative_to(settings.output_dir.parent)))}}} \\\\
Target anti-repetition setting & \\texttt{{no\\_repeat\\_ngram\\_size = 3}} \\\\
Scoring mode & masked, released \\texttt{{sasha/regardv3}} scorer \\\\
Bootstrap samples & {settings.n_bootstrap} for bias metrics; {settings.quality_n_bootstrap} for quality metrics \\\\
Compared decoding configs & {len(quality_comparison_df)} \\\\
\\bottomrule
\\end{{tabularx}}
\\caption{{Week 5 anti-repetition setup.}}
\\end{{table}}

\\section{{Anti-Repetition Results}}
The key Week 3 claim remains summarized by the generated summary artifact:

\\begin{{quote}}
{_tex_escape(str(antirep_summary["main_week3_conclusion_text"]))}
\\end{{quote}}

\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}lrrrrrr@{{}}}}
\\toprule
Config & Base rep-3 & Anti rep-3 & $\\Delta$ rep-3 & Base d2 & Anti d2 & $\\Delta$ d2 \\\\
\\midrule
{_render_quality_table_rows(quality_comparison_df)}
\\bottomrule
\\end{{tabularx}}
\\caption{{Quality changes under anti-repetition. Negative $\\Delta$ rep-3 and positive $\\Delta$ d2 are desirable.}}
\\end{{table}}

\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}lrrrr@{{}}}}
\\toprule
Config & Base $\\Delta_{{neg}}$ & Anti $\\Delta_{{neg}}$ & Change & Sign flip \\\\
\\midrule
{_render_key_trace_rows(key_trace_df)}
\\bottomrule
\\end{{tabularx}}
\\caption{{Highlighted Week 3 trace: \\texttt{{description / Black man vs White woman}}.}}
\\end{{table}}

{masking_section}

\\section{{Final Interpretation}}
Three points matter most:
\\begin{{itemize}}
\\item Anti-repetition improved repeated 3-gram rate in {antirep_summary["repetition_improved_config_count"]} of {len(quality_comparison_df)} decoding configs.
\\item Distinct-2 improved in {antirep_summary["distinct2_improved_config_count"]} of {len(quality_comparison_df)} decoding configs.
\\item The highlighted key trace {"stayed positive across all configurations" if antirep_summary["key_trace"]["stays_positive_across_configs"] else "did not stay positive across all configurations"}.
\\end{{itemize}}

The complete machine-readable summary is stored at
\\texttt{{{_tex_escape(str(paths.summary_path.relative_to(settings.output_dir.parent)))}}}. Final integrated
project status is stored at
\\texttt{{{_tex_escape(str(paths.final_summary_path.relative_to(settings.output_dir.parent)))}}}.

\\section{{Artifact References}}
\\begin{{itemize}}
\\item \\texttt{{{_tex_escape(str(paths.quality_comparison_path.relative_to(settings.output_dir.parent)))}}}
\\item \\texttt{{{_tex_escape(str(paths.gap_comparison_path.relative_to(settings.output_dir.parent)))}}}
\\item \\texttt{{{_tex_escape(str(paths.regard_comparison_path.relative_to(settings.output_dir.parent)))}}}
\\item \\texttt{{{_tex_escape(str(paths.key_trace_path.relative_to(settings.output_dir.parent)))}}}
\\item \\texttt{{{_tex_escape(str(paths.quality_plot_path.relative_to(settings.output_dir.parent)))}}}
\\item \\texttt{{{_tex_escape(str(paths.gap_plot_path.relative_to(settings.output_dir.parent)))}}}
{masking_artifact_references}
\\end{{itemize}}

No large raw generations should be pasted into this report. Include only minimal excerpts with an
explicit content warning if excerpts are needed later.

\\end{{document}}
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_week5_antirepetition(
    settings: Settings,
    *,
    target_no_repeat_ngram_size: int = 3,
) -> Week5ArtifactPaths:
    baseline_bundle = _build_week3_bundle(settings, BASELINE_STEM, no_repeat_ngram_size=0)
    antirep_bundle = _build_week3_bundle(
        settings,
        ANTIREP_STEM,
        no_repeat_ngram_size=target_no_repeat_ngram_size,
    )

    baseline_quality_df = pd.read_csv(baseline_bundle.quality_metrics_path)
    antirep_quality_df = pd.read_csv(antirep_bundle.quality_metrics_path)
    baseline_regard_df = pd.read_csv(baseline_bundle.regard_metrics_path)
    antirep_regard_df = pd.read_csv(antirep_bundle.regard_metrics_path)
    baseline_gap_df = pd.read_csv(baseline_bundle.gap_metrics_path)
    antirep_gap_df = pd.read_csv(antirep_bundle.gap_metrics_path)

    quality_comparison_df = compare_quality_metrics(baseline_quality_df, antirep_quality_df)
    regard_comparison_df = compare_regard_distributions(baseline_regard_df, antirep_regard_df)
    gap_comparison_df = compare_gap_metrics(baseline_gap_df, antirep_gap_df)
    key_trace_df = extract_key_trace(gap_comparison_df)

    metrics_dir = settings.output_dir / "metrics"
    plots_dir = settings.output_dir / "plots"
    reports_dir = settings.output_dir / "reports"
    gap_comparison_path = _write_csv(
        gap_comparison_df,
        metrics_dir / "week5_antirep_gap_comparison.csv",
    )
    quality_comparison_path = _write_csv(
        quality_comparison_df,
        metrics_dir / "week5_antirep_quality_comparison.csv",
    )
    regard_comparison_path = _write_csv(
        regard_comparison_df,
        metrics_dir / "week5_antirep_regard_comparison.csv",
    )
    key_trace_path = _write_csv(
        key_trace_df,
        metrics_dir / "week5_antirep_key_trace.csv",
    )

    masking_summary_path = metrics_dir / "week5_masking_summary.json"
    antirep_summary = summarize_antirepetition(
        quality_comparison_df,
        gap_comparison_df,
        key_trace_df,
        baseline_bundle=baseline_bundle,
        antirep_bundle=antirep_bundle,
        masking_summary_path=masking_summary_path if masking_summary_path.exists() else None,
    )
    summary_path = metrics_dir / "week5_antirep_summary.json"
    summary_path.write_text(json.dumps(antirep_summary, indent=2, sort_keys=True), encoding="utf-8")

    quality_plot_path = _plot_quality_deltas(
        quality_comparison_df,
        plots_dir / "week5_antirep_quality_delta.png",
    )
    gap_plot_path = _plot_gap_deltas(
        key_trace_df,
        plots_dir / "week5_antirep_gap_delta.png",
    )

    comparison_paths = {
        "gap_comparison": str(gap_comparison_path.resolve()),
        "quality_comparison": str(quality_comparison_path.resolve()),
        "regard_comparison": str(regard_comparison_path.resolve()),
        "key_trace": str(key_trace_path.resolve()),
        "antirep_summary": str(summary_path.resolve()),
        "quality_plot": str(quality_plot_path.resolve()),
        "gap_plot": str(gap_plot_path.resolve()),
    }
    final_summary = _build_final_summary(antirep_summary, comparison_paths)
    final_summary_path = reports_dir / "week5_final_summary.json"
    final_summary_path.parent.mkdir(parents=True, exist_ok=True)
    final_summary_path.write_text(
        json.dumps(final_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    artifact_paths = Week5ArtifactPaths(
        baseline_bundle=baseline_bundle,
        antirep_bundle=antirep_bundle,
        gap_comparison_path=gap_comparison_path,
        quality_comparison_path=quality_comparison_path,
        regard_comparison_path=regard_comparison_path,
        key_trace_path=key_trace_path,
        summary_path=summary_path,
        quality_plot_path=quality_plot_path,
        gap_plot_path=gap_plot_path,
        final_summary_path=final_summary_path,
        report_path=settings.output_dir.parent / "docs" / "week5" / "report.tex",
    )
    report_path = _write_report(
        settings=settings,
        quality_comparison_df=quality_comparison_df,
        key_trace_df=key_trace_df,
        antirep_summary=antirep_summary,
        paths=artifact_paths,
    )
    return Week5ArtifactPaths(
        baseline_bundle=artifact_paths.baseline_bundle,
        antirep_bundle=artifact_paths.antirep_bundle,
        gap_comparison_path=artifact_paths.gap_comparison_path,
        quality_comparison_path=artifact_paths.quality_comparison_path,
        regard_comparison_path=artifact_paths.regard_comparison_path,
        key_trace_path=artifact_paths.key_trace_path,
        summary_path=artifact_paths.summary_path,
        quality_plot_path=artifact_paths.quality_plot_path,
        gap_plot_path=artifact_paths.gap_plot_path,
        final_summary_path=artifact_paths.final_summary_path,
        report_path=report_path,
    )
