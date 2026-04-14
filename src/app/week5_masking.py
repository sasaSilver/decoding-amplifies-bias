from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .metrics import compute_week3_metrics
from .models import MaskingArtifactPaths, MaskingBundle, MaskingScoreArtifact
from .settings.masking import masking_settings
from .settings.settings import Settings


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def _collect_score_files_for_masking(
    settings: Settings,
    use_masking: bool,
    no_repeat_ngram_size: int,
) -> list[MaskingScoreArtifact]:
    manifests_dir = settings.output_dir / "manifests"
    scores_dir = settings.output_dir / "scores"
    model_reference = settings.scoring.resolved_model_reference()
    unique_files: dict[str, MaskingScoreArtifact] = {}

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
        if _coerce_bool(score_manifest.get("use_masking")) != use_masking:
            continue

        generation_manifest_path = manifests_dir / f"{generations_cache_key}.json"
        if not generation_manifest_path.exists():
            continue

        generation_manifest = _read_json_object(generation_manifest_path)
        decoding = _decoding_from_generation_manifest(generation_manifest)
        if _coerce_int(decoding.get("no_repeat_ngram_size"), default=0) != no_repeat_ngram_size:
            continue
        signature = _generation_signature_from_manifest(generation_manifest)
        artifact = MaskingScoreArtifact(
            score_path=score_path,
            score_manifest_path=score_manifest_path,
            generation_manifest_path=generation_manifest_path,
            signature=signature,
            decoding=decoding,
            use_masking=use_masking,
        )
        existing = unique_files.get(signature)
        if existing is None or score_path.name > existing.score_path.name:
            unique_files[signature] = artifact

    def sort_key(item: MaskingScoreArtifact) -> tuple[int, float, int, float]:
        strategy = str(item.decoding["strategy"])
        temperature = _coerce_float(item.decoding.get("temperature")) or 0.0
        top_k = _coerce_int(item.decoding.get("top_k"), default=0)
        top_p = _coerce_float(item.decoding.get("top_p")) or 0.0
        return (_strategy_rank(strategy), temperature, top_k, top_p)

    return sorted(unique_files.values(), key=sort_key)


def _build_combined_scores(
    settings: Settings,
    stem: str,
    score_files: list[MaskingScoreArtifact],
    use_masking: bool,
) -> tuple[Path, Path]:
    if not score_files:
        raise ValueError(
            f"No scored files found for use_masking={use_masking}. "
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
        "use_masking": use_masking,
        "no_repeat_ngram_size": masking_settings.target_no_repeat_ngram_size,
        "n_decoding_configs": len(score_files),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return combined_path, manifest_path


def _build_masking_bundle(
    settings: Settings,
    stem: str,
    use_masking: bool,
) -> MaskingBundle:
    score_files = _collect_score_files_for_masking(
        settings,
        use_masking,
        masking_settings.target_no_repeat_ngram_size,
    )
    if len(score_files) != 10:
        raise ValueError(
            "Expected 10 unique scored decoding configs for "
            f"use_masking={use_masking} and "
            f"no_repeat_ngram_size={masking_settings.target_no_repeat_ngram_size}, "
            f"found {len(score_files)}."
        )

    combined_scores_path, combined_manifest_path = _build_combined_scores(
        settings=settings,
        stem=stem,
        score_files=score_files,
        use_masking=use_masking,
    )
    metric_paths = compute_week3_metrics(
        scores_path=combined_scores_path,
        output_dir=settings.output_dir,
        n_bootstrap=settings.n_bootstrap,
        quality_n_bootstrap=settings.quality_n_bootstrap,
        ci_level=settings.ci_level,
    )
    return MaskingBundle(
        combined_scores_path=combined_scores_path,
        combined_manifest_path=combined_manifest_path,
        regard_metrics_path=metric_paths["week3_regard_distributions"],
        gap_metrics_path=metric_paths["week3_negative_gaps_with_ci"],
        score_files=[artifact.score_path for artifact in score_files],
        use_masking=use_masking,
    )


def _gap_sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def compare_gap_metrics(
    masked_df: pd.DataFrame,
    unmasked_df: pd.DataFrame,
) -> pd.DataFrame:
    masked_prepared = _prepare_decoding_frame(masked_df)
    unmasked_prepared = _prepare_decoding_frame(unmasked_df)

    masked_columns = [
        "decoding_key",
        "config_label",
        "config_order",
        *masking_settings.identity_columns,
        "prompt_type",
        "group_a",
        "group_b",
        "gap_neg",
        "ci_lower",
        "ci_upper",
        "p_neg_a",
        "p_neg_b",
        "n_samples_a",
        "n_samples_b",
    ]
    unmasked_columns = [
        "decoding_key",
        "prompt_type",
        "group_a",
        "group_b",
        "gap_neg",
        "ci_lower",
        "ci_upper",
        "p_neg_a",
        "p_neg_b",
        "n_samples_a",
        "n_samples_b",
    ]
    masked_rename_map = {
        "gap_neg": "masked_gap_neg",
        "ci_lower": "masked_ci_lower",
        "ci_upper": "masked_ci_upper",
        "p_neg_a": "masked_p_neg_a",
        "p_neg_b": "masked_p_neg_b",
        "n_samples_a": "masked_n_samples_a",
        "n_samples_b": "masked_n_samples_b",
    }
    unmasked_rename_map = {
        "gap_neg": "unmasked_gap_neg",
        "ci_lower": "unmasked_ci_lower",
        "ci_upper": "unmasked_ci_upper",
        "p_neg_a": "unmasked_p_neg_a",
        "p_neg_b": "unmasked_p_neg_b",
        "n_samples_a": "unmasked_n_samples_a",
        "n_samples_b": "unmasked_n_samples_b",
    }
    masked_subset = cast(pd.DataFrame, masked_prepared.loc[:, masked_columns])
    unmasked_subset = cast(pd.DataFrame, unmasked_prepared.loc[:, unmasked_columns])
    masked_renamed = _rename_frame_columns(masked_subset, masked_rename_map)
    unmasked_renamed = _rename_frame_columns(unmasked_subset, unmasked_rename_map)

    merged = masked_renamed.merge(
        unmasked_renamed,
        on=["decoding_key", "prompt_type", "group_a", "group_b"],
        how="inner",
        validate="one_to_one",
    )
    merged["delta_gap_neg"] = merged["unmasked_gap_neg"] - merged["masked_gap_neg"]
    merged["sign_changed"] = merged.apply(
        lambda row: _gap_sign(float(row["masked_gap_neg"]))
        != _gap_sign(float(row["unmasked_gap_neg"])),
        axis=1,
    )
    return merged.sort_values(["config_order", "prompt_type", "group_a", "group_b"]).reset_index(
        drop=True
    )


def compare_regard_distributions(
    masked_df: pd.DataFrame,
    unmasked_df: pd.DataFrame,
) -> pd.DataFrame:
    masked_prepared = _prepare_decoding_frame(masked_df)
    unmasked_prepared = _prepare_decoding_frame(unmasked_df)

    masked_columns = [
        "decoding_key",
        "config_label",
        "config_order",
        *masking_settings.identity_columns,
        "group",
        "total",
        *masking_settings.regard_columns,
    ]
    unmasked_columns = ["decoding_key", "group", "total", *masking_settings.regard_columns]
    masked_rename_map = {
        "total": "masked_total",
        **{column: f"masked_{column}" for column in masking_settings.regard_columns},
    }
    unmasked_rename_map = {
        "total": "unmasked_total",
        **{column: f"unmasked_{column}" for column in masking_settings.regard_columns},
    }
    masked_subset = cast(pd.DataFrame, masked_prepared.loc[:, masked_columns])
    unmasked_subset = cast(pd.DataFrame, unmasked_prepared.loc[:, unmasked_columns])
    masked_renamed = _rename_frame_columns(masked_subset, masked_rename_map)
    unmasked_renamed = _rename_frame_columns(unmasked_subset, unmasked_rename_map)

    merged = masked_renamed.merge(
        unmasked_renamed,
        on=["decoding_key", "group"],
        how="inner",
        validate="one_to_one",
    )
    for column in masking_settings.regard_columns:
        merged[f"delta_{column}"] = merged[f"unmasked_{column}"] - merged[f"masked_{column}"]
    return merged.sort_values(["config_order", "group"]).reset_index(drop=True)


def extract_key_trace(gap_comparison_df: pd.DataFrame) -> pd.DataFrame:
    trace = cast(
        pd.DataFrame,
        gap_comparison_df[
            (gap_comparison_df["prompt_type"] == masking_settings.key_trace["prompt_type"])
            & (gap_comparison_df["group_a"] == masking_settings.key_trace["group_a"])
            & (gap_comparison_df["group_b"] == masking_settings.key_trace["group_b"])
        ].copy(),
    )
    return trace.sort_values(by=["config_order"]).reset_index(drop=True)


def summarize_masking_sensitivity(
    gap_comparison_df: pd.DataFrame,
    distribution_comparison_df: pd.DataFrame,
    key_trace_df: pd.DataFrame,
    *,
    masked_bundle: MaskingBundle,
    unmasked_bundle: MaskingBundle,
) -> dict[str, Any]:
    sign_flip_count = (
        sum(bool(value) for value in gap_comparison_df["sign_changed"].tolist())
        if not gap_comparison_df.empty
        else 0
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
            "delta_gap_neg": _require_float(max_gap_shift_series["delta_gap_neg"]),
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
            "masked_gap_neg": _require_float(row["masked_gap_neg"]),
            "unmasked_gap_neg": _require_float(row["unmasked_gap_neg"]),
        }
        for row in _dataframe_records(top_gap_shifts)
    ]

    key_trace_sign_flip = (
        bool(key_trace_df["sign_changed"].any()) if not key_trace_df.empty else False
    )
    key_trace_positive_all = (
        bool(
            (key_trace_df["masked_gap_neg"] > 0).all()
            and (key_trace_df["unmasked_gap_neg"] > 0).all()
        )
        if not key_trace_df.empty
        else False
    )

    if key_trace_positive_all and not key_trace_sign_flip:
        main_conclusion_changed = False
        main_conclusion_text = (
            "Masking does not change the main Week 3 conclusion. "
            "The highlighted description / Black man vs White woman gap stays positive "
            "across every decoding configuration under both masked and unmasked scoring. "
            "The main Week 3 takeaway remains stable."
        )
    else:
        main_conclusion_changed = True
        main_conclusion_text = (
            "Masking changes the highlighted Week 3 trace enough to weaken the original "
            "qualitative conclusion, so the main Week 3 takeaway is no longer fully stable."
        )

    return {
        "created_at_utc": _utc_now_iso(),
        "masked_bundle": {
            "combined_scores_path": str(masked_bundle.combined_scores_path.resolve()),
            "gap_metrics_path": str(masked_bundle.gap_metrics_path.resolve()),
            "regard_metrics_path": str(masked_bundle.regard_metrics_path.resolve()),
        },
        "unmasked_bundle": {
            "combined_scores_path": str(unmasked_bundle.combined_scores_path.resolve()),
            "gap_metrics_path": str(unmasked_bundle.gap_metrics_path.resolve()),
            "regard_metrics_path": str(unmasked_bundle.regard_metrics_path.resolve()),
        },
        "total_compared_decoding_configs": int(len(gap_comparison_df["decoding_key"].unique()))
        if not gap_comparison_df.empty
        else 0,
        "total_compared_prompt_type_group_pairs": int(len(gap_comparison_df))
        if not gap_comparison_df.empty
        else 0,
        "sign_flip_count": sign_flip_count,
        "max_absolute_delta_gap_neg": (
            max(abs(_require_float(value)) for value in gap_comparison_df["delta_gap_neg"].tolist())
            if not gap_comparison_df.empty
            else 0.0
        ),
        "largest_gap_shift": max_gap_shift_row,
        "top_5_largest_gap_shifts": top_gap_shift_records,
        "key_trace": {
            "prompt_type": masking_settings.key_trace["prompt_type"],
            "group_a": masking_settings.key_trace["group_a"],
            "group_b": masking_settings.key_trace["group_b"],
            "rows": [
                {
                    "config_label": str(row["config_label"]),
                    "masked_gap_neg": _require_float(row["masked_gap_neg"]),
                    "unmasked_gap_neg": _require_float(row["unmasked_gap_neg"]),
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
        "ethics_notice": (
            "Generated and scored text may contain offensive content. "
            "Do not publish or commit large raw dumps."
        ),
    }


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run_week5_masking_sensitivity(settings: Settings) -> MaskingArtifactPaths:
    masked_bundle = _build_masking_bundle(settings, masking_settings.masked_stem, use_masking=True)
    unmasked_bundle = _build_masking_bundle(
        settings, masking_settings.unmasked_stem, use_masking=False
    )

    masked_gap_df = pd.read_csv(masked_bundle.gap_metrics_path)
    unmasked_gap_df = pd.read_csv(unmasked_bundle.gap_metrics_path)
    masked_regard_df = pd.read_csv(masked_bundle.regard_metrics_path)
    unmasked_regard_df = pd.read_csv(unmasked_bundle.regard_metrics_path)

    gap_comparison_df = compare_gap_metrics(masked_gap_df, unmasked_gap_df)
    distribution_comparison_df = compare_regard_distributions(masked_regard_df, unmasked_regard_df)
    key_trace_df = extract_key_trace(gap_comparison_df)

    metrics_dir = settings.output_dir / "metrics"
    gap_comparison_path = _write_csv(
        gap_comparison_df,
        metrics_dir / "week5_masking_gap_comparison.csv",
    )
    distribution_comparison_path = _write_csv(
        distribution_comparison_df,
        metrics_dir / "week5_masking_distribution_comparison.csv",
    )
    key_trace_path = _write_csv(
        key_trace_df,
        metrics_dir / "week5_masking_key_trace.csv",
    )

    masking_summary = summarize_masking_sensitivity(
        gap_comparison_df=gap_comparison_df,
        distribution_comparison_df=distribution_comparison_df,
        key_trace_df=key_trace_df,
        masked_bundle=masked_bundle,
        unmasked_bundle=unmasked_bundle,
    )
    summary_path = metrics_dir / "week5_masking_summary.json"
    summary_path.write_text(json.dumps(masking_summary, indent=2, sort_keys=True), encoding="utf-8")

    artifact_paths = MaskingArtifactPaths(
        masked_bundle=masked_bundle,
        unmasked_bundle=unmasked_bundle,
        gap_comparison_path=gap_comparison_path,
        distribution_comparison_path=distribution_comparison_path,
        key_trace_path=key_trace_path,
        summary_path=summary_path,
    )

    return artifact_paths
