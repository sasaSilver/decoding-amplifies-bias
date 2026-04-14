import json
from pathlib import Path

import pandas as pd
import pytest

from app.settings.settings import Settings
from app.week5_masking import (
    MaskingBundle,
    _collect_score_files_for_masking,
    compare_gap_metrics,
    compare_regard_distributions,
    extract_key_trace,
    format_config_label,
    summarize_masking_sensitivity,
)


def _gap_frame(use_masking: bool, key_trace_gap: float, other_gap: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoding_strategy": ["greedy", "top_k"],
            "do_sample": [False, True],
            "temperature": [None, None],
            "top_k": [None, 20],
            "top_p": [None, None],
            "no_repeat_ngram_size": [0, 0],
            "prompt_type": ["description", "occupation"],
            "group_a": ["Black man", "Black woman"],
            "group_b": ["White woman", "White woman"],
            "gap_neg": [key_trace_gap, other_gap],
            "ci_lower": [key_trace_gap - 0.05, other_gap - 0.05],
            "ci_upper": [key_trace_gap + 0.05, other_gap + 0.05],
            "p_neg_a": [0.4, 0.3],
            "p_neg_b": [0.2, 0.2],
            "n_samples_a": [100, 100],
            "n_samples_b": [100, 100],
        }
    )


def _regard_frame(use_masking: bool, negative_shift: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoding_strategy": ["greedy", "greedy"],
            "do_sample": [False, False],
            "temperature": [None, None],
            "top_k": [None, None],
            "top_p": [None, None],
            "no_repeat_ngram_size": [0, 0],
            "group": ["Black man", "White woman"],
            "negative": [0.40 + negative_shift, 0.20 + negative_shift],
            "neutral": [0.10, 0.10],
            "positive": [0.40 - negative_shift, 0.60 - negative_shift],
            "other": [0.10, 0.10],
            "total": [100, 100],
        }
    )


def _bundle(tmp_path: Path, stem: str, use_masking: bool) -> MaskingBundle:
    base_path = tmp_path / stem
    return MaskingBundle(
        combined_scores_path=base_path.with_suffix(".parquet"),
        combined_manifest_path=base_path.with_suffix(".json"),
        regard_metrics_path=base_path.with_name(f"{stem}_regard.csv"),
        gap_metrics_path=base_path.with_name(f"{stem}_gaps.csv"),
        score_files=[],
        use_masking=use_masking,
    )


def test_collect_score_files_for_masking_excludes_antirepetition_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    scores_dir = output_dir / "scores"
    manifests_dir = output_dir / "manifests"
    scores_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    settings = Settings(
        output_dir=output_dir,
        generations_path=output_dir / "generations",
    )
    model_reference = settings.scoring.resolved_model_reference()

    baseline_generation_key = "baseline_generation"
    antirep_generation_key = "antirep_generation"
    pd.DataFrame({"label": ["positive"]}).to_parquet(
        scores_dir / "baseline_score.parquet", index=False
    )
    pd.DataFrame({"label": ["positive"]}).to_parquet(
        scores_dir / "antirep_score.parquet", index=False
    )

    (manifests_dir / "baseline_score.json").write_text(
        (
            "{\n"
            f'  "generations_cache_key": "{baseline_generation_key}",\n'
            f'  "model_reference": "{model_reference}",\n'
            '  "use_masking": true\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    (manifests_dir / "antirep_score.json").write_text(
        (
            "{\n"
            f'  "generations_cache_key": "{antirep_generation_key}",\n'
            f'  "model_reference": "{model_reference}",\n'
            '  "use_masking": true\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    baseline_generation_manifest = {
        "model_name": "gpt2",
        "prompt_bank_digest": "baseline",
        "max_new_tokens": 40,
        "n_samples_per_prompt": 50,
        "seeds": [0, 1, 2],
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "no_repeat_ngram_size": 0,
        },
    }
    antirep_generation_manifest = {
        "model_name": "gpt2",
        "prompt_bank_digest": "antirep",
        "max_new_tokens": 40,
        "n_samples_per_prompt": 50,
        "seeds": [0, 1, 2],
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "no_repeat_ngram_size": 3,
        },
    }
    (manifests_dir / f"{baseline_generation_key}.json").write_text(
        json.dumps(baseline_generation_manifest, indent=2),
        encoding="utf-8",
    )
    (manifests_dir / f"{antirep_generation_key}.json").write_text(
        json.dumps(antirep_generation_manifest, indent=2),
        encoding="utf-8",
    )

    artifacts = _collect_score_files_for_masking(
        settings,
        use_masking=True,
        no_repeat_ngram_size=0,
    )

    assert len(artifacts) == 1
    assert artifacts[0].score_path.name == "baseline_score.parquet"


def test_compare_gap_metrics_computes_expected_deltas() -> None:
    masked_df = _gap_frame(use_masking=True, key_trace_gap=0.18, other_gap=0.10)
    unmasked_df = _gap_frame(use_masking=False, key_trace_gap=0.15, other_gap=0.08)

    comparison_df = compare_gap_metrics(masked_df, unmasked_df)

    assert comparison_df["config_label"].tolist() == ["Greedy", "Top-k 20"]
    assert comparison_df["delta_gap_neg"].tolist() == [
        pytest.approx(-0.03),
        pytest.approx(-0.02),
    ]
    assert comparison_df["sign_changed"].tolist() == [False, False]


def test_compare_regard_distributions_tracks_label_shifts() -> None:
    masked_df = _regard_frame(use_masking=True, negative_shift=0.0)
    unmasked_df = _regard_frame(use_masking=False, negative_shift=-0.05)

    comparison_df = compare_regard_distributions(masked_df, unmasked_df)

    assert comparison_df["group"].tolist() == ["Black man", "White woman"]
    assert comparison_df["delta_negative"].tolist() == [
        pytest.approx(-0.05),
        pytest.approx(-0.05),
    ]
    assert comparison_df["delta_positive"].tolist() == [
        pytest.approx(0.05),
        pytest.approx(0.05),
    ]


def test_compare_gap_metrics_marks_sign_changes_and_extracts_key_trace() -> None:
    masked_df = _gap_frame(use_masking=True, key_trace_gap=0.18, other_gap=0.10)
    unmasked_df = _gap_frame(use_masking=False, key_trace_gap=-0.05, other_gap=0.08)

    comparison_df = compare_gap_metrics(masked_df, unmasked_df)
    key_trace_df = extract_key_trace(comparison_df)

    assert comparison_df["sign_changed"].tolist() == [True, False]
    assert len(key_trace_df) == 1
    assert key_trace_df.iloc[0]["prompt_type"] == "description"
    assert key_trace_df.iloc[0]["group_a"] == "Black man"
    assert key_trace_df.iloc[0]["group_b"] == "White woman"
    assert key_trace_df.iloc[0]["delta_gap_neg"] == pytest.approx(-0.23)


def test_summarize_masking_sensitivity_preserves_main_conclusion_when_key_trace_stays_positive(
    tmp_path: Path,
) -> None:
    gap_comparison_df = compare_gap_metrics(
        _gap_frame(use_masking=True, key_trace_gap=0.16, other_gap=0.10),
        _gap_frame(use_masking=False, key_trace_gap=0.14, other_gap=0.09),
    )
    distribution_comparison_df = compare_regard_distributions(
        _regard_frame(use_masking=True, negative_shift=0.0),
        _regard_frame(use_masking=False, negative_shift=-0.02),
    )
    key_trace_df = extract_key_trace(gap_comparison_df)

    summary = summarize_masking_sensitivity(
        gap_comparison_df=gap_comparison_df,
        distribution_comparison_df=distribution_comparison_df,
        key_trace_df=key_trace_df,
        masked_bundle=_bundle(tmp_path, "masked", use_masking=True),
        unmasked_bundle=_bundle(tmp_path, "unmasked", use_masking=False),
    )

    assert summary["main_week3_conclusion_changed"] is False
    assert summary["sign_flip_count"] == 0
    assert summary["key_trace"]["stays_positive_across_configs"] is True


def test_summarize_masking_sensitivity_detects_sign_flips(
    tmp_path: Path,
) -> None:
    gap_comparison_df = compare_gap_metrics(
        _gap_frame(use_masking=True, key_trace_gap=0.16, other_gap=0.10),
        _gap_frame(use_masking=False, key_trace_gap=-0.05, other_gap=-0.02),
    )
    distribution_comparison_df = compare_regard_distributions(
        _regard_frame(use_masking=True, negative_shift=0.0),
        _regard_frame(use_masking=False, negative_shift=-0.02),
    )
    key_trace_df = extract_key_trace(gap_comparison_df)

    summary = summarize_masking_sensitivity(
        gap_comparison_df=gap_comparison_df,
        distribution_comparison_df=distribution_comparison_df,
        key_trace_df=key_trace_df,
        masked_bundle=_bundle(tmp_path, "masked", use_masking=True),
        unmasked_bundle=_bundle(tmp_path, "unmasked", use_masking=False),
    )

    assert summary["main_week3_conclusion_changed"] is True
    assert summary["sign_flip_count"] == 2
    assert summary["key_trace"]["has_sign_flip"] is True


def test_format_config_label_matches_week3_naming() -> None:
    assert format_config_label("greedy", None, None, None) == "Greedy"
    assert format_config_label("temperature", 1.3, None, None) == "Temperature 1.3"
    assert format_config_label("top_k", None, 20, None) == "Top-k 20"
    assert format_config_label("top_p", None, None, 0.9) == "Top-p 0.9"
