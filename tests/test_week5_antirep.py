from pathlib import Path

import pandas as pd
import pytest

from app.week5_antirep import (
    Week3Bundle,
    compare_gap_metrics,
    compare_quality_metrics,
    compare_regard_distributions,
    extract_key_trace,
    format_config_label,
    summarize_antirepetition,
)


def _quality_frame(
    no_repeat_ngram_size: int, repeated_delta: float, distinct2_delta: float
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoding_strategy": ["greedy", "top_k"],
            "do_sample": [False, True],
            "temperature": [None, None],
            "top_k": [None, 20],
            "top_p": [None, None],
            "no_repeat_ngram_size": [no_repeat_ngram_size, no_repeat_ngram_size],
            "n_generations": [100, 100],
            "distinct_1": [0.01, 0.05],
            "distinct_2": [0.02 + distinct2_delta, 0.30 + distinct2_delta],
            "repeated_3gram_rate": [0.99 + repeated_delta, 0.40 + repeated_delta],
            "longest_repetition_span": [0.0, 2.0],
            "distinct_1_ci_lower": [0.009, 0.04],
            "distinct_1_ci_upper": [0.011, 0.06],
            "distinct_2_ci_lower": [0.018, 0.28],
            "distinct_2_ci_upper": [0.022, 0.32],
            "repeated_3gram_rate_ci_lower": [0.98, 0.39],
            "repeated_3gram_rate_ci_upper": [1.00, 0.41],
            "longest_repetition_span_ci_lower": [0.0, 2.0],
            "longest_repetition_span_ci_upper": [0.0, 2.0],
        }
    )


def _regard_frame(no_repeat_ngram_size: int, negative_shift: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoding_strategy": ["greedy", "greedy"],
            "do_sample": [False, False],
            "temperature": [None, None],
            "top_k": [None, None],
            "top_p": [None, None],
            "no_repeat_ngram_size": [no_repeat_ngram_size, no_repeat_ngram_size],
            "group": ["Black man", "White woman"],
            "negative": [0.40 + negative_shift, 0.20 + negative_shift],
            "neutral": [0.10, 0.10],
            "positive": [0.40 - negative_shift, 0.60 - negative_shift],
            "other": [0.10, 0.10],
            "total": [100, 100],
        }
    )


def _gap_frame(no_repeat_ngram_size: int, key_trace_gap: float, other_gap: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoding_strategy": ["greedy", "top_k"],
            "do_sample": [False, True],
            "temperature": [None, None],
            "top_k": [None, 20],
            "top_p": [None, None],
            "no_repeat_ngram_size": [no_repeat_ngram_size, no_repeat_ngram_size],
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


def _bundle(tmp_path: Path, stem: str, no_repeat_ngram_size: int) -> Week3Bundle:
    base_path = tmp_path / stem
    return Week3Bundle(
        combined_scores_path=base_path.with_suffix(".parquet"),
        combined_manifest_path=base_path.with_suffix(".json"),
        regard_metrics_path=base_path.with_name(f"{stem}_regard.csv"),
        gap_metrics_path=base_path.with_name(f"{stem}_gaps.csv"),
        quality_metrics_path=base_path.with_name(f"{stem}_quality.csv"),
        summary_path=base_path.with_name(f"{stem}_summary.json"),
        score_files=[],
        no_repeat_ngram_size=no_repeat_ngram_size,
    )


def test_compare_quality_metrics_computes_expected_deltas() -> None:
    baseline_df = _quality_frame(no_repeat_ngram_size=0, repeated_delta=0.0, distinct2_delta=0.0)
    antirep_df = _quality_frame(no_repeat_ngram_size=3, repeated_delta=-0.10, distinct2_delta=0.05)

    comparison_df = compare_quality_metrics(baseline_df, antirep_df)

    assert comparison_df["config_label"].tolist() == ["Greedy", "Top-k 20"]
    assert comparison_df["delta_repeated_3gram_rate"].tolist() == [
        pytest.approx(-0.10),
        pytest.approx(-0.10),
    ]
    assert comparison_df["delta_distinct_2"].tolist() == [
        pytest.approx(0.05),
        pytest.approx(0.05),
    ]


def test_compare_regard_distributions_tracks_label_shifts() -> None:
    baseline_df = _regard_frame(no_repeat_ngram_size=0, negative_shift=0.0)
    antirep_df = _regard_frame(no_repeat_ngram_size=3, negative_shift=-0.05)

    comparison_df = compare_regard_distributions(baseline_df, antirep_df)

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
    baseline_df = _gap_frame(no_repeat_ngram_size=0, key_trace_gap=0.18, other_gap=0.10)
    antirep_df = _gap_frame(no_repeat_ngram_size=3, key_trace_gap=0.12, other_gap=-0.02)

    comparison_df = compare_gap_metrics(baseline_df, antirep_df)
    key_trace_df = extract_key_trace(comparison_df)

    assert comparison_df["sign_changed"].tolist() == [False, True]
    assert len(key_trace_df) == 1
    assert key_trace_df.iloc[0]["prompt_type"] == "description"
    assert key_trace_df.iloc[0]["group_a"] == "Black man"
    assert key_trace_df.iloc[0]["group_b"] == "White woman"
    assert key_trace_df.iloc[0]["delta_gap_neg"] == pytest.approx(-0.06)


def test_summarize_antirepetition_preserves_main_conclusion_when_key_trace_stays_positive(
    tmp_path: Path,
) -> None:
    quality_comparison_df = compare_quality_metrics(
        _quality_frame(no_repeat_ngram_size=0, repeated_delta=0.0, distinct2_delta=0.0),
        _quality_frame(no_repeat_ngram_size=3, repeated_delta=-0.08, distinct2_delta=0.03),
    )
    gap_comparison_df = compare_gap_metrics(
        _gap_frame(no_repeat_ngram_size=0, key_trace_gap=0.16, other_gap=0.10),
        _gap_frame(no_repeat_ngram_size=3, key_trace_gap=0.11, other_gap=0.08),
    )
    key_trace_df = extract_key_trace(gap_comparison_df)

    summary = summarize_antirepetition(
        quality_comparison_df=quality_comparison_df,
        gap_comparison_df=gap_comparison_df,
        key_trace_df=key_trace_df,
        baseline_bundle=_bundle(tmp_path, "baseline", no_repeat_ngram_size=0),
        antirep_bundle=_bundle(tmp_path, "antirep", no_repeat_ngram_size=3),
        masking_summary_path=None,
    )

    assert summary["main_week3_conclusion_changed"] is False
    assert summary["repetition_improved_config_count"] == 2
    assert summary["key_trace"]["stays_positive_across_configs"] is True
    assert summary["masking_integration"]["status"] == "pending"


def test_format_config_label_matches_week3_naming() -> None:
    assert format_config_label("greedy", None, None, None) == "Greedy"
    assert format_config_label("temperature", 1.3, None, None) == "Temperature 1.3"
    assert format_config_label("top_k", None, 20, None) == "Top-k 20"
    assert format_config_label("top_p", None, None, 0.9) == "Top-p 0.9"
