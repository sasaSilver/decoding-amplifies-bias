from pathlib import Path

import pandas as pd

from app.metrics import (
    compute_quality_metrics_with_ci_by_decoding,
    compute_week3_metrics,
)
from app.quality import compute_quality_metrics_by_decoding


def test_quality_metrics_by_decoding_accepts_legacy_greedy_records() -> None:
    df = pd.DataFrame(
        {
            "completion_text": ["alpha alpha alpha beta", "alpha beta gamma"],
        }
    )

    metrics_df = compute_quality_metrics_by_decoding(df)

    assert len(metrics_df) == 1
    assert metrics_df.iloc[0]["decoding_strategy"] == "greedy"
    assert not bool(metrics_df.iloc[0]["do_sample"])


def test_compute_quality_metrics_with_ci_by_decoding() -> None:
    df = pd.DataFrame(
        {
            "decoding_strategy": ["greedy", "greedy", "top_k", "top_k"],
            "do_sample": [False, False, True, True],
            "temperature": [None, None, None, None],
            "top_k": [None, None, 20, 20],
            "top_p": [None, None, None, None],
            "no_repeat_ngram_size": [0, 0, 0, 0],
            "completion_text": [
                "alpha alpha alpha beta",
                "alpha beta gamma",
                "delta epsilon zeta",
                "delta epsilon delta epsilon",
            ],
        }
    )

    metrics_df = compute_quality_metrics_with_ci_by_decoding(df, n_bootstrap=50, ci_level=0.95)

    assert len(metrics_df) == 2
    assert "distinct_1_ci_lower" in metrics_df.columns
    assert "distinct_2_ci_upper" in metrics_df.columns
    assert "repeated_3gram_rate_ci_lower" in metrics_df.columns
    assert "longest_repetition_span_ci_upper" in metrics_df.columns


def test_compute_week3_metrics(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "prompt_id": ["p1", "p2", "p3", "p4"],
            "prompt_type": ["occupation", "occupation", "occupation", "occupation"],
            "demographic": ["group_a", "group_b", "group_a", "group_b"],
            "completion_text": [
                "alpha alpha alpha beta",
                "alpha beta gamma",
                "delta epsilon zeta",
                "delta epsilon delta epsilon",
            ],
            "regard_label": ["negative", "positive", "negative", "neutral"],
            "decoding_strategy": ["greedy", "greedy", "top_k", "top_k"],
            "do_sample": [False, False, True, True],
            "temperature": [None, None, None, None],
            "top_k": [None, None, 20, 20],
            "top_p": [None, None, None, None],
            "no_repeat_ngram_size": [0, 0, 0, 0],
        }
    )
    scores_path = tmp_path / "scores.parquet"
    df.to_parquet(scores_path, index=False)

    metric_paths = compute_week3_metrics(
        scores_path=scores_path,
        output_dir=tmp_path,
        n_bootstrap=50,
        ci_level=0.95,
    )

    assert "week3_regard_distributions" in metric_paths
    assert "week3_negative_gaps_with_ci" in metric_paths
    assert "week3_quality_metrics_with_ci" in metric_paths
    assert "week3_summary" in metric_paths

    for path in metric_paths.values():
        assert path.exists()

    quality_df = pd.read_csv(metric_paths["week3_quality_metrics_with_ci"])
    assert "distinct_1" in quality_df.columns
    assert "distinct_1_ci_lower" in quality_df.columns

    gap_df = pd.read_csv(metric_paths["week3_negative_gaps_with_ci"])
    assert "decoding_strategy" in gap_df.columns
    assert "ci_lower" in gap_df.columns
