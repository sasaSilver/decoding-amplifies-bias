import pandas as pd
import pytest

from app.quality import (
    compute_distinct_n,
    compute_longest_repetition_span,
    compute_quality_metrics,
    compute_quality_metrics_by_decoding,
    compute_repeated_ngram_rate,
    extract_ngrams,
    tokenize_text,
)


def test_tokenize_text_splits_on_whitespace() -> None:
    assert tokenize_text("alpha   beta\ngamma") == ["alpha", "beta", "gamma"]


def test_extract_ngrams_returns_expected_windows() -> None:
    tokens = ["a", "b", "c", "d"]
    assert extract_ngrams(tokens, 2) == [("a", "b"), ("b", "c"), ("c", "d")]


def test_compute_distinct_n() -> None:
    texts = ["a b c", "a b d"]
    assert compute_distinct_n(texts, 1) == pytest.approx(4 / 6)
    assert compute_distinct_n(texts, 2) == pytest.approx(3 / 4)


def test_compute_repeated_ngram_rate() -> None:
    texts = ["a b c a b c", "a b d"]
    assert compute_repeated_ngram_rate(texts, n=3) == pytest.approx(1 / 5)


def test_compute_longest_repetition_span() -> None:
    assert compute_longest_repetition_span("alpha alpha alpha beta") == 3
    assert compute_longest_repetition_span("alpha beta gamma") == 0


def test_compute_quality_metrics() -> None:
    metrics = compute_quality_metrics(["alpha alpha alpha beta", "alpha beta gamma"])
    assert set(metrics) == {
        "distinct_1",
        "distinct_2",
        "repeated_3gram_rate",
        "longest_repetition_span",
    }
    assert metrics["longest_repetition_span"] == 3.0


def test_compute_quality_metrics_by_decoding() -> None:
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

    metrics_df = compute_quality_metrics_by_decoding(df)

    assert len(metrics_df) == 2
    assert "distinct_1" in metrics_df.columns
    assert "distinct_2" in metrics_df.columns
    assert "repeated_3gram_rate" in metrics_df.columns
    assert "longest_repetition_span" in metrics_df.columns
    assert "n_generations" in metrics_df.columns
