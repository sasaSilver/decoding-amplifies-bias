from collections import Counter
from collections.abc import Iterable, Sequence

import pandas as pd

DECODING_GROUP_COLUMNS = [
    "decoding_strategy",
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "no_repeat_ngram_size",
]

DECODING_COLUMN_DEFAULTS = {
    "decoding_strategy": "greedy",
    "do_sample": False,
    "temperature": None,
    "top_k": None,
    "top_p": None,
    "no_repeat_ngram_size": 0,
}


def tokenize_text(text: str) -> list[str]:
    return text.strip().split()


def ensure_decoding_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column_name, default_value in DECODING_COLUMN_DEFAULTS.items():
        if column_name not in normalized.columns:
            normalized[column_name] = default_value
    return normalized


def extract_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be at least 1.")
    if len(tokens) < n:
        return []

    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_distinct_n(texts: Iterable[str], n: int) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(extract_ngrams(tokenize_text(text), n))

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def compute_repeated_ngram_rate(texts: Iterable[str], n: int = 3) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(extract_ngrams(tokenize_text(text), n))

    if not all_ngrams:
        return 0.0

    counts = Counter(all_ngrams)
    repeated_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_occurrences / len(all_ngrams)


def compute_longest_repetition_span(text: str) -> int:
    tokens = tokenize_text(text)
    if len(tokens) < 2:
        return 0

    longest_run = 1
    current_run = 1

    for index in range(1, len(tokens)):
        if tokens[index] == tokens[index - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    if longest_run == 1:
        return 0

    return longest_run


def compute_quality_metrics(texts: Sequence[str]) -> dict[str, float]:
    if not texts:
        return {
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "repeated_3gram_rate": 0.0,
            "longest_repetition_span": 0.0,
        }

    return {
        "distinct_1": compute_distinct_n(texts, 1),
        "distinct_2": compute_distinct_n(texts, 2),
        "repeated_3gram_rate": compute_repeated_ngram_rate(texts, n=3),
        "longest_repetition_span": float(
            max(compute_longest_repetition_span(text) for text in texts)
        ),
    }


def compute_quality_metrics_by_decoding(
    df: pd.DataFrame,
    text_col: str = "completion_text",
) -> pd.DataFrame:
    normalized = ensure_decoding_columns(df)
    rows: list[dict[str, object]] = []

    for decoding_values, group_df in normalized.groupby(DECODING_GROUP_COLUMNS, dropna=False):
        row = dict(zip(DECODING_GROUP_COLUMNS, decoding_values, strict=True))
        metrics = compute_quality_metrics(group_df[text_col].tolist())
        row["n_generations"] = len(group_df)
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)
