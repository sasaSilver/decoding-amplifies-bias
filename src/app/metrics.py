import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models import RegardDistribution
from .quality import (
    DECODING_GROUP_COLUMNS,
    compute_quality_metrics,
    ensure_decoding_columns,
)
from .scoring import RegardLabelEnum


def compute_regard_distribution(
    df: pd.DataFrame,
    group_col: str = "demographic",
    label_col: str = "regard_label",
) -> dict[str, dict]:
    """Compute regard distribution per group.

    Args:
        df: DataFrame with scored generations.
        group_col: Column name for grouping (e.g., demographic).
        label_col: Column name for regard labels.

    Returns:
        Dictionary mapping group names to RegardDistribution dicts.
    """
    distributions = {}

    for group_name, group_df in df.groupby(group_col):
        total = len(group_df)
        if total == 0:
            continue

        label_counts = group_df[label_col].value_counts()
        negative = label_counts.get(RegardLabelEnum.NEGATIVE, 0) / total
        neutral = label_counts.get(RegardLabelEnum.NEUTRAL, 0) / total
        positive = label_counts.get(RegardLabelEnum.POSITIVE, 0) / total
        other = label_counts.get(RegardLabelEnum.OTHER, 0) / total

        distributions[group_name] = RegardDistribution(
            negative=negative,
            neutral=neutral,
            positive=positive,
            other=other,
            total=total,
        ).model_dump()

    return distributions


def compute_negative_regard_gap(
    df: pd.DataFrame,
    group_col: str = "demographic",
    label_col: str = "regard_label",
    prompt_type_col: str = "prompt_type",
) -> pd.DataFrame:
    """Compute negative-regard gaps between groups per prompt type.

    The gap is computed as: Δneg = P(neg|group_A) - P(neg|group_B)

    Args:
        df: DataFrame with scored generations.
        group_col: Column name for demographic groups.
        label_col: Column name for regard labels.
        prompt_type_col: Column name for prompt types.

    Returns:
        DataFrame with columns: prompt_type, group_a, group_b, gap_neg,
        p_neg_a, p_neg_b, n_samples_a, n_samples_b.
    """
    results = []

    # Get unique prompt types and groups
    prompt_types = df[prompt_type_col].unique()
    groups = df[group_col].unique()

    for prompt_type in prompt_types:
        prompt_df = df[df[prompt_type_col] == prompt_type]

        for i, group_a in enumerate(groups):
            for group_b in groups[i + 1 :]:
                # Get data for each group
                df_a = prompt_df[prompt_df[group_col] == group_a]
                df_b = prompt_df[prompt_df[group_col] == group_b]

                if len(df_a) == 0 or len(df_b) == 0:
                    continue

                # Compute P(neg) for each group
                p_neg_a = (df_a[label_col] == RegardLabelEnum.NEGATIVE).mean()
                p_neg_b = (df_b[label_col] == RegardLabelEnum.NEGATIVE).mean()

                # Compute gap
                gap_neg = p_neg_a - p_neg_b

                results.append(
                    {
                        "prompt_type": prompt_type,
                        "group_a": group_a,
                        "group_b": group_b,
                        "gap_neg": gap_neg,
                        "p_neg_a": p_neg_a,
                        "p_neg_b": p_neg_b,
                        "n_samples_a": len(df_a),
                        "n_samples_b": len(df_b),
                    }
                )

    return pd.DataFrame(results)


def compute_bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Array of values to bootstrap.
        n_bootstrap: Number of bootstrap samples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if len(values) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(random_seed)
    n = len(values)

    # Bootstrap samples
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    # Compute CI
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def compute_bootstrap_ci_for_gap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_col: str = "regard_label",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap CI for negative-regard gap between two groups.

    Args:
        df_a: DataFrame for group A.
        df_b: DataFrame for group B.
        label_col: Column name for regard labels.
        n_bootstrap: Number of bootstrap samples.
        ci_level: Confidence level.
        random_seed: Random seed.

    Returns:
        Tuple of (lower_bound, upper_bound) for the gap.
    """
    if len(df_a) == 0 or len(df_b) == 0:
        return 0.0, 0.0

    # Convert to binary (1 for negative, 0 otherwise)
    values_a = np.asarray((df_a[label_col] == RegardLabelEnum.NEGATIVE).astype(float))
    values_b = np.asarray((df_b[label_col] == RegardLabelEnum.NEGATIVE).astype(float))

    rng = np.random.default_rng(random_seed)
    n_a = len(values_a)
    n_b = len(values_b)

    # Bootstrap samples
    bootstrap_gaps = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample_a = rng.choice(values_a, size=n_a, replace=True)
        sample_b = rng.choice(values_b, size=n_b, replace=True)
        bootstrap_gaps[i] = np.mean(sample_a) - np.mean(sample_b)

    # Compute CI
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_gaps, 100 * alpha / 2)
    upper = np.percentile(bootstrap_gaps, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def compute_bootstrap_ci_for_dataframe_metric(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    if len(df) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(random_seed)
    bootstrap_values = np.zeros(n_bootstrap)

    for index in range(n_bootstrap):
        sample_indices = rng.choice(len(df), size=len(df), replace=True)
        sampled_df = df.iloc[sample_indices]
        bootstrap_values[index] = metric_fn(sampled_df)

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def compute_regard_distribution_by_decoding(
    df: pd.DataFrame,
    group_col: str = "demographic",
    label_col: str = "regard_label",
) -> pd.DataFrame:
    normalized = ensure_decoding_columns(df)
    rows: list[dict[str, object]] = []

    for decoding_values, decoding_df in normalized.groupby(DECODING_GROUP_COLUMNS, dropna=False):
        decoding_row = dict(zip(DECODING_GROUP_COLUMNS, decoding_values, strict=True))
        distributions = compute_regard_distribution(
            decoding_df,
            group_col=group_col,
            label_col=label_col,
        )
        for group_name, distribution in distributions.items():
            rows.append(
                {
                    **decoding_row,
                    "group": group_name,
                    **distribution,
                }
            )

    return pd.DataFrame(rows)


def compute_negative_regard_gap_by_decoding(
    df: pd.DataFrame,
    group_col: str = "demographic",
    label_col: str = "regard_label",
    prompt_type_col: str = "prompt_type",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    normalized = ensure_decoding_columns(df)
    rows: list[dict[str, object]] = []

    for decoding_values, decoding_df in normalized.groupby(DECODING_GROUP_COLUMNS, dropna=False):
        decoding_row = dict(zip(DECODING_GROUP_COLUMNS, decoding_values, strict=True))
        prompt_types = decoding_df[prompt_type_col].unique()
        groups = decoding_df[group_col].unique()

        for prompt_type in prompt_types:
            prompt_df = decoding_df[decoding_df[prompt_type_col] == prompt_type]

            for index, group_a in enumerate(groups):
                for group_b in groups[index + 1 :]:
                    df_a = prompt_df[prompt_df[group_col] == group_a]
                    df_b = prompt_df[prompt_df[group_col] == group_b]

                    if len(df_a) == 0 or len(df_b) == 0:
                        continue

                    p_neg_a = (df_a[label_col] == RegardLabelEnum.NEGATIVE).mean()
                    p_neg_b = (df_b[label_col] == RegardLabelEnum.NEGATIVE).mean()
                    gap_neg = p_neg_a - p_neg_b
                    ci_lower, ci_upper = compute_bootstrap_ci_for_gap(
                        df_a,
                        df_b,
                        label_col=label_col,
                        n_bootstrap=n_bootstrap,
                        ci_level=ci_level,
                    )

                    rows.append(
                        {
                            **decoding_row,
                            "prompt_type": prompt_type,
                            "group_a": group_a,
                            "group_b": group_b,
                            "gap_neg": gap_neg,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "p_neg_a": p_neg_a,
                            "p_neg_b": p_neg_b,
                            "n_samples_a": len(df_a),
                            "n_samples_b": len(df_b),
                        }
                    )

    return pd.DataFrame(rows)


def compute_quality_metrics_with_ci_by_decoding(
    df: pd.DataFrame,
    text_col: str = "completion_text",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    normalized = ensure_decoding_columns(df)
    rows: list[dict[str, object]] = []
    metric_names = [
        "distinct_1",
        "distinct_2",
        "repeated_3gram_rate",
        "longest_repetition_span",
    ]
    alpha = 1 - ci_level

    for decoding_values, decoding_df in normalized.groupby(DECODING_GROUP_COLUMNS, dropna=False):
        decoding_row = dict(zip(DECODING_GROUP_COLUMNS, decoding_values, strict=True))
        texts = decoding_df[text_col].tolist()
        metrics = compute_quality_metrics(texts)
        row: dict[str, object] = {
            **decoding_row,
            "n_generations": len(decoding_df),
            **metrics,
        }

        if texts:
            rng = np.random.default_rng(42)
            text_array = np.asarray(texts, dtype=object)
            bootstrap_metrics = {
                metric_name: np.zeros(n_bootstrap)
                for metric_name in metric_names
            }

            for bootstrap_index in range(n_bootstrap):
                sample_indices = rng.choice(len(text_array), size=len(text_array), replace=True)
                sampled_texts = text_array[sample_indices].tolist()
                sampled_metrics = compute_quality_metrics(sampled_texts)

                for metric_name in metric_names:
                    bootstrap_metrics[metric_name][bootstrap_index] = sampled_metrics[metric_name]

            for metric_name in metric_names:
                row[f"{metric_name}_ci_lower"] = float(
                    np.percentile(bootstrap_metrics[metric_name], 100 * alpha / 2)
                )
                row[f"{metric_name}_ci_upper"] = float(
                    np.percentile(bootstrap_metrics[metric_name], 100 * (1 - alpha / 2))
                )
        else:
            for metric_name in metric_names:
                row[f"{metric_name}_ci_lower"] = 0.0
                row[f"{metric_name}_ci_upper"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def compute_week3_metrics(
    scores_path: Path,
    output_dir: Path,
    n_bootstrap: int = 1000,
    quality_n_bootstrap: int | None = None,
    ci_level: float = 0.95,
) -> dict[str, Path]:
    df = ensure_decoding_columns(pd.read_parquet(scores_path))
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cache_key = scores_path.stem
    resolved_quality_n_bootstrap = quality_n_bootstrap or n_bootstrap

    output_paths: dict[str, Path] = {}

    regard_path = metrics_dir / f"{cache_key}_week3_regard_distributions.csv"
    compute_regard_distribution_by_decoding(df).to_csv(regard_path, index=False)
    output_paths["week3_regard_distributions"] = regard_path

    gap_path = metrics_dir / f"{cache_key}_week3_negative_gaps_with_ci.csv"
    compute_negative_regard_gap_by_decoding(
        df,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
    ).to_csv(gap_path, index=False)
    output_paths["week3_negative_gaps_with_ci"] = gap_path

    quality_path = metrics_dir / f"{cache_key}_week3_quality_metrics_with_ci.csv"
    compute_quality_metrics_with_ci_by_decoding(
        df,
        n_bootstrap=resolved_quality_n_bootstrap,
        ci_level=ci_level,
    ).to_csv(quality_path, index=False)
    output_paths["week3_quality_metrics_with_ci"] = quality_path

    summary = {
        "cache_key": cache_key,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "total_samples": len(df),
        "n_bootstrap": n_bootstrap,
        "quality_n_bootstrap": resolved_quality_n_bootstrap,
        "ci_level": ci_level,
        "n_decoding_configs": int(df[DECODING_GROUP_COLUMNS].drop_duplicates().shape[0]),
    }
    summary_path = metrics_dir / f"{cache_key}_week3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    output_paths["week3_summary"] = summary_path

    return output_paths


def compute_baseline_metrics(
    scores_path: Path,
    output_dir: Path,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, Path]:
    """Compute baseline bias metrics from scored generations.

    Args:
        scores_path: Path to scored generations parquet file.
        output_dir: Directory for metric outputs.
        n_bootstrap: Number of bootstrap samples for CI.
        ci_level: Confidence level for CI.

    Returns:
        Dictionary mapping metric type to output file paths.
    """
    # Load scored data
    df = pd.read_parquet(scores_path)

    # Create output directory
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Extract cache key
    cache_key = scores_path.stem

    output_paths: dict[str, Path] = {}

    # 1. Regard distributions per group
    distributions = compute_regard_distribution(df)
    dist_data = []
    for group, dist in distributions.items():
        dist_data.append(
            {
                "group": group,
                **dist,
            }
        )
    dist_df = pd.DataFrame(dist_data)
    dist_path = metrics_dir / f"{cache_key}_regard_distributions.csv"
    dist_df.to_csv(dist_path, index=False)
    output_paths["regard_distributions"] = dist_path

    # 2. Negative-regard gaps per prompt type
    gaps_df = compute_negative_regard_gap(df)
    gaps_path = metrics_dir / f"{cache_key}_negative_gaps.csv"
    gaps_df.to_csv(gaps_path, index=False)
    output_paths["negative_gaps"] = gaps_path

    # 3. Negative-regard gaps with bootstrap CIs
    gaps_with_ci = []
    prompt_types = df["prompt_type"].unique()
    groups = df["demographic"].unique()

    for prompt_type in prompt_types:
        prompt_df = df[df["prompt_type"] == prompt_type]

        for i, group_a in enumerate(groups):
            for group_b in groups[i + 1 :]:
                df_a = prompt_df[prompt_df["demographic"] == group_a]
                df_b = prompt_df[prompt_df["demographic"] == group_b]

                if len(df_a) == 0 or len(df_b) == 0:
                    continue

                p_neg_a = (df_a["regard_label"] == RegardLabelEnum.NEGATIVE).mean()
                p_neg_b = (df_b["regard_label"] == RegardLabelEnum.NEGATIVE).mean()
                gap = p_neg_a - p_neg_b

                ci_lower, ci_upper = compute_bootstrap_ci_for_gap(
                    df_a, df_b, n_bootstrap=n_bootstrap, ci_level=ci_level
                )

                gaps_with_ci.append(
                    {
                        "prompt_type": prompt_type,
                        "group_a": group_a,
                        "group_b": group_b,
                        "gap_neg": gap,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "p_neg_a": p_neg_a,
                        "p_neg_b": p_neg_b,
                        "n_samples_a": len(df_a),
                        "n_samples_b": len(df_b),
                    }
                )

    gaps_ci_df = pd.DataFrame(gaps_with_ci)
    gaps_ci_path = metrics_dir / f"{cache_key}_negative_gaps_with_ci.csv"
    gaps_ci_df.to_csv(gaps_ci_path, index=False)
    output_paths["negative_gaps_with_ci"] = gaps_ci_path

    # 4. Overall label distribution (sanity check)
    label_counts = df["regard_label"].value_counts()
    label_dist = label_counts / len(df)
    overall_dist_path = metrics_dir / f"{cache_key}_overall_label_distribution.csv"
    label_dist.to_csv(overall_dist_path)
    output_paths["overall_label_distribution"] = overall_dist_path

    # 5. Summary statistics
    summary = {
        "cache_key": cache_key,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "total_samples": len(df),
        "unique_prompts": df["prompt_id"].nunique(),
        "unique_demographics": df["demographic"].nunique(),
        "unique_prompt_types": df["prompt_type"].nunique(),
        "label_distribution": label_dist.to_dict(),
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
    }
    summary_path = metrics_dir / f"{cache_key}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    output_paths["summary"] = summary_path

    return output_paths
