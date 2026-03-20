import json
from pathlib import Path

import pandas as pd

from .models import SanityCheckResult
from .scoring import RegardLabelEnum


def verify_label_distribution(
    df: pd.DataFrame,
    label_col: str = "regard_label",
    min_samples_per_label: int = 10,
    expected_labels: list[str] | None = None,
) -> SanityCheckResult:
    if expected_labels is None:
        expected_labels = [label.value for label in RegardLabelEnum]

    total_samples = len(df)
    if total_samples == 0:
        return SanityCheckResult(
            check_name="label_distribution",
            passed=False,
            message="No samples found in dataframe.",
            details={"total_samples": 0},
        )

    label_counts = df[label_col].value_counts()
    label_proportions = label_counts / total_samples

    missing_labels = set(expected_labels) - set(label_counts.index)
    if missing_labels:
        return SanityCheckResult(
            check_name="label_distribution",
            passed=False,
            message=f"Missing expected labels: {missing_labels}",
            details={
                "total_samples": total_samples,
                "label_counts": label_counts.to_dict(),
                "missing_labels": list(missing_labels),
            },
        )

    low_sample_labels = [
        label for label, count in label_counts.items() if count < min_samples_per_label
    ]
    if low_sample_labels:
        return SanityCheckResult(
            check_name="label_distribution",
            passed=False,
            message=f"Labels with fewer than {min_samples_per_label} samples: {low_sample_labels}",
            details={
                "total_samples": total_samples,
                "label_counts": label_counts.to_dict(),
                "low_sample_labels": low_sample_labels,
            },
        )

    if len(label_counts) == 1:
        return SanityCheckResult(
            check_name="label_distribution",
            passed=False,
            message="All samples have the same label.",
            details={
                "total_samples": total_samples,
                "label_counts": label_counts.to_dict(),
            },
        )

    return SanityCheckResult(
        check_name="label_distribution",
        passed=True,
        message="Label distribution looks reasonable.",
        details={
            "total_samples": total_samples,
            "label_counts": label_counts.to_dict(),
            "label_proportions": label_proportions.to_dict(),
        },
    )


def spot_check_scored_outputs(
    df: pd.DataFrame,
    n_samples: int = 20,
    random_seed: int = 42,
    text_col: str = "completion_text",
    label_col: str = "regard_label",
    demographic_col: str = "demographic",
    prompt_text_col: str = "prompt_text",
) -> list[dict]:
    if len(df) == 0:
        return []

    samples = []
    labels = df[label_col].unique()

    samples_per_label = max(1, n_samples // len(labels))

    for label in labels:
        label_df = df[df[label_col] == label]
        n_to_sample = min(samples_per_label, len(label_df))

        if n_to_sample > 0:
            label_samples = label_df.sample(
                n=n_to_sample,
                random_state=random_seed,
            )
            samples.extend(label_samples.to_dict("records"))

    if len(samples) < n_samples:
        remaining = n_samples - len(samples)
        additional_samples = df.sample(
            n=remaining,
            random_state=random_seed + 1,
        )
        samples.extend(additional_samples.to_dict("records"))

    formatted_samples = []
    for i, sample in enumerate(samples):
        formatted_samples.append(
            {
                "sample_index": i,
                "prompt_text": sample[prompt_text_col][:200] + "..."
                if len(sample[prompt_text_col]) > 200
                else sample[prompt_text_col],
                "demographic": sample[demographic_col],
                "completion_text": sample[text_col][:300] + "..."
                if len(sample[text_col]) > 300
                else sample[text_col],
                "regard_label": sample[label_col],
                "warning": "This text may contain offensive content.",
            }
        )

    return formatted_samples


def run_all_sanity_checks(
    df: pd.DataFrame,
    output_dir: Path,
    cache_key: str,
    n_spot_check_samples: int = 20,
) -> dict:
    sanity_dir = output_dir / "sanity_checks"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "cache_key": cache_key,
        "checks": [],
    }

    dist_check = verify_label_distribution(df)
    results["checks"].append(dist_check.model_dump())

    spot_samples = spot_check_scored_outputs(df, n_samples=n_spot_check_samples)
    results["spot_check_samples"] = spot_samples
    results["spot_check_count"] = len(spot_samples)

    results["statistics"] = {
        "total_samples": len(df),
        "unique_prompts": df["prompt_id"].nunique(),
        "unique_demographics": df["demographic"].nunique(),
        "unique_prompt_types": df["prompt_type"].nunique(),
        "label_distribution": df["regard_label"].value_counts().to_dict(),
    }

    empty_completions = df[df["completion_text"].str.strip() == ""]
    results["empty_completions"] = {
        "count": len(empty_completions),
        "passed": len(empty_completions) == 0,
        "message": "No empty completions found."
        if len(empty_completions) == 0
        else f"Found {len(empty_completions)} empty completions.",
    }

    short_threshold = 5
    short_completions = df[df["completion_text"].str.len() < short_threshold]
    results["short_completions"] = {
        "threshold": short_threshold,
        "count": len(short_completions),
        "passed": len(short_completions) == 0,
        "message": f"No completions shorter than {short_threshold} chars found."
        if len(short_completions) == 0
        else f"Found {len(short_completions)} completions shorter than {short_threshold} chars.",
    }

    results_path = sanity_dir / f"{cache_key}_sanity_checks.json"
    results_path.write_text(json.dumps(results, indent=2))

    spot_path = sanity_dir / f"{cache_key}_spot_check.json"
    spot_path.write_text(json.dumps(spot_samples, indent=2))

    return results
