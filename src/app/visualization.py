import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import compute_baseline_metrics, compute_negative_regard_gap
from .scoring import RegardLabelEnum


def plot_regard_distribution(
    distributions: dict[str, dict[str, float]],
    output_path: Path,
    title: str = "Regard Distribution by Demographic",
) -> None:
    df = pd.DataFrame.from_dict(distributions, orient="index")
    df = df[[label.value for label in RegardLabelEnum]]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        RegardLabelEnum.NEGATIVE: "#e74c3c",
        RegardLabelEnum.NEUTRAL: "#95a5a6",
        RegardLabelEnum.POSITIVE: "#2ecc71",
        RegardLabelEnum.OTHER: "#f39c12",
    }

    bottom = None
    for label in [label for label in RegardLabelEnum]:
        if label in df.columns:
            values = df[label]
            ax.bar(
                df.index,
                values,
                bottom=bottom,
                label=label,
                color=colors.get(label, "#3498db"),
            )
            if bottom is None:
                bottom = values.copy()
            else:
                bottom += values

    ax.set_xlabel("Demographic")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.legend(title="Regard Label")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_negative_gaps(
    gaps_df: pd.DataFrame,
    output_path: Path,
    title: str = "Negative Regard Gaps by Prompt Type",
) -> None:
    if gaps_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No gap data available", ha="center", va="center")
        ax.set_title(title)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    gaps_df["label"] = (
        gaps_df["prompt_type"] + "\n" + gaps_df["group_a"] + " vs " + gaps_df["group_b"]
    )

    gaps_df["yerr_lower"] = gaps_df["gap_neg"] - gaps_df["ci_lower"]
    gaps_df["yerr_upper"] = gaps_df["ci_upper"] - gaps_df["gap_neg"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = range(len(gaps_df))
    colors = ["#e74c3c" if gap > 0 else "#2ecc71" for gap in gaps_df["gap_neg"]]

    ax.bar(
        x_pos,
        gaps_df["gap_neg"],
        yerr=[gaps_df["yerr_lower"], gaps_df["yerr_upper"]],
        capsize=5,
        color=colors,
        alpha=0.7,
    )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Prompt Type and Group Comparison")
    ax.set_ylabel("Negative Regard Gap (P_neg_A - P_neg_B)")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(gaps_df["label"], rotation=45, ha="right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_baseline_tables(
    scores_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    df = pd.read_parquet(scores_path)
    cache_key = scores_path.stem

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    dist_df = (
        df.groupby("demographic")["regard_label"].value_counts(normalize=True).unstack(fill_value=0)
    )
    dist_df = dist_df[[label.value for label in RegardLabelEnum]]
    dist_df = dist_df.round(4)
    dist_path = tables_dir / f"{cache_key}_regard_distribution_table.csv"
    dist_df.to_csv(dist_path)
    output_paths["regard_distribution"] = dist_path

    gaps_df = compute_negative_regard_gap(df)
    if not gaps_df.empty:
        gaps_df = gaps_df.round(4)
        gaps_path = tables_dir / f"{cache_key}_negative_gaps_table.csv"
        gaps_df.to_csv(gaps_path, index=False)
        output_paths["negative_gaps"] = gaps_path

    summary_data = {
        "Metric": [
            "Total Samples",
            "Unique Prompts",
            "Unique Demographics",
            "Unique Prompt Types",
        ],
        "Value": [
            len(df),
            df["prompt_id"].nunique(),
            df["demographic"].nunique(),
            df["prompt_type"].nunique(),
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = tables_dir / f"{cache_key}_summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    output_paths["summary"] = summary_path

    prompt_type_dist = (
        df.groupby(["prompt_type", "demographic"])["regard_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    prompt_type_dist = prompt_type_dist[[label.value for label in RegardLabelEnum]]
    prompt_type_dist = prompt_type_dist.round(4)
    prompt_type_dist_path = tables_dir / f"{cache_key}_prompt_type_regard_distribution.csv"
    prompt_type_dist.to_csv(prompt_type_dist_path)
    output_paths["prompt_type_regard_distribution"] = prompt_type_dist_path

    return output_paths


def create_baseline_plots(
    scores_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    df = pd.read_parquet(scores_path)
    cache_key = scores_path.stem

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    dist_dict = {}
    for demographic, group_df in df.groupby("demographic"):
        label_counts = group_df["regard_label"].value_counts(normalize=True)
        dist_dict[demographic] = {
            label: label_counts.get(label, 0.0)
            for label in [label.value for label in RegardLabelEnum]
        }

    dist_plot_path = plots_dir / f"{cache_key}_regard_distribution.png"
    plot_regard_distribution(
        dist_dict,
        dist_plot_path,
        title="Regard Distribution by Demographic (Greedy Decoding)",
    )
    output_paths["regard_distribution"] = dist_plot_path

    metric_paths = compute_baseline_metrics(scores_path, output_dir, n_bootstrap=100, ci_level=0.95)
    gaps_ci_path = metric_paths.get("negative_gaps_with_ci")  # type: ignore

    if gaps_ci_path and gaps_ci_path.exists():
        gaps_df = pd.read_csv(gaps_ci_path)
    else:
        from .metrics import compute_negative_regard_gap

        gaps_df = compute_negative_regard_gap(df)

    gaps_plot_path = plots_dir / f"{cache_key}_negative_gaps.png"
    plot_negative_gaps(
        gaps_df,
        gaps_plot_path,
        title="Negative Regard Gaps by Prompt Type (Greedy Decoding)",
    )
    output_paths["negative_gaps"] = gaps_plot_path

    return output_paths


def generate_baseline_report(
    scores_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    cache_key = scores_path.stem

    table_paths = create_baseline_tables(scores_path, output_dir)

    plot_paths = create_baseline_plots(scores_path, output_dir)

    report = {
        "cache_key": cache_key,
        "scores_path": str(scores_path),
        "tables": {k: str(v) for k, v in table_paths.items()},
        "plots": {k: str(v) for k, v in plot_paths.items()},
        "ethics_notice": (
            "Generated text may contain offensive content. "
            "Do not publish or commit large raw dumps."
        ),
    }

    report_path = output_dir / "reports" / f"{cache_key}_baseline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    return report
