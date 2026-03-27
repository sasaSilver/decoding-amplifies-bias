import json
from pathlib import Path

import pandas as pd

from app.exai.benchmark import build_explanation_benchmark
from app.exai.config import ExAIBenchmarkConfig, ExAIPaths


def _score_frame(score_key: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    payload = []
    for index, row in enumerate(rows):
        payload.append(
            {
                "cache_key": score_key,
                "model_name": "gpt2",
                "prompt_id": row["prompt_id"],
                "template_id": f"template_{index}",
                "prompt_type": row["prompt_type"],
                "demographic": row["demographic"],
                "prompt_text": f"Prompt for {row['prompt_id']}",
                "decoding_strategy": row.get("decoding_strategy", "greedy"),
                "do_sample": row.get("do_sample", False),
                "seed": row["seed"],
                "max_new_tokens": 40,
                "sample_index": row["sample_index"],
                "raw_text": f"Prompt {row['completion_text']}",
                "completion_text": row["completion_text"],
                "regard_label": row["regard_label"],
                "scoring_masked": row.get("scoring_masked", True),
            }
        )
    return pd.DataFrame(payload)


def _write_score_artifacts(repo_root: Path) -> tuple[Path, Path]:
    outputs_dir = repo_root / "outputs"
    scores_dir = outputs_dir / "scores"
    manifests_dir = outputs_dir / "manifests"
    generations_dir = outputs_dir / "generations"
    scores_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(parents=True, exist_ok=True)

    score_a = scores_dir / "score_a.parquet"
    score_b = scores_dir / "score_b.parquet"
    rows_a = [
        {
            "prompt_id": "a_neg_1",
            "prompt_type": "occupation",
            "demographic": "Group A",
            "seed": 0,
            "sample_index": 0,
            "completion_text": "Negative example A1",
            "regard_label": "negative",
        },
        {
            "prompt_id": "a_neg_2",
            "prompt_type": "descriptor",
            "demographic": "Group B",
            "seed": 0,
            "sample_index": 1,
            "completion_text": "Negative example B1",
            "regard_label": "negative",
        },
        {
            "prompt_id": "a_neu_1",
            "prompt_type": "occupation",
            "demographic": "Group A",
            "seed": 1,
            "sample_index": 0,
            "completion_text": "Neutral example A1",
            "regard_label": "neutral",
        },
        {
            "prompt_id": "a_pos_1",
            "prompt_type": "descriptor",
            "demographic": "Group B",
            "seed": 1,
            "sample_index": 1,
            "completion_text": "Positive example B1",
            "regard_label": "positive",
        },
    ]
    rows_b = [
        {
            "prompt_id": "b_neu_1",
            "prompt_type": "descriptor",
            "demographic": "Group C",
            "seed": 2,
            "sample_index": 0,
            "completion_text": "Neutral example C1",
            "regard_label": "neutral",
        },
        {
            "prompt_id": "b_pos_1",
            "prompt_type": "occupation",
            "demographic": "Group D",
            "seed": 2,
            "sample_index": 1,
            "completion_text": "Positive example D1",
            "regard_label": "positive",
        },
        {
            "prompt_id": "b_other_1",
            "prompt_type": "occupation",
            "demographic": "Group C",
            "seed": 3,
            "sample_index": 0,
            "completion_text": "Other example C1",
            "regard_label": "other",
        },
        {
            "prompt_id": "b_other_2",
            "prompt_type": "descriptor",
            "demographic": "Group D",
            "seed": 3,
            "sample_index": 1,
            "completion_text": "Other example D1",
            "regard_label": "other",
        },
    ]
    _score_frame("score_a", rows_a).to_parquet(score_a, index=False)
    _score_frame("score_b", rows_b).to_parquet(score_b, index=False)

    for score_path in (score_a, score_b):
        generation_path = generations_dir / f"{score_path.stem}.parquet"
        generation_path.write_text("placeholder", encoding="utf-8")
        (manifests_dir / f"{score_path.stem}.json").write_text(
            json.dumps(
                {
                    "cache_key": score_path.stem,
                    "generations_cache_key": f"gen_{score_path.stem}",
                    "generations_path": str(generation_path.resolve()),
                    "use_masking": True,
                    "artifacts": {"scores_path": str(score_path.resolve())},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    combined_manifest = manifests_dir / "combined_week3_combined.json"
    combined_manifest.write_text(
        json.dumps(
            {
                "cache_key": "combined",
                "created_from_scores": [str(score_a.resolve()), str(score_b.resolve())],
                "use_masking": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return score_a, combined_manifest


def test_build_explanation_benchmark_is_deterministic_and_traceable(tmp_path: Path) -> None:
    _, combined_manifest = _write_score_artifacts(tmp_path)
    paths = ExAIPaths(root=tmp_path / "outputs" / "exai")
    config = ExAIBenchmarkConfig(
        repo_root=tmp_path,
        source_manifest_path=combined_manifest,
        examples_per_label=1,
        selection_seed=5,
        output_paths=paths,
    )

    first = build_explanation_benchmark(config)
    second = build_explanation_benchmark(config)

    assert first.record_count == 4
    assert first.benchmark_path == second.benchmark_path
    benchmark_df = pd.read_parquet(first.benchmark_path)
    assert set(benchmark_df["predicted_label"]) == {"negative", "neutral", "positive", "other"}
    assert all(benchmark_df["source_score_path"].str.endswith(".parquet"))
    assert all(benchmark_df["source_score_manifest_path"].str.endswith(".json"))
    assert all(benchmark_df["scoring_masked"])
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["label_counts"] == {
        "negative": 1,
        "neutral": 1,
        "positive": 1,
        "other": 1,
    }
