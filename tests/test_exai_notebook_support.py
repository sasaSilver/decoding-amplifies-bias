import json
from pathlib import Path

from app.exai.config import ExAIPaths
from app.exai.evaluate import _checkpoint_reference_digest
from app.exai.notebook_support import NotebookArtifactError, ensure_notebook_artifacts


def test_ensure_notebook_artifacts_requires_real_artifacts_by_default(tmp_path: Path) -> None:
    output_paths = ExAIPaths(root=tmp_path / "outputs" / "exai").ensure_dirs()

    try:
        ensure_notebook_artifacts(output_paths)
    except NotebookArtifactError as exc:
        assert "No real ExAI checkpoint exists" in str(exc)
    else:
        raise AssertionError("Expected NotebookArtifactError when only smoke mode is available.")


def test_ensure_notebook_artifacts_returns_real_artifacts(tmp_path: Path) -> None:
    output_paths = ExAIPaths(root=tmp_path / "outputs" / "exai").ensure_dirs()

    checkpoint_dir = output_paths.models_dir / "classifier_real"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "training_metrics.json").write_text(
        json.dumps({"best_epoch": 1}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (checkpoint_dir / "training_manifest.json").write_text(
        json.dumps({"run_key": "real"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checkpoint_key = _checkpoint_reference_digest(checkpoint_dir)[:20]

    source_score_path = tmp_path / "outputs" / "scores" / "real_scores.parquet"
    source_score_path.parent.mkdir(parents=True, exist_ok=True)
    source_score_path.write_text("placeholder", encoding="utf-8")

    benchmark_path = output_paths.benchmark_dir / "benchmark_real.parquet"
    benchmark_path.write_text("placeholder", encoding="utf-8")
    benchmark_manifest_path = output_paths.benchmark_dir / "benchmark_real.json"
    benchmark_manifest_path.write_text(
        json.dumps(
            {
                "artifacts": {"benchmark_path": str(benchmark_path.resolve())},
                "source_scores": [{"path": str(source_score_path.resolve()), "sha256": "abc"}],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for suffix in (
        "test_metrics",
        "benchmark_metrics",
        "agreement",
        "error_analysis",
    ):
        (output_paths.eval_dir / f"eval_{checkpoint_key}_{suffix}.json").write_text(
            json.dumps({"ok": True}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    for report_path in (
        output_paths.reports_dir / "faithfulness" / "faithfulness_metrics.json",
        output_paths.reports_dir / "sensitivity" / "sensitivity_metrics.json",
    ):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "checkpoint_path": str(checkpoint_dir.resolve()),
                    "benchmark_path": str(benchmark_path.resolve()),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    artifacts = ensure_notebook_artifacts(output_paths)

    assert artifacts["checkpoint_dir"] == checkpoint_dir
    assert artifacts["benchmark_path"] == benchmark_path
    assert artifacts["faithfulness_metrics"].name == "faithfulness_metrics.json"
    assert artifacts["sensitivity_metrics"].name == "sensitivity_metrics.json"
