from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from .benchmark import build_explanation_benchmark
from .config import ExAIBenchmarkConfig, ExAIDataConfig, ExAIEvalConfig, ExAIPaths
from .evaluate import _checkpoint_reference_digest, evaluate_exai_classifier
from .faithfulness import run_faithfulness_benchmark
from .inference import ExAIInferenceRunner
from .lrp_transformer import TransformerLRPExplainer
from .render import render_benchmark_explanations, render_text_explanation
from .sensitivity import run_sensitivity_benchmark
from .utils import utc_now_iso

SMOKE_LABELS = [
    ("negative", "The doctor was rude."),
    ("neutral", "The teacher entered the room."),
    ("positive", "The nurse helped the patient."),
    ("other", "The artist waited outside."),
]


class NotebookArtifactError(RuntimeError):
    """Raised when the notebook is asked to consume missing real ExAI artifacts."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_smoke_vocab(vocab_path: Path) -> None:
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        ".",
        "!",
        ",",
        "the",
        "doctor",
        "was",
        "rude",
        "teacher",
        "entered",
        "room",
        "nurse",
        "helped",
        "patient",
        "artist",
        "waited",
        "outside",
        "overall",
        "seemed",
    ]
    vocab_path.write_text("\n".join(tokens), encoding="utf-8")


def ensure_local_smoke_checkpoint(paths: ExAIPaths) -> Path:
    checkpoint_dir = paths.models_dir / "classifier_smoke_local"
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        return checkpoint_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = checkpoint_dir / "vocab.txt"
    _build_smoke_vocab(vocab_path)
    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(checkpoint_dir)

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=256,
        num_labels=4,
        id2label={0: "negative", 1: "neutral", 2: "positive", 3: "other"},
        label2id={"negative": 0, "neutral": 1, "positive": 2, "other": 3},
    )
    model = BertForSequenceClassification(config)
    model.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def ensure_smoke_dataset(paths: ExAIPaths) -> Path:
    dataset_dir = paths.metadata_dir / "smoke_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "regard.tsv"
    if dataset_path.exists():
        return dataset_dir

    rows = ["label\ttext\tdemographic"]
    demographics = ["doctor", "teacher", "nurse", "artist"]
    for repeat_index in range(4):
        for (label, text), demographic in zip(SMOKE_LABELS, demographics, strict=False):
            rows.append(f"{label}\t{text[:-1]} {repeat_index}.\t{demographic}")
    dataset_path.write_text("\n".join(rows), encoding="utf-8")
    return dataset_dir


def ensure_smoke_score_sources(paths: ExAIPaths) -> Path:
    source_dir = paths.root / "benchmark_source"
    scores_dir = source_dir / "scores"
    manifests_dir = source_dir / "manifests"
    generations_dir = source_dir / "generations"
    scores_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(parents=True, exist_ok=True)

    score_path = scores_dir / "smoke_scores.parquet"
    manifest_path = manifests_dir / "smoke_scores.json"
    combined_manifest_path = manifests_dir / "smoke_scores_week3_combined.json"
    if not score_path.exists():
        rows: list[dict[str, Any]] = []
        prompt_types = ["occupation", "descriptor", "occupation", "descriptor"]
        demographics = ["Group A", "Group B", "Group C", "Group D"]
        for index, ((label, text), prompt_type, demographic) in enumerate(
            zip(SMOKE_LABELS, prompt_types, demographics, strict=False)
        ):
            rows.append(
                {
                    "cache_key": "smoke_scores",
                    "model_name": "gpt2",
                    "prompt_id": f"smoke_{index}",
                    "template_id": f"template_{index}",
                    "prompt_type": prompt_type,
                    "demographic": demographic,
                    "prompt_text": f"Prompt {index}",
                    "decoding_strategy": "greedy",
                    "do_sample": False,
                    "seed": index,
                    "max_new_tokens": 40,
                    "sample_index": index,
                    "raw_text": text,
                    "completion_text": text,
                    "regard_label": label,
                    "scoring_masked": True,
                }
            )
        pd.DataFrame(rows).to_parquet(score_path, index=False)

    generation_path = generations_dir / "smoke_generations.parquet"
    if not generation_path.exists():
        generation_path.write_text("placeholder", encoding="utf-8")

    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(
                {
                    "cache_key": "smoke_scores",
                    "created_at_utc": utc_now_iso(),
                    "generations_cache_key": "smoke_generations",
                    "generations_path": str(generation_path.resolve()),
                    "artifacts": {"scores_path": str(score_path.resolve())},
                    "use_masking": True,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if not combined_manifest_path.exists():
        combined_manifest_path.write_text(
            json.dumps(
                {
                    "cache_key": "smoke_combined",
                    "created_at_utc": utc_now_iso(),
                    "created_from_scores": [str(score_path.resolve())],
                    "use_masking": True,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return combined_manifest_path


def _ensure_smoke_notebook_artifacts(resolved_paths: ExAIPaths) -> dict[str, Path]:
    dataset_dir = ensure_smoke_dataset(resolved_paths)

    existing_checkpoints = sorted(resolved_paths.models_dir.glob("classifier_*"))
    existing_benchmarks = sorted(resolved_paths.benchmark_dir.glob("benchmark_*.parquet"))
    if existing_checkpoints and existing_benchmarks:
        checkpoint_dir = max(existing_checkpoints, key=lambda path: path.stat().st_mtime)
        benchmark_path = max(existing_benchmarks, key=lambda path: path.stat().st_mtime)
    else:
        checkpoint_dir = ensure_local_smoke_checkpoint(resolved_paths)
        combined_manifest_path = ensure_smoke_score_sources(resolved_paths)
        benchmark_result = build_explanation_benchmark(
            ExAIBenchmarkConfig(
                repo_root=resolved_paths.repo_root,
                source_manifest_path=combined_manifest_path,
                examples_per_label=1,
                selection_seed=13,
                output_paths=resolved_paths,
            )
        )
        benchmark_path = benchmark_result.benchmark_path

    if not any(resolved_paths.eval_dir.glob("*.json")):
        evaluate_exai_classifier(
            data_config=ExAIDataConfig(
                dataset_path=dataset_dir,
                split_seed=13,
                train_fraction=0.5,
                validation_fraction=0.25,
                use_masking=True,
                output_paths=resolved_paths,
            ),
            eval_config=ExAIEvalConfig(
                checkpoint_path=checkpoint_dir,
                benchmark_path=benchmark_path,
                batch_size=4,
                max_length=64,
                device="cpu",
                output_paths=resolved_paths,
            ),
        )

    render_text_explanation(
        checkpoint_path=checkpoint_dir,
        text="The doctor was rude.",
        output_dir=resolved_paths.explanations_dir,
        target_label="negative",
        device="cpu",
        max_length=64,
    )
    render_benchmark_explanations(
        checkpoint_path=checkpoint_dir,
        benchmark_path=benchmark_path,
        output_dir=resolved_paths.explanations_dir,
        max_examples=5,
        device="cpu",
        max_length=64,
    )
    runner = ExAIInferenceRunner(checkpoint_dir, device="cpu", max_length=64)
    explainer = TransformerLRPExplainer(runner)
    faithfulness_artifacts = run_faithfulness_benchmark(
        runner=runner,
        explainer=explainer,
        benchmark_path=benchmark_path,
        output_dir=resolved_paths.reports_dir / "faithfulness",
        removal_count=1,
        random_seed=13,
    )
    sensitivity_artifacts = run_sensitivity_benchmark(
        runner=runner,
        explainer=explainer,
        benchmark_path=benchmark_path,
        output_dir=resolved_paths.reports_dir / "sensitivity",
        top_k=2,
    )
    return {
        "checkpoint_dir": checkpoint_dir,
        "benchmark_path": benchmark_path,
        "eval_dir": resolved_paths.eval_dir,
        "faithfulness_metrics": faithfulness_artifacts["metrics"],
        "sensitivity_metrics": sensitivity_artifacts["metrics"],
    }


def _latest_real_checkpoint(paths: ExAIPaths) -> Path:
    candidates = [
        path
        for path in sorted(paths.models_dir.glob("classifier_*"))
        if path.name != "classifier_smoke_local"
        and (path / "training_manifest.json").exists()
        and (path / "training_metrics.json").exists()
    ]
    if not candidates:
        raise NotebookArtifactError(
            "No real ExAI checkpoint exists under "
            f"{paths.models_dir}. Run `train-exai-classifier --dataset-path data/regard` "
            "with a local `bert-base-uncased` checkpoint first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _is_smoke_score_path(paths: ExAIPaths, score_path: Path) -> bool:
    benchmark_source_root = (paths.root / "benchmark_source").resolve()
    try:
        return score_path.resolve().is_relative_to(benchmark_source_root)
    except ValueError:
        return False


def _latest_real_benchmark(paths: ExAIPaths) -> Path:
    candidates: list[Path] = []
    for manifest_path in sorted(paths.benchmark_dir.glob("benchmark_*.json")):
        payload = _read_json(manifest_path)
        benchmark_path = Path(payload["artifacts"]["benchmark_path"]).expanduser().resolve()
        source_scores = [
            Path(item["path"]).expanduser().resolve() for item in payload.get("source_scores", [])
        ]
        if not benchmark_path.exists():
            continue
        if source_scores and all(
            _is_smoke_score_path(paths, score_path) for score_path in source_scores
        ):
            continue
        candidates.append(benchmark_path)

    if not candidates:
        raise NotebookArtifactError(
            "No real ExAI benchmark exists under "
            f"{paths.benchmark_dir}. Run `build-exai-benchmark` against the saved week 3 score "
            "manifests before opening the notebook."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _required_eval_artifacts(paths: ExAIPaths, checkpoint_dir: Path) -> dict[str, Path]:
    eval_key = _checkpoint_reference_digest(checkpoint_dir)[:20]
    artifacts = {
        "test_metrics": paths.eval_dir / f"eval_{eval_key}_test_metrics.json",
        "benchmark_metrics": paths.eval_dir / f"eval_{eval_key}_benchmark_metrics.json",
        "agreement_metrics": paths.eval_dir / f"eval_{eval_key}_agreement.json",
        "error_analysis": paths.eval_dir / f"eval_{eval_key}_error_analysis.json",
    }
    missing = [path for path in artifacts.values() if not path.exists()]
    if missing:
        raise NotebookArtifactError(
            "Missing real ExAI evaluation artifacts for checkpoint "
            f"{checkpoint_dir}. Run `eval-exai-classifier` with the saved real checkpoint and "
            "benchmark before opening the notebook."
        )
    return artifacts


def _require_report_metrics(
    *,
    report_path: Path,
    checkpoint_dir: Path,
    benchmark_path: Path,
    command_name: str,
) -> Path:
    if not report_path.exists():
        raise NotebookArtifactError(
            f"Missing {report_path.name}. Run `{command_name}` with the real checkpoint and "
            "benchmark before opening the notebook."
        )

    payload = _read_json(report_path)
    if payload.get("checkpoint_path") != str(checkpoint_dir.resolve()):
        raise NotebookArtifactError(
            f"{report_path.name} does not reference the real checkpoint at {checkpoint_dir}. "
            f"Re-run `{command_name}` after the real training run completes."
        )
    if payload.get("benchmark_path") != str(benchmark_path.resolve()):
        raise NotebookArtifactError(
            f"{report_path.name} does not reference the real benchmark at {benchmark_path}. "
            f"Re-run `{command_name}` after rebuilding the real benchmark."
        )
    return report_path


def ensure_notebook_artifacts(
    paths: ExAIPaths | None = None,
    *,
    allow_smoke: bool = False,
) -> dict[str, Path]:
    resolved_paths = (paths or ExAIPaths()).ensure_dirs()
    if allow_smoke:
        return _ensure_smoke_notebook_artifacts(resolved_paths)

    checkpoint_dir = _latest_real_checkpoint(resolved_paths)
    benchmark_path = _latest_real_benchmark(resolved_paths)
    eval_artifacts = _required_eval_artifacts(resolved_paths, checkpoint_dir)
    faithfulness_metrics = _require_report_metrics(
        report_path=resolved_paths.reports_dir / "faithfulness" / "faithfulness_metrics.json",
        checkpoint_dir=checkpoint_dir,
        benchmark_path=benchmark_path,
        command_name="exai-faithfulness",
    )
    sensitivity_metrics = _require_report_metrics(
        report_path=resolved_paths.reports_dir / "sensitivity" / "sensitivity_metrics.json",
        checkpoint_dir=checkpoint_dir,
        benchmark_path=benchmark_path,
        command_name="exai-sensitivity",
    )
    return {
        "checkpoint_dir": checkpoint_dir,
        "benchmark_path": benchmark_path,
        "eval_dir": resolved_paths.eval_dir,
        "test_metrics": eval_artifacts["test_metrics"],
        "benchmark_metrics": eval_artifacts["benchmark_metrics"],
        "agreement_metrics": eval_artifacts["agreement_metrics"],
        "error_analysis": eval_artifacts["error_analysis"],
        "faithfulness_metrics": faithfulness_metrics,
        "sensitivity_metrics": sensitivity_metrics,
    }
