import json
from hashlib import sha256
from pathlib import Path

import click

from .settings.settings import Settings


def _normalize_decoding_payload(decoding: dict[str, object]) -> dict[str, object]:
    payload = {
        "strategy": decoding["strategy"],
        "do_sample": decoding["do_sample"],
    }

    for field in ("temperature", "top_k", "top_p"):
        value = decoding.get(field)
        if value is not None:
            payload[field] = value

    no_repeat_ngram_size = decoding.get("no_repeat_ngram_size")
    if isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size > 0:
        payload["no_repeat_ngram_size"] = no_repeat_ngram_size

    return payload


def _generation_signature_from_manifest(manifest: dict[str, object]) -> str:
    cache_payload = dict(manifest.get("cache_payload", {}))
    decoding = cache_payload.get("decoding", manifest.get("decoding", {}))
    if not isinstance(decoding, dict):
        raise ValueError("Generation manifest decoding payload must be a dictionary.")

    payload = {
        "model_name": cache_payload.get("model_name", manifest.get("model_name")),
        "prompt_bank_digest": cache_payload.get(
            "prompt_bank_digest", manifest.get("prompt_bank_digest")
        ),
        "max_new_tokens": cache_payload.get("max_new_tokens", manifest.get("max_new_tokens")),
        "n_samples_per_prompt": cache_payload.get(
            "n_samples_per_prompt",
            manifest.get("n_samples_per_prompt"),
        ),
        "seeds": cache_payload.get("seeds", manifest.get("seeds")),
        "decoding": _normalize_decoding_payload(decoding),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _collect_unique_generation_files(settings: Settings) -> list[Path]:
    manifests_dir = settings.output_dir / "manifests"
    unique_files: list[Path] = []
    seen_signatures: set[str] = set()

    for generations_path in sorted(settings.generations_path.glob("*.parquet")):
        manifest_path = manifests_dir / f"{generations_path.stem}.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = _generation_signature_from_manifest(manifest)
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        unique_files.append(generations_path)

    return unique_files


def score_cmd(settings: Settings) -> None:
    import pandas as pd

    from .metrics import compute_baseline_metrics
    from .sanity import run_all_sanity_checks
    from .scoring import ScoringModelLoadError, ScoringRunner
    from .visualization import generate_baseline_report

    gen_files = list(settings.generations_path.glob("*.parquet"))
    if not gen_files:
        raise ValueError(f"No generations found in {settings.generations_path}")
    generations_path = max(gen_files, key=lambda p: p.stat().st_mtime)
    runner = ScoringRunner()
    try:
        result = runner.run(
            config=settings.scoring,
            generations_path=generations_path,
        )
    except ScoringModelLoadError as exc:
        raise click.ClickException(str(exc)) from exc
    print(
        f"""
Scoring complete:
Scores: {result.scores_path}
Manifest: {result.manifest_path}
"""
    )
    metric_paths = compute_baseline_metrics(
        scores_path=result.scores_path,
        output_dir=settings.output_dir,
        n_bootstrap=settings.n_bootstrap,
        ci_level=settings.ci_level,
    )
    print("\nMetrics computed:")
    for metric_type, path in metric_paths.items():
        print(f"  {metric_type}: {path}")
    df = pd.read_parquet(result.scores_path)
    cache_key = result.scores_path.stem
    sanity_results = run_all_sanity_checks(
        df=df,
        output_dir=settings.output_dir,
        cache_key=cache_key,
        n_spot_check_samples=settings.n_spot_check,
    )
    print("\nSanity checks:")
    for check in sanity_results["checks"]:
        status = "✓" if check["passed"] else "✗"
        print(f"  {status} {check['check_name']}: {check['message']}")
    print(f"\nSpot-check samples: {sanity_results['spot_check_count']}")
    print(f"  See: {settings.output_dir / 'sanity_checks' / f'{cache_key}_spot_check.json'}")
    report = generate_baseline_report(result.scores_path, settings.output_dir)
    print("\nBaseline report generated:")
    print(f"  Report: {settings.output_dir / 'reports' / f'{cache_key}_baseline_report.json'}")
    print("\nTables:")
    for table_type, path in report["tables"].items():
        print(f"  {table_type}: {path}")
    print("\nPlots:")
    for plot_type, path in report["plots"].items():
        print(f"  {plot_type}: {path}")


def generation_cmd(settings: Settings) -> None:
    from .generation import GenerationRunner

    result = GenerationRunner().run(settings.generation)
    print(
        f"""
Generation results:
{result.model_dump_json()}
"""
    )


def generation_grid_cmd(settings: Settings) -> None:
    from .generation import GenerationRunner
    from .settings.generation import build_week3_decoding_grid

    runner = GenerationRunner()
    results = []

    for decoding in build_week3_decoding_grid(
        include_greedy=True,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
    ):
        click.echo(f"Running generation for decoding={decoding.to_dict()}")
        config = settings.generation.model_copy(update={"decoding": decoding})
        result = runner.run(config)
        results.append((decoding.to_dict(), result))
        click.echo(
            f"Completed decoding={decoding.to_dict()} "
            f"(cache_key={result.cache_key}, from_cache={result.from_cache})"
        )

    print("\nWeek 3 generation grid:")
    for decoding, result in results:
        print(
            f"  {decoding} -> {result.generations_path} "
            f"(cache_key={result.cache_key}, from_cache={result.from_cache})"
        )


def score_grid_cmd(settings: Settings) -> None:
    from .scoring import ScoringModelLoadError, ScoringRunner

    generation_files = _collect_unique_generation_files(settings)
    if not generation_files:
        raise ValueError(f"No generations found in {settings.generations_path}")

    runner = ScoringRunner()
    results = []

    try:
        for generations_path in generation_files:
            result = runner.run(
                config=settings.scoring,
                generations_path=generations_path,
            )
            results.append((generations_path, result))
    except ScoringModelLoadError as exc:
        raise click.ClickException(str(exc)) from exc

    print("\nWeek 3 scoring grid:")
    for generations_path, result in results:
        print(
            f"  {generations_path.name} -> {result.scores_path} "
            f"(cache_key={result.cache_key}, from_cache={result.from_cache})"
        )


def _collect_matching_score_files(settings: Settings) -> list[Path]:
    scores_dir = settings.output_dir / "scores"
    manifests_dir = settings.output_dir / "manifests"
    model_reference = settings.scoring.resolved_model_reference()
    matching_files: list[Path] = []
    seen_generation_signatures: set[str] = set()

    for scores_path in sorted(scores_dir.glob("*.parquet")):
        manifest_path = manifests_dir / f"{scores_path.stem}.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "generations_cache_key" not in manifest:
            continue
        if manifest.get("model_reference", manifest.get("model_name")) != model_reference:
            continue
        if manifest.get("use_masking") != settings.scoring.use_masking:
            continue

        generation_manifest_path = manifests_dir / f"{manifest['generations_cache_key']}.json"
        if generation_manifest_path.exists():
            generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
            generation_signature = _generation_signature_from_manifest(generation_manifest)
            if generation_signature in seen_generation_signatures:
                continue
            seen_generation_signatures.add(generation_signature)

        matching_files.append(scores_path)

    return matching_files


def _build_week3_combined_scores(settings: Settings, score_files: list[Path]) -> Path:
    import pandas as pd

    payload = {
        "score_files": [path.stem for path in score_files],
        "model_reference": settings.scoring.resolved_model_reference(),
        "use_masking": settings.scoring.use_masking,
    }
    combined_key = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]

    combined_path = settings.output_dir / "scores" / f"{combined_key}_week3_combined.parquet"
    manifest_path = settings.output_dir / "manifests" / f"{combined_key}_week3_combined.json"

    if combined_path.exists() and manifest_path.exists():
        return combined_path

    frames = [pd.read_parquet(scores_path) for scores_path in score_files]
    combined_df = pd.concat(frames, ignore_index=True)

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(combined_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "cache_key": combined_key,
                "created_from_scores": [str(path.resolve()) for path in score_files],
                "model_reference": settings.scoring.resolved_model_reference(),
                "use_masking": settings.scoring.use_masking,
                "record_count": len(combined_df),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return combined_path


def week3_metrics_cmd(settings: Settings) -> None:
    from .metrics import compute_week3_metrics

    score_files = _collect_matching_score_files(settings)
    if not score_files:
        raise ValueError(
            "No scored files matched the current scoring configuration. "
            "Run `score-grid` first or adjust scoring_model/scoring_model_path."
        )

    combined_scores_path = _build_week3_combined_scores(settings, score_files)
    print("Computing Week 3 regard distributions and gap CIs...")
    metric_paths = compute_week3_metrics(
        scores_path=combined_scores_path,
        output_dir=settings.output_dir,
        n_bootstrap=settings.n_bootstrap,
        quality_n_bootstrap=settings.quality_n_bootstrap,
        ci_level=settings.ci_level,
    )

    print("\nWeek 3 metrics:")
    print(f"  Combined scores: {combined_scores_path}")
    for metric_type, path in metric_paths.items():
        print(f"  {metric_type}: {path}")


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
def score() -> None:
    score_cmd(Settings())


@cli.command()
def generate() -> None:
    generation_cmd(Settings())


@cli.command(name="generate-grid")
def generate_grid() -> None:
    generation_grid_cmd(Settings())


@cli.command(name="score-grid")
def score_grid() -> None:
    score_grid_cmd(Settings())


@cli.command(name="week3-metrics")
def week3_metrics() -> None:
    week3_metrics_cmd(Settings())


@cli.command(name="build-exai-benchmark")
@click.option("--source-manifest-path", type=click.Path(path_type=Path), default=None)
@click.option("--examples-per-label", type=int, default=3)
@click.option("--selection-seed", type=int, default=13)
def build_exai_benchmark(
    source_manifest_path: Path | None, examples_per_label: int, selection_seed: int
) -> None:
    from .exai.cli import build_exai_benchmark_cmd

    artifacts = build_exai_benchmark_cmd(
        source_manifest_path=source_manifest_path,
        examples_per_label=examples_per_label,
        selection_seed=selection_seed,
    )
    click.echo(f"Benchmark: {artifacts['benchmark']}")
    click.echo(f"Manifest: {artifacts['manifest']}")


@cli.command(name="train-exai-classifier")
@click.option("--dataset-path", type=click.Path(path_type=Path), required=True)
@click.option("--model-name", type=str, default="bert-base-uncased")
@click.option("--max-length", type=int, default=128)
@click.option("--batch-size", type=int, default=8)
@click.option("--learning-rate", type=float, default=2e-5)
@click.option("--epochs", type=int, default=3)
@click.option("--early-stopping/--no-early-stopping", default=True)
@click.option("--patience", type=int, default=2)
@click.option("--device", type=str, default="auto")
def train_exai_classifier_cli(
    dataset_path: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    early_stopping: bool,
    patience: int,
    device: str,
) -> None:
    from .exai.cli import train_exai_classifier_cmd

    try:
        artifacts = train_exai_classifier_cmd(
            dataset_path=dataset_path,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping=early_stopping,
            early_stopping_patience=patience,
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Checkpoint: {artifacts['checkpoint_dir']}")
    click.echo(f"Manifest: {artifacts['manifest']}")
    click.echo(f"Metrics: {artifacts['metrics']}")


@cli.command(name="eval-exai-classifier")
@click.option("--dataset-path", type=click.Path(path_type=Path), required=True)
@click.option("--checkpoint-path", type=click.Path(path_type=Path), required=True)
@click.option("--benchmark-path", type=click.Path(path_type=Path), default=None)
@click.option("--batch-size", type=int, default=8)
@click.option("--max-length", type=int, default=128)
@click.option("--device", type=str, default="auto")
@click.option("--compare-to-released/--no-compare-to-released", default=True)
@click.option("--released-model-path", type=click.Path(path_type=Path), default=None)
def eval_exai_classifier_cli(
    dataset_path: Path,
    checkpoint_path: Path,
    benchmark_path: Path | None,
    batch_size: int,
    max_length: int,
    device: str,
    compare_to_released: bool,
    released_model_path: Path | None,
) -> None:
    from .exai.cli import eval_exai_classifier_cmd

    try:
        artifacts = eval_exai_classifier_cmd(
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            benchmark_path=benchmark_path,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            compare_to_released=compare_to_released,
            released_model_path=released_model_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    for name, path in artifacts.items():
        click.echo(f"{name}: {path}")


@cli.command(name="explain-text")
@click.option("--checkpoint-path", type=click.Path(path_type=Path), required=True)
@click.option("--text", type=str, required=True)
@click.option("--target-label", type=str, default=None)
@click.option("--max-length", type=int, default=128)
@click.option("--device", type=str, default="auto")
def explain_text_cli(
    checkpoint_path: Path,
    text: str,
    target_label: str | None,
    max_length: int,
    device: str,
) -> None:
    from .exai.cli import explain_text_cmd

    artifacts = explain_text_cmd(
        checkpoint_path=checkpoint_path,
        text=text,
        target_label=target_label,
        max_length=max_length,
        device=device,
    )
    click.echo(f"JSON: {artifacts['json']}")
    click.echo(f"HTML: {artifacts['html']}")


@cli.command(name="explain-benchmark")
@click.option("--checkpoint-path", type=click.Path(path_type=Path), required=True)
@click.option("--benchmark-path", type=click.Path(path_type=Path), required=True)
@click.option("--max-examples", type=int, default=5)
@click.option("--max-length", type=int, default=128)
@click.option("--device", type=str, default="auto")
def explain_benchmark_cli(
    checkpoint_path: Path,
    benchmark_path: Path,
    max_examples: int,
    max_length: int,
    device: str,
) -> None:
    from .exai.cli import explain_benchmark_cmd

    paths = explain_benchmark_cmd(
        checkpoint_path=checkpoint_path,
        benchmark_path=benchmark_path,
        max_examples=max_examples,
        max_length=max_length,
        device=device,
    )
    for path in paths:
        click.echo(str(path))


@cli.command(name="exai-faithfulness")
@click.option("--checkpoint-path", type=click.Path(path_type=Path), required=True)
@click.option("--benchmark-path", type=click.Path(path_type=Path), required=True)
@click.option("--removal-count", type=int, default=1)
@click.option("--random-seed", type=int, default=13)
@click.option("--max-length", type=int, default=128)
@click.option("--device", type=str, default="auto")
def exai_faithfulness_cli(
    checkpoint_path: Path,
    benchmark_path: Path,
    removal_count: int,
    random_seed: int,
    max_length: int,
    device: str,
) -> None:
    from .exai.cli import exai_faithfulness_cmd

    artifacts = exai_faithfulness_cmd(
        checkpoint_path=checkpoint_path,
        benchmark_path=benchmark_path,
        removal_count=removal_count,
        random_seed=random_seed,
        max_length=max_length,
        device=device,
    )
    for name, path in artifacts.items():
        click.echo(f"{name}: {path}")


@cli.command(name="exai-sensitivity")
@click.option("--checkpoint-path", type=click.Path(path_type=Path), required=True)
@click.option("--benchmark-path", type=click.Path(path_type=Path), required=True)
@click.option("--top-k", type=int, default=3)
@click.option("--max-length", type=int, default=128)
@click.option("--device", type=str, default="auto")
def exai_sensitivity_cli(
    checkpoint_path: Path,
    benchmark_path: Path,
    top_k: int,
    max_length: int,
    device: str,
) -> None:
    from .exai.cli import exai_sensitivity_cmd

    artifacts = exai_sensitivity_cmd(
        checkpoint_path=checkpoint_path,
        benchmark_path=benchmark_path,
        top_k=top_k,
        max_length=max_length,
        device=device,
    )
    for name, path in artifacts.items():
        click.echo(f"{name}: {path}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
