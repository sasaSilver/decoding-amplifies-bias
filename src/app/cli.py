import click
import pandas as pd

from .generation import GenerationRunner
from .metrics import compute_baseline_metrics
from .sanity import run_all_sanity_checks
from .scoring import ScoringRunner
from .settings.settings import Settings
from .visualization import generate_baseline_report


def score_cmd(settings: Settings) -> None:
    gen_files = list(settings.generations_path.glob("*.parquet"))
    if not gen_files:
        raise ValueError(f"No generations found in {settings.generations_path}")
    generations_path = max(gen_files, key=lambda p: p.stat().st_mtime)
    runner = ScoringRunner()
    result = runner.run(
        config=settings.scoring,
        generations_path=generations_path,
    )
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
    result = GenerationRunner().run(settings.generation)
    print(
f"""
Generation results:
{result.model_dump_json()}
"""
    )

@click.group()
def cli():
    ...

@cli.command()
def score():
    score_cmd(settings)

@cli.command()
def generate():
    generation_cmd(settings)
    


if __name__ == "__main__":
    settings = Settings()
    print(
f"""
App settings:
{settings.model_dump()}
"""
    )
    cli()
