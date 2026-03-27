from pathlib import Path

from click.testing import CliRunner

from app.cli import cli


def test_exai_cli_commands_are_registered(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_train_exai_classifier_cmd(**kwargs: object) -> dict[str, Path]:
        calls["train"] = kwargs
        return {
            "checkpoint_dir": Path("/tmp/checkpoint"),
            "manifest": Path("/tmp/manifest.json"),
            "metrics": Path("/tmp/metrics.json"),
        }

    def fake_eval_exai_classifier_cmd(**kwargs: object) -> dict[str, Path]:
        calls["eval"] = kwargs
        return {
            "test_metrics": Path("/tmp/test.json"),
            "benchmark_metrics": Path("/tmp/bench.json"),
            "agreement_metrics": Path("/tmp/agreement.json"),
            "error_analysis": Path("/tmp/error.json"),
        }

    def fake_build_exai_benchmark_cmd(**kwargs: object) -> dict[str, Path]:
        calls["benchmark"] = kwargs
        return {
            "benchmark": Path("/tmp/benchmark.parquet"),
            "manifest": Path("/tmp/benchmark.json"),
        }

    def fake_explain_text_cmd(**kwargs: object) -> dict[str, Path]:
        calls["explain_text"] = kwargs
        return {
            "json": Path("/tmp/explanation.json"),
            "html": Path("/tmp/explanation.html"),
        }

    def fake_explain_benchmark_cmd(**kwargs: object) -> list[Path]:
        calls["explain_benchmark"] = kwargs
        return [Path("/tmp/example.html")]

    def fake_exai_faithfulness_cmd(**kwargs: object) -> dict[str, Path]:
        calls["faithfulness"] = kwargs
        return {
            "metrics": Path("/tmp/faithfulness.json"),
            "plot": Path("/tmp/faithfulness.png"),
        }

    def fake_exai_sensitivity_cmd(**kwargs: object) -> dict[str, Path]:
        calls["sensitivity"] = kwargs
        return {
            "metrics": Path("/tmp/sensitivity.json"),
            "plot": Path("/tmp/sensitivity.png"),
        }

    monkeypatch.setattr("app.exai.cli.train_exai_classifier_cmd", fake_train_exai_classifier_cmd)
    monkeypatch.setattr("app.exai.cli.eval_exai_classifier_cmd", fake_eval_exai_classifier_cmd)
    monkeypatch.setattr("app.exai.cli.build_exai_benchmark_cmd", fake_build_exai_benchmark_cmd)
    monkeypatch.setattr("app.exai.cli.explain_text_cmd", fake_explain_text_cmd)
    monkeypatch.setattr("app.exai.cli.explain_benchmark_cmd", fake_explain_benchmark_cmd)
    monkeypatch.setattr("app.exai.cli.exai_faithfulness_cmd", fake_exai_faithfulness_cmd)
    monkeypatch.setattr("app.exai.cli.exai_sensitivity_cmd", fake_exai_sensitivity_cmd)

    runner = CliRunner()
    assert runner.invoke(cli, ["build-exai-benchmark"]).exit_code == 0
    assert (
        runner.invoke(
            cli,
            ["train-exai-classifier", "--dataset-path", "/tmp/regard"],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            cli,
            [
                "eval-exai-classifier",
                "--dataset-path",
                "/tmp/regard",
                "--checkpoint-path",
                "/tmp/checkpoint",
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            cli,
            ["explain-text", "--checkpoint-path", "/tmp/checkpoint", "--text", "Example"],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            cli,
            [
                "explain-benchmark",
                "--checkpoint-path",
                "/tmp/checkpoint",
                "--benchmark-path",
                "/tmp/benchmark.parquet",
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            cli,
            [
                "exai-faithfulness",
                "--checkpoint-path",
                "/tmp/checkpoint",
                "--benchmark-path",
                "/tmp/benchmark.parquet",
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            cli,
            [
                "exai-sensitivity",
                "--checkpoint-path",
                "/tmp/checkpoint",
                "--benchmark-path",
                "/tmp/benchmark.parquet",
            ],
        ).exit_code
        == 0
    )

    assert {
        "benchmark",
        "train",
        "eval",
        "explain_text",
        "explain_benchmark",
        "faithfulness",
        "sensitivity",
    } <= calls.keys()
