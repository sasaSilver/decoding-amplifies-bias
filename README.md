# Decoding Amplifies Bias

Proposal-locked implementation of the study in [REQUIREMENTS.md](REQUIREMENTS.md).

See week1 report [here](docs/week1/report.pdf)

## Prompt bank

The fixed Week 1 prompt bank lives at `data/prompt_bank_v1.csv`. It contains 48
resolved prompts across 12 templates and 4 demographics.

## Run the greedy baseline

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m app.cli generate
```

Artifacts are written under `outputs/`:

- `outputs/generations/<cache_key>.parquet`
- `outputs/manifests/<cache_key>.json`

The manifest records the prompt bank digest, model, greedy decoding config,
seeds, max tokens, sample count, and environment versions. Generated text may
contain offensive content and should not be committed or broadly shared.

## Run scoring

```bash
PYTHONPATH=src python -m app.cli score
```

The first scoring run needs network access to download `sasha/regardv3`, unless
that model already exists in your local Hugging Face cache or you set
`scoring_model` to a local model directory.

## Development checks

```bash
ruff format .
ruff check .
pytest
pyright
```
