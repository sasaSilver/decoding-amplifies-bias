# Decoding Amplifies Bias

Proposal-locked implementation of the study in [REQUIREMENTS.md](REQUIREMENTS.md).

See week1 report [here](docs/week1/report.pdf)

## Prompt bank

The fixed Week 1 prompt bank lives at `data/prompt_bank_v1.csv`. It contains 48
resolved prompts across 12 templates and 4 demographics.

## Run the greedy baseline

```bash
PYTHONPATH=src python -m decoding_amplifies_bias --n-samples 50 --max-new-tokens 40
```

Artifacts are written under `outputs/`:

- `outputs/generations/<cache_key>.parquet`
- `outputs/manifests/<cache_key>.json`

The manifest records the prompt bank digest, model, greedy decoding config,
seeds, max tokens, sample count, and environment versions. Generated text may
contain offensive content and should not be committed or broadly shared.

## Development checks

```bash
ruff format .
ruff check .
pytest
pyright
```


#  "torch",
#  "transformers",
#  "pandas",
#  "pyarrow",
#  "tqdm",
# "pydantic-settings>=2.13.1",
# "ipykernel>=6.31.0",
# "pydantic>=2.12.5",