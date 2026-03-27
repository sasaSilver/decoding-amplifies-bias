# Decoding Amplifies Bias

Repository for two connected implementation tracks:

- bias classifier training, token-level explanations, and explanation validation
- text generation and scoring pipeline locked to the study in [REQUIREMENTS.md](REQUIREMENTS.md)

Both tracks share the same repository and some infrastructure, but ExAI code is isolated under
`src/app/exai/` and ExAI artifacts are written under `outputs/exai/`.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Generation and Scoring

### Prompt bank

The fixed prompt bank lives at `data/prompt_bank_v1.csv`. It contains 48 resolved prompts across 12
templates and 4 demographics.

### Greedy baseline

```bash
PYTHONPATH=src python -m app.cli generate
PYTHONPATH=src python -m app.cli score
```

Artifacts are written under `outputs/`:

- `outputs/generations/<cache_key>.parquet`
- `outputs/manifests/<cache_key>.json`
- `outputs/scores/<cache_key>.parquet`
- `outputs/metrics/<cache_key>_summary.json`

The scoring run uses the released `sasha/regardv3` model. The first scoring run needs network access
unless the model already exists in your local Hugging Face cache or you point the scorer to a local
directory.

### Week 3 grid

```bash
PYTHONPATH=src python -m app.cli generate-grid
PYTHONPATH=src python -m app.cli score-grid
PYTHONPATH=src python -m app.cli week3-metrics
```

The generation grid uses the exact study configs:

- greedy
- temperature `{0.7, 1.0, 1.3}`
- top-k `{20, 50, 100}`
- top-p `{0.8, 0.9, 0.95}`

Optional anti-repetition is controlled with `no_repeat_ngram_size=3`.

## ExAI Baseline

### What is included

The ExAI implementation includes:

- regard dataset ingestion and deterministic splits
- a reproducible explanation benchmark built from saved scored artifacts
- fine-tuning of a 4-class BERT classifier
- a stable inference wrapper for explanation
- LRP for linear layers and a runnable Transformer-level approximation
- token-level explanation rendering
- faithfulness and sensitivity evaluation
- namespaced CLI commands and a reproducible notebook

### Code and artifact boundaries

- ExAI code: `src/app/exai/`
- ExAI runtime artifacts: `outputs/exai/`
- ExAI docs and notebook: `related_projects/ex-ai/`

Key ExAI artifact directories:

- `outputs/exai/metadata/`
- `outputs/exai/models/`
- `outputs/exai/benchmark/`
- `outputs/exai/eval/`
- `outputs/exai/explanations/`
- `outputs/exai/reports/`

### Data

The ExAI classifier expects the labeled TSV corpus under `data/regard/`.

### Download the local BERT checkpoint

```bash
mkdir -p models
huggingface-cli download bert-base-uncased --local-dir models/bert-base-uncased
```

### Train the ExAI classifier

```bash
PYTHONPATH=src python -m app.cli train-exai-classifier \
  --dataset-path data/regard \
  --model-name models/bert-base-uncased \
  --max-length 128 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --epochs 3 \
  --early-stopping \
  --patience 2 \
  --device auto
```

### Build the explanation benchmark

```bash
PYTHONPATH=src python -m app.cli build-exai-benchmark
```

### Evaluate the classifier

```bash
# replace <checkpoint_dir> and <benchmark_path> with the saved artifacts from your run
PYTHONPATH=src python -m app.cli eval-exai-classifier \
  --dataset-path data/regard \
  --checkpoint-path <checkpoint_dir> \
  --benchmark-path <benchmark_path> \
  --batch-size 8 \
  --max-length 128 \
  --device auto \
  --compare-to-released
```

### Generate explanation artifacts

```bash
# replace <checkpoint_dir> and <benchmark_path> with the saved artifacts from your run
PYTHONPATH=src python -m app.cli explain-benchmark \
  --checkpoint-path <checkpoint_dir> \
  --benchmark-path <benchmark_path> \
  --max-examples 5 \
  --max-length 128 \
  --device auto
```

### Run explanation validation

```bash
# replace <checkpoint_dir> and <benchmark_path> with the saved artifacts from your run
PYTHONPATH=src python -m app.cli exai-faithfulness \
  --checkpoint-path <checkpoint_dir> \
  --benchmark-path <benchmark_path> \
  --removal-count 1 \
  --random-seed 13 \
  --max-length 128 \
  --device auto

PYTHONPATH=src python -m app.cli exai-sensitivity \
  --checkpoint-path <checkpoint_dir> \
  --benchmark-path <benchmark_path> \
  --top-k 3 \
  --max-length 128 \
  --device auto
```

### Notebook workflow

The notebook at `related_projects/ex-ai/exai_workflow.ipynb` consumes saved ExAI artifacts. It is
configured to resolve the main repository root and write only to `outputs/exai/`, even when opened
from inside `related_projects/ex-ai/`.

## Development Checks

```bash
ruff format .
ruff check .
pytest
pyright
```

Generated and scored text may contain offensive content. Do not commit or broadly share large raw
outputs; keep excerpts minimal.
