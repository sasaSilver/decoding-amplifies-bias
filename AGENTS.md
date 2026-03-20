# AGENTS.md

## Agent workflow for implementing this repo (proposal-locked)

This repo must implement the project exactly as described in `REQUIREMENTS.md` (which mirrors the proposal).
Agents must not invent additional milestones or scope.

---

## Mandatory dev tooling
### VS Code + Pylance (required)
- Use Pylance for static type checking.
- Fix type issues; don’t “paper over” them with broad ignores.

### Linting / formatting (required)
- Use a linter and formatter; preferred: Ruff.
  - `ruff format .`
  - `ruff check .`
- Code must be lint-clean and formatted before considering a task complete.

### Tests (required)
- Use `pytest` for smoke tests:
  - prompt bank validation
  - generation caching determinism (cache key stability)
  - scoring pipeline smoke test on a tiny subset

---

## Milestone discipline (W1–W5 only)
Agents must implement work strictly per week:

### W1 rules
- Implement only: prompt bank + GPT-2 greedy runner + caching/logging.
- No regard scoring yet.
- No decoding sweeps yet.

### W2 rules
- Add regard scoring (released classifier) + masking (XYZ) + baseline gap computation + sanity checks.

### W3 rules
- Run decoding grid exactly as specified:
  - Temperature: {0.7, 1.0, 1.3}
  - Top-k: {20, 50, 100}
  - Top-p: {0.8, 0.9, 0.95}
  - Optional anti-rep: no-repeat 3-gram
- Compute bias + quality metrics + bootstrap CIs.

### W4 rules (optional)
- Only if explicitly chosen: fine-tune BERT regard classifier and compare vs released.

### W5 rules
- Run ablations (masking sensitivity; anti-repetition).
- Finalize plots/tables and key findings summary.
- Enforce ethics constraints (no large raw output dumps; minimal excerpts + warnings).

---

## Reproducibility, caching, and logging (must be present throughout)
- Generation must log:
  - prompts, seeds, decoding config
  - max_new_tokens, N samples per prompt/demographic
- Caching is required (proposal mitigation for Kaggle limits):
  - cache generations
  - avoid recomputation on reruns
- Record environment snapshot (python/torch/transformers versions, device).
- Respect fairness constraints:
  - same checkpoint, same prompt bank, same N, same max length across decoding configs
  - multiple seeds per config and report variance

---

## Data handling / ethics rules (non-negotiable)
- Outputs may contain offensive text.
- Do NOT publish or commit large raw generations.
- Reports should summarize and include only minimal excerpts needed for analysis, with warnings.
- Treat `ewsheng/nlg-bias` as a research artifact if license is unclear; do not redistribute beyond allowed submission context.

---

## What agents should add to the repo (recommended)
- `.gitignore` to exclude:
  - `outputs/` (generations, metrics, reports)
  - large artifacts (`*.parquet`, cached models if any)
- `pyproject.toml`:
  - ruff config
  - pytest config
  - pinned core dependencies for reproducibility

---

## Definition of done for any agent task
A task is done only when:
- It matches the current week’s deliverables (no scope creep),
- Lint passes and tests pass,
- Results are reproducible via recorded config/env and caching works,
- Outputs respect ethics constraints (no raw dumps).