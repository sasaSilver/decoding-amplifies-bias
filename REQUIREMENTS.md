# REQUIREMENTS.md

## Project proposal implementation: “Decoding Amplifies Bias: Measuring ‘Regard’ in Open-Ended Text Generation”
This repository implements the study described in the project proposal:
- Question: does **decoding** (greedy / temperature / top-k / top-p) amplify or mitigate bias (measured as “regard”) **for a fixed checkpoint**?
- Framing: “regard” = how positively/negatively a generation portrays a demographic. :contentReference[oaicite:1]{index=1}

Team members (as in proposal):
- Ivan Chabanov
- Alexander Michailov :contentReference[oaicite:2]{index=2}

---

## Scope and boundaries (must follow proposal)

### In-scope
1) **Generator**: GPT-2 small (pretrained), **no fine-tuning**. :contentReference[oaicite:3]{index=3}  
2) **Decoding study**:
   - Baseline: greedy only.
   - Extension: decoding sweep (temperature / top-k / top-p) + optional anti-repetition. :contentReference[oaicite:4]{index=4}
3) **Bias scoring**: use the released **regard classifier** from the nlg-bias workflow to label generations and compute group-level bias. :contentReference[oaicite:5]{index=5}
4) **(Optional, strong)**: replicate classifier by fine-tuning a BERT regard classifier and compare vs released model (accuracy + agreement). :contentReference[oaicite:6]{index=6}
5) **Evaluation**: regard distribution per group, negative-regard gaps, uncertainty via bootstrap CIs; plus generation quality controls (distinct-1/2, repetition metrics, optional toxicity proxy aggregate). 
6) **Ethics/safety**: do not publish large raw outputs; include only minimal excerpts needed for analysis + warnings. :contentReference[oaicite:8]{index=8}

### Out-of-scope (for this repo)
- Fine-tuning the generator (GPT-2) (explicitly not in proposal). :contentReference[oaicite:9]{index=9}
- Anything beyond the specified decoding grid and optional ablations.
- Any deployment / productization.

---

## Data and resources (as specified)

### Data source(s)
- Regard-labeled dataset & tools: `sasha/regardv3`
  - dataset path: `data/regard/*.tsv`
  - label in column 1, text in column 2. :contentReference[oaicite:10]{index=10}
- Prompt templates: paper-style templates (occupation/descriptors) with a demographic slot; use a **fixed prompt bank** for reproducibility. :contentReference[oaicite:11]{index=11}
- Generated samples: GPT-2 completions per (prompt type × demographic × decoding). :contentReference[oaicite:12]{index=12}

### License / usage rights constraint
- Use released checkpoints under their published licenses/terms.
- If `sasha/regardv3` has no explicit license, treat it as a research artifact: use with attribution and do not redistribute beyond class submission. :contentReference[oaicite:13]{index=13}

---

## Preprocessing & protocol requirements (must implement)

### Preprocessing steps
- **Mask demographics for scoring**: replace demographic mention with `XYZ` before running the regard classifier (standard in nlg-bias workflow). :contentReference[oaicite:14]{index=14}
- Tokenize with BERT tokenizer; max length 128 (truncate/pad). :contentReference[oaicite:15]{index=15}
- Generation: fixed max new tokens; filter empty outputs; log prompts, seeds, and decoding config. :contentReference[oaicite:16]{index=16}

### Experimental protocol constraints (fairness constraints)
- Same checkpoint, same prompt bank, same #samples, same max length across decoding settings. :contentReference[oaicite:17]{index=17}
- Multiple seeds per config; report variance. :contentReference[oaicite:18]{index=18}

### Target scale (feasible on Kaggle)
- Fixed prompt bank: **30–80 prompts** × selected demographics (keep scope manageable). 
- Samples per prompt/demographic: **N = 50** (scale to Kaggle budget). 
- **3 seeds** per decoding config. 
- Score masked text; run a small ablation: masked vs unmasked scoring to test sensitivity. 

---

## Decoding grid (must match proposal)

Generation-time decoding settings:
- Greedy: argmax token
- Temperature sampling: T ∈ {0.7, 1.0, 1.3}
- Top-k sampling: k ∈ {20, 50, 100}
- Top-p (nucleus) sampling: p ∈ {0.8, 0.9, 0.95}
- Anti-repetition (optional): no-repeat 3-gram 

---

## Evaluation plan (must implement)

### Primary bias metrics
- Regard distribution per group: P(neg), P(neu), P(pos), P(other). 
- Negative-regard gap: Δneg between groups within the same prompt type. 
- Uncertainty: bootstrap confidence intervals over prompts/samples. 

### Generation quality controls (for interpretation)
- Diversity: distinct-1/2 (unique n-grams / total). 
- Degeneration: repeated n-gram rate; longest repetition span. :contentReference[oaicite:28]{index=28}
- (Optional) toxicity proxy score (aggregate only). :contentReference[oaicite:29]{index=29}

---

## Risk constraints (must be reflected in implementation choices)
- Kaggle runtime/quota may cap N or grid size → mitigate by caching generations; prioritize high-signal configs (greedy + one temp + one top-k + one top-p) if needed. 
- Regard subjectivity / classifier error → mitigate using CIs, manual spot-check (20–40 examples), and classifier agreement if fine-tuning is done. :contentReference[oaicite:31]{index=31}
- Failure mode: bias shifts due to verbosity/length changes under decoding → mitigate by fixing max length; normalizing by length; comparing per-prompt not only global aggregates. :contentReference[oaicite:32]{index=32}

---

## Ethics & safety requirements
- The study may surface offensive generations.
- Do not publish large raw outputs; include only minimal excerpts needed for analysis + warnings. :contentReference[oaicite:33]{index=33}
- No personal data; prompts are synthetic templates; datasets are research artifacts. :contentReference[oaicite:34]{index=34}

---

## Required artifacts (repo outputs)
Implementation must produce, at minimum:

### A) Prompt bank artifacts
- Fixed prompt bank file (versioned) with 30–80 prompts and selected demographics:
  - store prompt_id, prompt_type, demographic(s), prompt text

### B) Generation artifacts
- Saved generations for each (prompt × demographic × decoding × seed × sample_index)
- Must include in records:
  - prompt_id, demographic(s), decoding config, seed, max_new_tokens, sample index, raw text, completion text
- Must implement caching for generations. 

### C) Scoring artifacts
- Regard classifier outputs per generation:
  - predicted label in {neg, neu, pos, other}
  - record whether scoring used masked or unmasked text
- Group-level aggregations needed for bias metrics. 

### D) Metrics artifacts
- Bias metrics + quality metrics tables, including bootstrap CI outputs. 

### E) Report artifacts
- Final plots/tables and key findings summary (see W5). :contentReference[oaicite:38]{index=38}

---

# Timeline (Week-by-week milestones) — MUST MATCH PROPOSAL

## W1 — Prompt bank + GPT-2 runner; greedy baseline; caching/logging
Deliver:
- Prompt bank creation (fixed 30–80 prompts + selected demographics) and validation.
- GPT-2 runner (pretrained GPT-2 small), greedy decoding only.
- Caching for generations.
- Logging:
  - prompts, seeds, decoding config
  - run manifests (config + environment versions)
- Output: cached greedy generations ready for scoring. 

## W2 — Integrate regard scoring; baseline bias gaps + sanity checks
Deliver:
- Integrate released regard classifier (nlg-bias workflow).
- Implement demographic masking for scoring (XYZ replacement).
- Compute baseline:
  - regard distributions per group
  - Δneg gaps per prompt type
- Sanity checks:
  - verify label distribution looks reasonable
  - spot-check a small sample of scored outputs (documented)
- Output: baseline tables/plots for greedy. 

## W3 — Decoding grid runs; compute bias + quality metrics; bootstrap CIs
Deliver:
- Run decoding sweep (T / top-k / top-p) with fairness constraints:
  - same checkpoint, same prompt bank, same N, same max length
  - 3 seeds per config
- Compute:
  - regard distributions, Δneg gaps
  - quality metrics (distinct-1/2; repetition metrics; optional toxicity aggregate)
  - bootstrap confidence intervals over prompts/samples
- Output: metric tables + uncertainty, ready for final reporting. 

## W4 (Optional) — Fine-tune BERT regard classifier; compare agreement/robustness
Deliver (only if doing the optional extension):
- Fine-tune a BERT-base regard classifier on the released labeled set.
- Training details to record:
  - cross-entropy objective; AdamW; early stopping on dev metric
  - report batch size / LR / epochs and best checkpoint
- Compare to released classifier:
  - accuracy
  - agreement/robustness of bias-gap conclusions
- Output: comparison tables + notes. 

## W5 — Ablations (masking, anti-repetition); finalize plots/tables + key findings
Deliver:
- Ablations:
  - masked vs unmasked scoring sensitivity
  - anti-repetition (no-repeat 3-gram) effects (if used)
- Finalize:
  - plots/tables for decoding vs bias gaps + quality controls
  - written key findings summary suitable for submission
- Ensure ethics constraints: no large raw output dumps; minimal excerpts + warnings. 

---

## References (as listed in proposal)
- Sheng et al., EMNLP 2019
- Holtzman et al., ICLR 2020
- Radford et al., GPT-2, 2019
- Devlin et al., BERT, NAACL 2019 :contentReference[oaicite:44]{index=44}