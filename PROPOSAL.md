# Decoding Amplifies Bias: Measuring “Regard” in Open-Ended Text Generation

## Team members (full names)
- Ivan Chabanov (i.chabanov@innopolis.university)
- Alexander Michailov (i.michailov@innopolis.university)

## Problem Statement & Motivation

**Problem.** Language models can produce different generations for prompts that differ only by a demographic mention (e.g., gender, religion, nationality). These differences can manifest as social bias in open-ended generation. We adopt the “regard” framing: how positively/negatively a generation portrays the demographic.

**Open question we study.** Does the decoding algorithm (greedy / temperature / top-k / top-p) amplify or mitigate bias? Even with a fixed model checkpoint, decoding changes diversity and repetition; it may also shift the distribution of regard.

**Why it matters.**
- **Practical:** decoding is a deployment-time knob (no retraining) that might reduce harmful outputs.
- **Scientific:** separates “model bias” from “sampling-induced bias”.
- **Feasible:** end-to-end on Kaggle notebooks with controlled experiments and clear metrics.

## Data

**Data source(s).**
- Regard-labeled dataset & tools: `sasha/regardv3` (`data/regard/*.tsv`; label in col1, text in col2).
- Prompt templates: paper-style templates (e.g., occupation/descriptors) with a demographic slot; we will use a fixed prompt bank for reproducibility.
- Generated samples: GPT-2 completions per (prompt type × demographic × decoding).

**License / usage rights.**
- GPT-2 and BERT: use released checkpoints under their published licenses/terms (cite in References).
- `sasha/regardv3` repo: if no explicit license is present, treat as research artifact: use with attribution; do not redistribute beyond class submission.

**Preprocessing steps.**
- Mask demographics for scoring: replace demographic mention with `XYZ` before running regard classifier (standard in `sasha/regardv3` workflow).
- Tokenize with BERT tokenizer; max length 128 (truncate/pad).
- Generation: fixed max new tokens; filter empty outputs; log prompts, seeds, and decoding config.

## Baseline Plan

**Baseline (first implementation).**
- Generator: pretrained GPT-2 small (no fine-tuning).
- Decoding: greedy decoding only.
- Bias scoring: use released regard classifier from `sasha/regardv3` to label generations and compute group-level bias.

**Core extension (main contribution).**
- Decoding sweep: temperature/top-k/top-p (+ optional anti-repetition) and measure how regard gaps shift.
- Classifier replication (optional but strong): fine-tune a single BERT regard classifier on the released labeled set and compare to released model (accuracy + agreement).

**Decoding grid (generation-time only).**

| Method | Rule | Knob(s) |
|---|---|---|
| Greedy | argmax token | — |
| Temp sample | softmax(z/T) | T ∈ {0.7, 1.0, 1.3} |
| Top-k | sample from top k | k ∈ {20, 50, 100} |
| Top-p | nucleus mass ≥ p | p ∈ {0.8, 0.9, 0.95} |
| Anti-rep (opt.) | reduce repeats | no-repeat 3-gram |

**Fairness constraints.**
- Same checkpoint, same prompt bank, same #samples, same max length across decoding settings.
- Multiple seeds per config; report variance.

## Evaluation Plan

**Primary bias metrics.**
- Regard distribution per group: P(neg), P(neu), P(pos), P(other).
- Negative-regard gap: Δneg between groups within the same prompt type.
- Uncertainty: bootstrap confidence intervals over prompts/samples.

**Generation quality controls (interpretation).**
- Diversity: distinct-1/2 (unique n-grams / total).
- Degeneration: repeated n-gram rate; longest repetition span.
- (Optional) toxicity proxy score (aggregate only).

**Protocol.**
- Fixed prompt bank (30–80 prompts) × selected demographics (keep scope manageable).
- Samples per prompt/demographic: N = 50 (scale to Kaggle budget); 3 seeds per decoding config.
- Score masked text; run small ablation: masked vs unmasked scoring to test sensitivity.

## Risks

**Compute limitations.**
- Kaggle runtime/quota limits may cap N or grid size.
- Mitigation: cache generations; prioritize high-signal configs (greedy, one temp, one top-k, one top-p).

**Data quality concerns.**
- Regard is subjective; classifier error can skew gaps.
- Mitigation: CIs + manual spot-check (20–40 examples) + classifier agreement if we fine-tune.

**Potential failure modes.**
- Bias shifts due to verbosity/length changes under decoding.
- Mitigation: fix max length; normalize by length; compare per-prompt rather than only global aggregates.

## Ethics & Safety Considerations

**Misuse potential.**
- The study may surface offensive generations.
- Mitigation: do not publish large raw outputs; include only minimal excerpts needed for analysis + warnings.

**Privacy / sensitivity.**
- No personal data; prompts are synthetic templates; datasets are research artifacts.

## Timeline (Week-by-week)

- **W1:** Prompt bank + GPT-2 runner; greedy baseline; caching/logging.
- **W2:** Integrate regard scoring; baseline bias gaps + sanity checks.
- **W3:** Decoding grid runs; compute bias + quality metrics; bootstrap CIs.
- **W4:** (Optional) fine-tune BERT regard classifier; compare agreement/robustness.
- **W5:** Ablations (masking, anti-repetition); finalize plots/tables + key findings.

## What is happening inside?

**Applicability.** We do not fine-tune a very large generator; we use a small pretrained LM and (optionally) fine-tune a classifier.

**Model architecture.**
- Generator: GPT-2 (decoder-only Transformer, causal self-attention).
- Regard classifier: BERT-base (encoder-only Transformer) + linear head (4-way).

**Training objective & details.**
- Generator: no training; controlled decoding only.
- Classifier: cross-entropy; AdamW; early stopping on dev metric; report batch size/LR/epochs and best checkpoint.

## References (2–6)

- Sheng et al. *The Woman Worked as a Babysitter: On Biases in Language Generation.* EMNLP 2019.
- Holtzman et al. *The Curious Case of Neural Text Degeneration.* ICLR 2020.
- Radford et al. *Language Models are Unsupervised Multitask Learners (GPT-2).* 2019.
- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.