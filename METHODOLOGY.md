# Methodology

## Question

Do Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 differ in factual
accuracy on TruthfulQA, and if so, is the difference concentrated in
particular question categories?

## Models

| | Qwen | Mistral |
|---|---|---|
| Repo | `Qwen/Qwen2.5-7B-Instruct` | `mistralai/Mistral-7B-Instruct-v0.3` |
| Commit | `a09a3545…` | `c170c708…` |
| Precision | 4-bit NF4, bf16 compute | 4-bit NF4, bf16 compute |

- **Commits.** The runs loaded revision `main`. The commits above are what
  `main` resolved to, taken from the only cached snapshot, which was
  downloaded before any scored run. `config/models.yaml` now pins them
  explicitly.
- **Why 4-bit.** Full bf16 needs about 15 GB of weights per model, which
  doesn't fit alongside anything else in 16 GB of unified memory; generation
  hung under memory pressure (`scripts/check_hardware_feasibility.py`).
  Both models run under identical 4-bit settings instead. Matching dtype
  and quantization were verified (`config/model_quantization_check.yaml`),
  and memory stayed stable across six load/generate/unload cycles
  (`config/memory_stability_check.yaml`).
- **Hardware.** Apple Silicon Mac with 16 GB unified memory, PyTorch MPS
  backend.

## Data

- **Source.** Official `truthfulqa/truthful_qa`, 817 questions.
- **Category merging.** Three genuine subcategory families were merged into
  their parents before the size filter: `Indexical Error: *`,
  `Confusion: *`, and `Misconceptions: Topical`.
- **Minimum category size of 20 questions.** 15 categories (605 questions)
  are eligible for per-category tests. 17 smaller categories (212 questions)
  are excluded from category-level testing but included in the overall
  comparison (`config/dataset.yaml`).

## Primary metric: MC1

- **Harness.** `lm-evaluation-harness` 0.4.13, task `truthfulqa_mc1`,
  with per-question logged samples. MC1 is binary per question: did the
  model assign the highest likelihood to the single correct option?
- **Chat template.** `apply_chat_template=True` for both models, with an
  identical explicit system instruction ("You are a helpful assistant.").
  Without it, Qwen2.5's template injects its own default system prompt and
  Mistral's has no system slot, so the two models would see asymmetric
  prompts. Both runs use the same code path and the same config value.
- **Version caveat.** Packages were installed unpinned and frozen to
  `requirements.txt` afterwards. The lm-eval task version is therefore
  whatever that install resolved to, not a deliberately selected
  known-good version.
- **Pre-registration.** The metric and the primary comparison were locked in
  `config/preregistration.yaml` before any full-scale scoring.

## Statistical analysis

All thresholds live in config (`config/significance_tests.yaml`,
`config/dataset.yaml`). The whole analysis is one script,
`scripts/run_significance_tests.py`.

**Primary test, pre-registered and uncorrected.**
- **Test.** McNemar's test on the 817 paired outcomes. The exact binomial
  version is used when there are fewer than 25 discordant pairs; otherwise
  the chi-square version with continuity correction. The choice is made
  from the observed count.
- **Effect sizes.**
  - Cohen's h.
  - Conditional (paired) odds ratio, with an exact CI from the
    Clopper–Pearson interval on the discordant split.
  - Paired bootstrap 95% percentile CI on the accuracy difference (10,000
    resamples of questions, seed 42).
- **Descriptive intervals.** Per-model accuracy with Wilson 95% CIs.

**Category-level tests, exploratory.** Two independent filters apply:

1. **Size filter.** A category must have at least 20 questions.
2. **Discordant-pairs filter.** A category that passes the size filter is
   *untestable* if it has fewer than 6 discordant pairs. With n discordant
   pairs, the smallest possible two-sided exact McNemar p-value is
   2 × 0.5ⁿ. For n ≤ 5 that is at least 0.0625, so the test cannot reach
   α = 0.05 however the pairs split. Including such categories would only
   enlarge the correction family.

A category can pass the first filter and fail the second. That is expected:
having enough questions is not the same as having enough disagreement
between the models.

Each testable category gets the same test and effect sizes as the primary
comparison. Holm–Bonferroni correction (α = 0.05) is applied across the
testable categories only; the primary test is not in that family.

## Secondary metric: free-text truthfulness (descriptive only)

- **Generation.** Zero-shot prompt `Question: {q}\nAnswer:`, chat-templated
  with the same system instruction, temperature 0.7, at most 100 new tokens
  (`config/secondary_metric_generation.yaml`).
- **Stop tokens.** Qwen's `<|im_end|>` is added to the stop tokens, and any
  output containing a leaked conversation turn is truncated.
- **Scoring.** A DeepEval GEval "Truthfulness" metric compares each answer
  with TruthfulQA's best answer and known incorrect answers. The judge is a
  local `microsoft/Phi-3-mini-4k-instruct` (commit `f39ac1d2…`), 4-bit,
  with greedy decoding.
- **Garbling flags.** Each generation is flagged for CJK characters and for
  leak truncation. Means are reported with and without garbled rows.
- **Scope.** Reported as mean scores only: no significance test, no
  correction, no inferential claim (`scripts/summarise_secondary_metric.py`).
- **Repeated-sampling check.** Because generation is sampled at temperature
  0.7, a seeded 60-question subset is regenerated twice more per model and
  re-judged, to show how much scores move between samples
  (`scripts/run_secondary_variance_check.py`).

## Reproducing

Run these from `scripts/`, in order:

| Step | Script(s) |
|---|---|
| Data | `prepare_dataset.py` |
| MC1 scoring | `run_full_mc1.py` |
| Free-text generation | `run_full_secondary.py` |
| Free-text judging | `score_full_secondary_deepeval.py` |
| Analysis | `run_significance_tests.py`, `summarise_secondary_metric.py`, `check_confusion_category.py`, `build_combined_results.py`, `plot_results.py` |
| Variance check | `run_secondary_variance_check.py` |

The analysis scripts need only the CSVs in `data/`. The generation and
judging scripts need the models and an Apple Silicon Mac (or they fall back
to CPU).
