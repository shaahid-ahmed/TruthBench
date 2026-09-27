# TruthBench

A pre-registered, paired statistical comparison of Qwen2.5-7B-Instruct and
Mistral-7B-Instruct-v0.3 on TruthfulQA. It tests whether the models differ
in factual accuracy, and whether any difference is concentrated in
particular question categories.

## Results at a glance

![MC1 accuracy by category](figures/mc1_by_category.png)

- **Primary (pre-registered).** Mistral scored higher on MC1: 50.7% vs
  47.2% over all 817 questions.
  - McNemar p = 0.040.
  - Accuracy difference −3.4 points, paired-bootstrap 95% CI [−6.6, −0.2].
  - Cohen's h = −0.07.
  - The difference is significant but small, and it comes from an
    underpowered comparison (about 54% power at the observed effect).
- **Category level (exploratory).** A null result. Of 15 categories large
  enough to consider, 8 had too few model disagreements to test. None of
  the other 7 survived Holm correction; the closest was Confusion, at Holm
  p = 0.055.
- **Secondary free-text metric (descriptive only).** Mistral 0.80 vs Qwen
  0.75 mean GEval score. The direction matches MC1, but the small local
  judge discriminates poorly, so these means are indicative only.

Full numbers are in [RESULTS.md](./RESULTS.md), methods in
[METHODOLOGY.md](./METHODOLOGY.md), and caveats in
[LIMITATIONS.md](./LIMITATIONS.md).

## Design

- **Models.** Both are pinned to exact commits (`config/models.yaml`) and
  run under identical 4-bit NF4 quantization. Full bf16 did not fit in 16 GB
  of unified memory.
- **Primary metric: MC1** (binary, loglikelihood-scored), computed with
  `lm-evaluation-harness`'s `truthfulqa_mc1` task.
  - The chat template is applied for both models.
  - Both get an identical explicit system instruction. Without it, Qwen's
    template auto-injects its own system prompt and Mistral's has no system
    slot.
- **Pre-registration.** The primary metric and the primary comparison were
  committed before any full-scale scoring (`config/preregistration.yaml`).
- **Primary test.** McNemar's test on the paired outcomes. The exact version
  is used below 25 discordant pairs, otherwise the asymptotic version with
  continuity correction.
  - Effect sizes: Cohen's h and the paired odds ratio (exact CI).
  - Paired bootstrap as a cross-check.
  - Not corrected for multiple comparisons, since it is the single
    pre-registered test.
- **Category tests.** The same tests are run per category, behind two
  independent filters:
  - at least 20 questions in the category;
  - at least 6 discordant pairs (below that, even an exact test cannot
    reach p < 0.05).
  - Holm correction is applied across the tested categories only.
- **Secondary metric.** Free-text answers (temperature 0.7) are judged by a
  DeepEval GEval metric with a local Phi-3-mini judge. Reported as means
  only: no tests, no inferential claims.

All thresholds are in `config/`. Nothing in the statistics depends on a
default buried in code.

## Dataset

- **Source.** Official
  [`truthfulqa/truthful_qa`](https://huggingface.co/datasets/truthfulqa/truthful_qa),
  817 questions.
- **Category merging.** Three subcategory families (`Indexical Error: *`,
  `Confusion: *`, `Misconceptions` + `Misconceptions: Topical`) were merged
  into their parents before the 20-question size threshold was applied.
  This recovered 63 questions that would otherwise have been excluded only
  because of the dataset's fine-grained labels.
- **Usable pool.** 605 of 817 questions are eligible for category-level
  testing. All 817 are used for the overall comparison.

## Project structure

```
config/          frozen settings (models, scoring, dataset, stats thresholds, prereg)
                 and small YAML result summaries
data/            dataset copy, per-question results, combined results table
figures/         result figure
scripts/         pipeline, analysis, and verification scripts
METHODOLOGY.md   full methods
RESULTS.md       full results
LIMITATIONS.md   known issues and caveats
```

Key outputs:
- `data/combined_results.csv`: one row per model × question, with both
  pipelines.
- `data/significance_test_results.csv`: every test statistic.

## Reproducing

See [METHODOLOGY.md § Reproducing](./METHODOLOGY.md#reproducing).
The analysis scripts run from the committed CSVs alone. Generation and
judging need the models and roughly a day of Apple Silicon compute.

## Status

Data collection, statistical analysis, and write-up are complete.

The repeated-sampling variance check for the secondary metric
(`scripts/run_secondary_variance_check.py`) is in progress. It is
descriptive only and does not affect the primary result.
