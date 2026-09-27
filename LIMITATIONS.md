# Limitations

## Known limitation: Qwen free-text generation instability under quantization

During pilot testing of the secondary (free-text) metric, Qwen2.5-7B-Instruct
intermittently produced corrupted output: non-English (CJK) tokens spliced
into otherwise-English text, occasionally followed by a hallucinated fake
conversation turn (the model continuing as if a new "user" message had
arrived). Mistral-7B-Instruct-v0.3 showed no equivalent behavior across the
same pilot questions.

**What was ruled out:**
- Not a missing end-of-turn token configuration: an explicit `eos_token_id`
  fix (covering both the tokenizer's default EOS and template-specific stop
  tokens like `<|im_end|>`) eliminated visible turn-leakage in the main pilot
  run, but the underlying corrupted-token issue persisted.
- Not resolved by reducing quantization severity: a diagnostic re-run of the
  four worst-affected questions at 8-bit (vs. the project's standard 4-bit
  NF4) showed identical corruption on all 4/4 questions.

**Likely cause:** the corrupted token typically appears at or near the point
where the model should emit its end-of-turn token, suggesting the model
occasionally substitutes an incorrect, embedding-space-adjacent token
(landing in Qwen's large CJK vocabulary) instead of the correct stop token
under quantization, which then allows generation to continue past where it
should have halted.

**Not confirmed:** a full-precision (bf16) baseline could not be tested as a
control: bf16 generation for a 7B model was already found to hang on this
project's hardware (16GB unified memory) during earlier feasibility testing,
so it is not confirmed whether this is fully eliminated at full precision.

**Scope of impact:** this affects only the secondary, descriptive/exploratory
free-text metric. The primary metric (MC1) is unaffected, since MC1 uses
loglikelihood scoring over fixed candidate answer text, not free generation,
and never invokes this failure mode.

**Decision:** both models remain at matched 4-bit NF4 quantization (per the
project's precision-parity requirement) for the secondary metric. This
limitation is disclosed rather than resolved, since a fix was not found
within the scope of this project's hardware constraints.
## Statistical power

The pre-study power analysis (`config/power_analysis.yaml`) showed that
several plausible effect sizes needed more questions than TruthfulQA has.
The observed effect falls in that range: 21% discordant pairs, split
58/42. That effect sits between the grid's 55% and 60% splits, which need
about 3,900 and 970 questions respectively. The pool has 817. A simulation
at the observed effect puts the power of the primary test at about 54%.

So the primary result (p = 0.040) comes from an underpowered comparison.
Significant results from underpowered designs tend to overestimate the
true effect. The 3.4-point gap should be read as "probably positive and
probably small", not as a precise estimate. Category-level tests have far
less power still. The category-level null result means "not detectable
with this data", not "no difference".

## The two category filters are independent

A category needs at least 20 questions to be considered, and separately
needs at least 6 discordant pairs to be tested. A category can pass the
first and fail the second: Economics has 31 questions but only 4
disagreements. This is expected, not a contradiction. Eight of the 15
size-eligible categories are untestable for this reason.

## Secondary metric: judge quality

The free-text metric depends on a small local judge (Phi-3-mini, 3.8B,
4-bit), and a spot check shows it discriminates poorly. On the Confusion
category, each question has a single correct name, so correctness can be
checked by string match. There, answers that lacked the correct name
scored the same on average as answers that contained it (about 0.83 vs
0.84). 93% of the wrong-name answers scored 0.5 or higher. The judge's own
rationales sometimes assert that an answer matches the reference when it
plainly does not, for example treating "Manning" as matching "Waugh"
(`config/confusion_check.yaml`).

Ten of 1,634 generations could not be scored because the judge's output
could not be parsed. They are left unscored rather than imputed. Together
with the garbling issue above, this is why the secondary metric is
descriptive only. Its means indicate direction at best.

## MC1 as a benchmark

- **Likelihood, not behaviour.** MC1 measures which fixed answer option a
  model assigns the highest likelihood. It does not measure what the model
  would say unprompted. It also depends on the harness's prompt format and
  its handling of the chat template.
- **Adversarial construction.** TruthfulQA questions are built to elicit
  imitative falsehoods, so accuracy on them does not generalise to
  everyday factual accuracy.
- **Contamination.** The dataset is public and old enough that it may
  appear in either model's training data, to an unknown and possibly
  different degree.

## Reproducibility caveats

- **GPU non-determinism.** MC1 is deterministic given fixed weights and
  inputs, but MPS kernels are not guaranteed bit-reproducible. Tiny
  numerical differences could flip a question whose top two options are
  near-tied.
- **Package versions.** Packages were installed unpinned and frozen
  afterwards, so the lm-eval task version is whatever that install
  resolved to.
- **Model revisions.** The models were loaded as revision `main`. The
  commit that `main` resolved to at run time is now recorded and pinned in
  `config/models.yaml`.
- **Pilot effect size.** The pilot's effect-size estimate (20 questions,
  4 discordant pairs) was too noisy to use and was not used for the power
  analysis.
