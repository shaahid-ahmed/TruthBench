"""Primary and category significance tests with thresholds read from config"""

import yaml
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from statsmodels.stats.multitest import multipletests

MODEL_A = "qwen"  # First model in every difference and ratio
MODEL_B = "mistral"

INPUT_PATH = "../data/full_mc1_results_combined.csv"
OUTPUT_PATH = "../data/significance_test_results.csv"

with open("../config/significance_tests.yaml") as f:
    cfg = yaml.safe_load(f)
with open("../config/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

ALPHA = cfg["alpha"]
EXACT_BELOW = cfg["exact_mcnemar_below_n_discordant"]
MIN_DISCORDANT = cfg["min_discordant_for_testable"]
N_BOOTSTRAP = cfg["bootstrap"]["n_resamples"]
CI_LEVEL = cfg["bootstrap"]["ci_level"]
SEED = cfg["bootstrap"]["seed"]
MIN_CATEGORY_SIZE = dataset_cfg["min_category_size_threshold"]
assert cfg["multiple_comparisons"]["method"] == "holm"

rng = np.random.default_rng(SEED)


def mcnemar_test(b, c):
    """Exact or asymptotic McNemar chosen by discordant count"""
    n = b + c
    if n < EXACT_BELOW:
        # Exact McNemar is a binomial test
        p = binomtest(b, n, 0.5, alternative="two-sided").pvalue if n > 0 else 1.0
        return "exact_binomial", np.nan, p
    stat = (abs(b - c) - 1) ** 2 / n
    return "asymptotic_cc", stat, chi2.sf(stat, df=1)


def odds_ratio_with_ci(b, c):
    """Paired odds ratio with an exact confidence interval"""
    n = b + c
    if n == 0:
        return np.nan, np.nan, np.nan
    or_hat = np.inf if c == 0 else b / c
    lo, hi = binomtest(b, n, 0.5).proportion_ci(confidence_level=1 - ALPHA, method="exact")
    to_or = lambda p: np.inf if p >= 1 else p / (1 - p)
    return or_hat, to_or(lo), to_or(hi)


def cohens_h(p_a, p_b):
    return 2 * np.arcsin(np.sqrt(p_a)) - 2 * np.arcsin(np.sqrt(p_b))


def paired_bootstrap_ci(a, b):
    """Percentile bootstrap interval on the paired accuracy difference"""
    n = len(a)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    tail = (1 - CI_LEVEL) / 2 * 100
    lo, hi = np.percentile(diffs, [tail, 100 - tail])
    return lo, hi


def analyse(sub):
    """All paired statistics for one set of questions"""
    a = sub[MODEL_A].to_numpy(dtype=float)
    b = sub[MODEL_B].to_numpy(dtype=float)
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    test_type, stat, p = mcnemar_test(a_only, b_only)
    or_hat, or_lo, or_hi = odds_ratio_with_ci(a_only, b_only)
    boot_lo, boot_hi = paired_bootstrap_ci(a, b)
    return {
        "n": len(sub),
        "n_discordant": a_only + b_only,
        f"{MODEL_A}_only_correct": a_only,
        f"{MODEL_B}_only_correct": b_only,
        f"{MODEL_A}_acc": a.mean(),
        f"{MODEL_B}_acc": b.mean(),
        "acc_diff": a.mean() - b.mean(),
        "test_type": test_type,
        "statistic": stat,
        "p_raw": p,
        "cohens_h": cohens_h(a.mean(), b.mean()),
        "odds_ratio": or_hat,
        "odds_ratio_ci_low": or_lo,
        "odds_ratio_ci_high": or_hi,
        "bootstrap_ci_low": boot_lo,
        "bootstrap_ci_high": boot_hi,
    }


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return "inf" if np.isinf(x) else f"{x:.{nd}f}"


# Load and confirm binary paired outcomes
df = pd.read_csv(INPUT_PATH)
assert set(df["model"].unique()) == {MODEL_A, MODEL_B}, df["model"].unique()
assert set(df["correct"].unique()) <= {True, False}, "T1: MC1 outcomes are not binary"
assert not df.duplicated(["model", "question"]).any()

pivot = df.pivot(index="question", columns="model", values="correct")
assert not pivot.isna().any().any(), "some questions are missing a model's result"
categories = df.drop_duplicates("question").set_index("question")["category"]
assert categories.groupby(level=0).nunique().max() == 1
pivot["category"] = categories.reindex(pivot.index)
assert not pivot["category"].isna().any()

print("=" * 78)
print("Parameters (from config/significance_tests.yaml and config/dataset.yaml)")
print(f"  alpha                              : {ALPHA}")
print(f"  E9 min category size               : {MIN_CATEGORY_SIZE}")
print(f"  exact McNemar if discordant <      : {EXACT_BELOW}")
print(f"  T6 untestable if discordant <      : {MIN_DISCORDANT}")
print(f"  bootstrap resamples                : {N_BOOTSTRAP} ({int(CI_LEVEL*100)}% percentile CI)")
print(f"  bootstrap RNG seed                 : {SEED}")
print(f"  T1 binary outcomes confirmed; {len(pivot)} fully paired questions")
print(f"  direction: differences = {MODEL_A} - {MODEL_B}; OR > 1 favours {MODEL_A}")

# Primary test
primary = analyse(pivot)
p_sig = primary["p_raw"] < ALPHA

print("\n" + "=" * 78)
print("PRIMARY (pre-registered, uncorrected): overall MC1, all questions")
print("=" * 78)
print(f"  n questions          : {primary['n']}")
print(f"  {MODEL_A} accuracy        : {primary[f'{MODEL_A}_acc']:.4f}")
print(f"  {MODEL_B} accuracy     : {primary[f'{MODEL_B}_acc']:.4f}")
print(f"  difference ({MODEL_A}-{MODEL_B}): {primary['acc_diff']:+.4f}")
print(f"  discordant pairs     : {primary['n_discordant']} "
      f"({MODEL_A}-only correct {primary[f'{MODEL_A}_only_correct']}, "
      f"{MODEL_B}-only correct {primary[f'{MODEL_B}_only_correct']})")
print(f"  McNemar test used    : {primary['test_type']} "
      f"(discordant {primary['n_discordant']} {'<' if primary['n_discordant'] < EXACT_BELOW else '>='} {EXACT_BELOW})")
if primary["test_type"] == "asymptotic_cc":
    print(f"  chi-square statistic : {primary['statistic']:.4f}")
print(f"  p-value              : {primary['p_raw']:.4f}  -> "
      f"{'SIGNIFICANT' if p_sig else 'NOT significant'} at alpha = {ALPHA}")
print(f"  Cohen's h            : {primary['cohens_h']:+.4f}")
print(f"  odds ratio           : {fmt(primary['odds_ratio'])}  "
      f"(95% exact CI {fmt(primary['odds_ratio_ci_low'])} – {fmt(primary['odds_ratio_ci_high'])})")
print(f"  paired bootstrap CI  : {primary['bootstrap_ci_low']:+.4f} to {primary['bootstrap_ci_high']:+.4f}")

# Category tests
cat_sizes = pivot["category"].value_counts()
rows = []
print("\n" + "=" * 78)
print("CATEGORY-LEVEL (exploratory, Holm-corrected)")
print("=" * 78)
print(f"\n2a. E9 filter: drop categories with fewer than {MIN_CATEGORY_SIZE} questions:")
for category, n_cat in cat_sizes.items():
    sub = pivot[pivot["category"] == category]
    if n_cat < MIN_CATEGORY_SIZE:
        print(f"  DROPPED  {category:<22} n={n_cat:>3} (< {MIN_CATEGORY_SIZE})")
        a, b = sub[MODEL_A].astype(bool), sub[MODEL_B].astype(bool)
        rows.append({
            "category": category, "n": int(n_cat),
            "n_discordant": int((a != b).sum()),
            "e9_status": f"dropped (n<{MIN_CATEGORY_SIZE})",
            "t6_status": "not evaluated (failed E9)",
            "test_type": "not tested",
        })
        continue
    r = analyse(sub)
    r["category"] = category
    r["e9_status"] = "pass"
    if r["n_discordant"] < MIN_DISCORDANT:
        r["t6_status"] = f"untestable (discordant<{MIN_DISCORDANT})"
        # No p value for untestable categories
        r["p_raw"] = np.nan
        r["statistic"] = np.nan
        r["test_type"] = "not tested"
    else:
        r["t6_status"] = "testable"
    rows.append(r)

e9_pass = [r["category"] for r in rows if r["e9_status"] == "pass"]
assert sorted(e9_pass) == sorted(dataset_cfg["usable_categories"]), \
    "E9 survivors don't match config/dataset.yaml usable_categories"
print(f"  -> {len(e9_pass)} categories pass E9 "
      f"({sum(r['n'] for r in rows if r['e9_status'] == 'pass')} questions), "
      f"{len(rows) - len(e9_pass)} dropped")

print(f"\n2c. T6 filter: among E9 survivors, flag untestable if discordant pairs < {MIN_DISCORDANT}.")
print("    E9 (enough questions) and T6 (enough disagreement between the models) are")
print("    two INDEPENDENT filters: passing E9 and failing T6 is expected, not a bug.")
for r in rows:
    if r["e9_status"] == "pass" and r["t6_status"] != "testable":
        print(f"  UNTESTABLE {r['category']:<22} n={r['n']:>3}, discordant={r['n_discordant']}")

tested = [r for r in rows if r.get("t6_status") == "testable"]
if tested:
    reject, p_holm, _, _ = multipletests([r["p_raw"] for r in tested], alpha=ALPHA, method="holm")
    for r, ph, rej in zip(tested, p_holm, reject):
        r["p_holm_corrected"] = ph
        r["significant_after_correction"] = bool(rej)
for r in rows:
    r.setdefault("p_holm_corrected", np.nan)
    r.setdefault("significant_after_correction", False)
    r["scope"] = "category"

# Output
SPEC_COLS = [
    "category", "n", "n_discordant", "e9_status", "t6_status", "p_raw",
    "p_holm_corrected", "significant_after_correction", "cohens_h",
    "odds_ratio", "bootstrap_ci_low", "bootstrap_ci_high",
]
EXTRA_COLS = [
    "test_type", "statistic", f"{MODEL_A}_acc", f"{MODEL_B}_acc", "acc_diff",
    f"{MODEL_A}_only_correct", f"{MODEL_B}_only_correct",
    "odds_ratio_ci_low", "odds_ratio_ci_high",
]
table = pd.DataFrame(rows).reindex(columns=["scope"] + SPEC_COLS + EXTRA_COLS)
table["_k"] = table["t6_status"].map(lambda s: 0 if s == "testable" else (1 if s.startswith("untestable") else 2))
table = table.sort_values(["_k", "n"], ascending=[True, False]).drop(columns="_k")

print("\n2b/2d. Results table (qwen - mistral; Holm across testable categories only)")
show = table[SPEC_COLS + ["test_type"]].copy()
for col in ["p_raw", "p_holm_corrected", "cohens_h", "bootstrap_ci_low", "bootstrap_ci_high"]:
    show[col] = show[col].map(fmt)
show["odds_ratio"] = show["odds_ratio"].map(lambda x: fmt(x, 3))
show = show.rename(columns={
    "n_discordant": "disc", "significant_after_correction": "sig_holm",
    "bootstrap_ci_low": "boot_lo", "bootstrap_ci_high": "boot_hi",
    "p_holm_corrected": "p_holm",
})
with pd.option_context("display.width", 250, "display.max_columns", None):
    print(show.to_string(index=False))

primary_row = {"scope": "primary", "category": "ALL (primary, uncorrected)",
               "e9_status": "n/a", "t6_status": "n/a",
               "p_holm_corrected": np.nan,
               "significant_after_correction": np.nan, **primary}
out = pd.concat([pd.DataFrame([primary_row]).reindex(columns=table.columns), table],
                ignore_index=True)
out.to_csv(OUTPUT_PATH, index=False)

n_tested = len(tested)
n_survived = sum(r["significant_after_correction"] for r in tested)

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  Primary: {MODEL_A} {primary[f'{MODEL_A}_acc']:.4f} vs {MODEL_B} {primary[f'{MODEL_B}_acc']:.4f}, "
      f"diff {primary['acc_diff']:+.4f} [{primary['bootstrap_ci_low']:+.4f}, {primary['bootstrap_ci_high']:+.4f}], "
      f"McNemar ({primary['test_type']}) p = {primary['p_raw']:.4f} "
      f"-> {'significant' if p_sig else 'not significant'} at {ALPHA}")
print(f"  Categories: {len(rows)} total, {len(e9_pass)} passed E9, "
      f"{len(e9_pass) - n_tested} flagged untestable at T6, {n_tested} tested")
print(f"  Saved {len(out)} rows (1 primary + {len(table)} categories) to {OUTPUT_PATH}")
print()
print(f"{n_survived} of {n_tested} categories survived Holm correction")
if n_survived == 0:
    print()
    print("*** NULL RESULT AT CATEGORY LEVEL: NO CATEGORY-LEVEL DIFFERENCE DISTINGUISHABLE ***")
    print("*** FROM NOISE AFTER CORRECTION. PRIMARY MC1 COMPARISON (STEP 1) IS THE        ***")
    print("*** REPORTABLE FINDING.  -> W2 takes the NULL-RESULT branch.                   ***")
else:
    names = [r["category"] for r in tested if r["significant_after_correction"]]
    print(f"CATEGORY-LEVEL FINDINGS SURVIVE CORRECTION: {', '.join(names)} "
          f"-> W2 leads with these (primary MC1 comparison still reported as the pre-registered result).")
