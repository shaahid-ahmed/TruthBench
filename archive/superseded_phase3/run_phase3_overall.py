"""
T1-T5: The PRIMARY, pre-registered comparison (per config/preregistration.yaml) —
overall MC1 accuracy, Qwen2.5-7B-Instruct vs Mistral-7B-Instruct-v0.3,
across the full 817-question pool.

T1: confirm binary data type (already known, verified here from real data)
T2: McNemar's test (exact if discordant pairs < 25, asymptotic w/ continuity
    correction otherwise — same threshold logic used for categories in T6)
T3: confidence interval for each model's overall accuracy (Wilson score)
T4: paired bootstrap as a cross-check on the same comparison
T5: effect size for paired binary data — conditional odds ratio (from the
    McNemar table) and Cohen's h (from the two accuracies), NOT Cohen's d
"""

import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

ALPHA = 0.05
N_BOOTSTRAP = 10000
RANDOM_SEED = 42

df = pd.read_csv("../data/full_mc1_results_combined.csv")

model_names = sorted(df["model"].unique())
assert len(model_names) == 2, f"Expected exactly 2 models, found {model_names}"
model_a, model_b = model_names

is_binary = set(df["correct"].unique()) <= {True, False}
print(f"T1: binary confirmed: {is_binary}")
assert is_binary

pivot = df.pivot(index="question", columns="model", values="correct")
n_total = len(pivot)
print(f"\nTotal paired questions: {n_total}")

both_correct = ((pivot[model_a] == True) & (pivot[model_b] == True)).sum()
both_wrong = ((pivot[model_a] == False) & (pivot[model_b] == False)).sum()
a_right_b_wrong = ((pivot[model_a] == True) & (pivot[model_b] == False)).sum()
b_right_a_wrong = ((pivot[model_a] == False) & (pivot[model_b] == True)).sum()
n_discordant = a_right_b_wrong + b_right_a_wrong

print(f"Both correct: {both_correct}, Both wrong: {both_wrong}")
print(f"{model_a} right/{model_b} wrong: {a_right_b_wrong}, {model_b} right/{model_a} wrong: {b_right_a_wrong}")
print(f"Discordant pairs: {n_discordant}")

contingency_table = [[both_correct, a_right_b_wrong], [b_right_a_wrong, both_wrong]]
use_exact = n_discordant < 25
print(f"\nT2: using {'EXACT' if use_exact else 'asymptotic (with correction)'} McNemar's test")

mcnemar_result = mcnemar(contingency_table, exact=use_exact, correction=not use_exact)
print(f"McNemar's statistic: {mcnemar_result.statistic:.4f}, p-value: {mcnemar_result.pvalue:.4f}")

acc_a = pivot[model_a].mean()
acc_b = pivot[model_b].mean()
ci_a = proportion_confint(pivot[model_a].sum(), n_total, alpha=ALPHA, method="wilson")
ci_b = proportion_confint(pivot[model_b].sum(), n_total, alpha=ALPHA, method="wilson")

print(f"\nT3: {model_a} accuracy: {acc_a:.4f}, 95% CI: ({ci_a[0]:.4f}, {ci_a[1]:.4f})")
print(f"T3: {model_b} accuracy: {acc_b:.4f}, 95% CI: ({ci_b[0]:.4f}, {ci_b[1]:.4f})")

rng = np.random.default_rng(RANDOM_SEED)
questions = pivot.index.to_numpy()
bootstrap_diffs = []
for _ in range(N_BOOTSTRAP):
    sample_idx = rng.choice(len(questions), size=len(questions), replace=True)
    sample = pivot.iloc[sample_idx]
    bootstrap_diffs.append(sample[model_a].mean() - sample[model_b].mean())

bootstrap_diffs = np.array(bootstrap_diffs)
observed_diff = acc_a - acc_b
boot_ci_low, boot_ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])
boot_p_value_proxy = 2 * min((bootstrap_diffs >= 0).mean(), (bootstrap_diffs <= 0).mean())

print(f"\nT4: Paired bootstrap ({N_BOOTSTRAP} resamples)")
print(f"Observed difference: {observed_diff:.4f}, 95% CI: ({boot_ci_low:.4f}, {boot_ci_high:.4f})")
print(f"Two-sided p-value proxy: {boot_p_value_proxy:.4f}")

if b_right_a_wrong == 0:
    odds_ratio = np.inf
    or_ci = (np.nan, np.nan)
else:
    odds_ratio = a_right_b_wrong / b_right_a_wrong
    log_or_se = np.sqrt(1 / a_right_b_wrong + 1 / b_right_a_wrong) if a_right_b_wrong > 0 else np.nan
    log_or = np.log(odds_ratio)
    z = norm.ppf(1 - ALPHA / 2)
    or_ci = (np.exp(log_or - z * log_or_se), np.exp(log_or + z * log_or_se))

phi_a = 2 * np.arcsin(np.sqrt(acc_a))
phi_b = 2 * np.arcsin(np.sqrt(acc_b))
cohens_h = phi_a - phi_b

print(f"\nT5: Conditional odds ratio: {odds_ratio:.4f}, 95% CI: ({or_ci[0]:.4f}, {or_ci[1]:.4f})")
print(f"Cohen's h: {cohens_h:.4f}")

output = {
    "primary_comparison": f"{model_a} vs {model_b}, overall MC1 accuracy",
    "n_total_questions": int(n_total),
    "T1_binary_confirmed": bool(is_binary),
    "contingency_table": {
        "both_correct": int(both_correct), "both_wrong": int(both_wrong),
        f"{model_a}_right_{model_b}_wrong": int(a_right_b_wrong),
        f"{model_b}_right_{model_a}_wrong": int(b_right_a_wrong),
        "n_discordant": int(n_discordant),
    },
    "T2_mcnemar": {
        "test_type": "exact" if use_exact else "asymptotic_with_correction",
        "statistic": float(mcnemar_result.statistic),
        "p_value": float(mcnemar_result.pvalue),
    },
    "T3_accuracy_with_ci": {
        model_a: {"accuracy": float(acc_a), "ci_95_wilson": [float(ci_a[0]), float(ci_a[1])]},
        model_b: {"accuracy": float(acc_b), "ci_95_wilson": [float(ci_b[0]), float(ci_b[1])]},
    },
    "T4_paired_bootstrap": {
        "n_resamples": N_BOOTSTRAP,
        "observed_difference": float(observed_diff),
        "ci_95": [float(boot_ci_low), float(boot_ci_high)],
        "two_sided_p_value_proxy": float(boot_p_value_proxy),
        "note": "Cross-check on the SAME comparison as T2, not a separate finding",
    },
    "T5_effect_size": {
        "conditional_odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else None,
        "odds_ratio_ci_95": [float(or_ci[0]), float(or_ci[1])] if np.isfinite(odds_ratio) else None,
        "cohens_h": float(cohens_h),
    },
}

with open("../config/phase3_overall_results.yaml", "w") as f:
    yaml.dump(output, f, sort_keys=False, default_flow_style=False)

print("\nSaved full T1-T5 results to ../config/phase3_overall_results.yaml")