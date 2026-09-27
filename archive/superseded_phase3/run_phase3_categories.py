"""
T6-T10: Category-level comparison, restricted to usable categories
(per config/dataset.yaml — categories below the size threshold were
already excluded in E9).

T6: McNemar's/exact-McNemar's per category. Categories with fewer than
    5 discordant pairs are flagged UNTESTABLE (p-value would be
    meaningless/unstable at that sparsity) rather than reported.
T7: Holm correction applied ONLY across testable categories (untestable
    ones are excluded from the correction family entirely, since they
    have no p-value to correct).
T8: final results table.
T9: explicit note distinguishing the overall (uncorrected, single test,
    from T2) comparison from these (corrected, multiple tests) category
    comparisons.
T10: identify which category findings survive correction.
"""

import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05
MIN_DISCORDANT_FOR_TEST = 5   # below this, flag untestable rather than report
EXACT_THRESHOLD = 25          # below this, use exact McNemar's; at/above, asymptotic+correction

df = pd.read_csv("../data/full_mc1_results_combined.csv")

with open("../config/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

usable_categories = dataset_cfg["usable_categories"]

model_names = sorted(df["model"].unique())
model_a, model_b = model_names

results_by_category = []

for category in usable_categories:
    cat_df = df[df["category"] == category]
    pivot = cat_df.pivot(index="question", columns="model", values="correct")
    n_cat = len(pivot)

    both_correct = ((pivot[model_a] == True) & (pivot[model_b] == True)).sum()
    both_wrong = ((pivot[model_a] == False) & (pivot[model_b] == False)).sum()
    a_right_b_wrong = ((pivot[model_a] == True) & (pivot[model_b] == False)).sum()
    b_right_a_wrong = ((pivot[model_a] == False) & (pivot[model_b] == True)).sum()
    n_discordant = a_right_b_wrong + b_right_a_wrong

    acc_a = pivot[model_a].mean()
    acc_b = pivot[model_b].mean()
    ci_a = proportion_confint(pivot[model_a].sum(), n_cat, alpha=ALPHA, method="wilson")
    ci_b = proportion_confint(pivot[model_b].sum(), n_cat, alpha=ALPHA, method="wilson")

    row = {
        "category": category,
        "n_questions": int(n_cat),
        f"{model_a}_accuracy": float(acc_a),
        f"{model_a}_ci": [float(ci_a[0]), float(ci_a[1])],
        f"{model_b}_accuracy": float(acc_b),
        f"{model_b}_ci": [float(ci_b[0]), float(ci_b[1])],
        "n_discordant": int(n_discordant),
    }

    if n_discordant < MIN_DISCORDANT_FOR_TEST:
        row["untestable"] = True
        row["p_value"] = None
        row["odds_ratio"] = None
        print(f"{category}: UNTESTABLE (only {n_discordant} discordant pairs)")
    else:
        contingency_table = [[both_correct, a_right_b_wrong], [b_right_a_wrong, both_wrong]]
        use_exact = n_discordant < EXACT_THRESHOLD
        mcnemar_result = mcnemar(contingency_table, exact=use_exact, correction=not use_exact)

        if b_right_a_wrong == 0 or a_right_b_wrong == 0:
            odds_ratio = np.inf if a_right_b_wrong > 0 else 0.0
        else:
            odds_ratio = a_right_b_wrong / b_right_a_wrong

        row["untestable"] = False
        row["test_type"] = "exact" if use_exact else "asymptotic_with_correction"
        row["p_value"] = float(mcnemar_result.pvalue)
        row["odds_ratio"] = float(odds_ratio) if np.isfinite(odds_ratio) else None
        print(f"{category}: n={n_cat}, discordant={n_discordant}, "
              f"p={mcnemar_result.pvalue:.4f} ({row['test_type']})")

    results_by_category.append(row)

# Holm correction, applied ONLY across testable categories
testable = [r for r in results_by_category if not r["untestable"]]
untestable = [r for r in results_by_category if r["untestable"]]

if testable:
    p_values = [r["p_value"] for r in testable]
    reject, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method="holm")
    for r, p_corr, sig in zip(testable, p_corrected, reject):
        r["p_value_corrected_holm"] = float(p_corr)
        r["significant_after_correction"] = bool(sig)

# final results table
table_rows = []
for r in results_by_category:
    table_rows.append({
        "category": r["category"],
        "n_questions": r["n_questions"],
        f"{model_a}_accuracy": round(r[f"{model_a}_accuracy"], 4),
        f"{model_b}_accuracy": round(r[f"{model_b}_accuracy"], 4),
        "n_discordant": r["n_discordant"],
        "untestable": r["untestable"],
        "p_value_raw": round(r["p_value"], 4) if r["p_value"] is not None else None,
        "p_value_corrected": round(r.get("p_value_corrected_holm"), 4) if r.get("p_value_corrected_holm") is not None else None,
        "significant_after_correction": r.get("significant_after_correction", False),
        "odds_ratio": round(r["odds_ratio"], 3) if r.get("odds_ratio") is not None else None,
    })

table_df = pd.DataFrame(table_rows)
print("\n=== T8: Final category results table ===")
print(table_df.to_string(index=False))

# which findings survive correction
surviving = table_df[table_df["significant_after_correction"] == True]
print(f"\nCategories surviving Holm correction")
if len(surviving) > 0:
    print(surviving[["category", f"{model_a}_accuracy", f"{model_b}_accuracy", "p_value_corrected"]].to_string(index=False))
else:
    print("No categories survived correction — no category-level finding is safe to lead with.")
    print("The overall comparison (T2, uncorrected, single pre-registered test) remains the primary finding.")

# explicit note
print("\nCorrection scope note")
print("The OVERALL comparison (T2) is a single pre-registered test — reported")
print("WITHOUT correction, per the pre-registration (config/preregistration.yaml).")
print(f"These {len(testable)} CATEGORY comparisons are a separate family — Holm")
print("correction applied ACROSS them, per T7's design.")

output = {
    "n_categories_tested": len(testable),
    "n_categories_untestable": len(untestable),
    "untestable_categories": [r["category"] for r in untestable],
    "correction_method": "holm",
    "correction_scope": "category-level tests only, NOT the overall T2 comparison",
    "category_results": results_by_category,
    "categories_surviving_correction": surviving["category"].tolist() if len(surviving) > 0 else [],
}

with open("../config/phase3_category_results.yaml", "w") as f:
    yaml.dump(output, f, sort_keys=False, default_flow_style=False)

print("\nSaved full T6-T10 results to ../config/phase3_category_results.yaml")