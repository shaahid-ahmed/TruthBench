"""Spot check the Confusion result and the tied categories"""

import pandas as pd

df = pd.read_csv("../data/full_mc1_results_combined.csv")

# List every Confusion question
confusion_pivot = df[df["category"] == "Confusion"].pivot(
    index="question", columns="model", values="correct"
)

print("Confusion category: per-question results\n")
print(confusion_pivot.to_string())

n_qwen_correct = confusion_pivot["qwen"].sum()
n_mistral_correct = confusion_pivot["mistral"].sum()
print(f"\nQwen correct: {n_qwen_correct}/{len(confusion_pivot)}")
print(f"Mistral correct: {n_mistral_correct}/{len(confusion_pivot)}")

# Check whether ties come from identical answers
for category in ["Misconceptions", "Economics"]:
    cat_pivot = df[df["category"] == category].pivot(
        index="question", columns="model", values="correct"
    )
    pct_identical = (cat_pivot["qwen"] == cat_pivot["mistral"]).mean()
    print(f"\n{category}: fraction of questions where both models "
          f"got the SAME outcome (both right or both wrong): {pct_identical:.2%} ===")
    if pct_identical < 1.0:
        print("(Not 100% identical per-question, the tied AGGREGATE accuracy is "
              "coincidental, from equal numbers of offsetting disagreements, "
              "not identical behavior.)")
    else:
        print("(100% identical per-question outcomes, genuinely matching behavior, "
              "not just a coincidental aggregate tie.)")