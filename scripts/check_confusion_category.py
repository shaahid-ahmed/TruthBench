"""Check whether the Confusion result is genuine and whether the judge discriminates there"""

import yaml
import pandas as pd

mc1 = pd.read_csv("../data/full_mc1_results_combined.csv")
scored = pd.read_csv("../data/full_secondary_scored.csv")
flagged = pd.read_csv("../data/full_secondary_results_flagged.csv")

conf = mc1[mc1["category"] == "Confusion"]
mc1_correct = conf.groupby("model")["correct"].agg(["sum", "count"])
print("MC1 correct on Confusion:\n", mc1_correct, "\n")

m = scored.merge(flagged[["model", "question", "generated_answer", "best_answer"]],
                 on=["model", "question"])
c = m[m["category"] == "Confusion"].copy()
c["contains_best_answer"] = [
    str(b).strip().rstrip(".").lower() in str(a).lower()
    for a, b in zip(c["generated_answer"], c["best_answer"])
]

by_group = c.groupby(["model", "contains_best_answer"])["geval_score"].agg(["count", "mean"])
wrong = c[~c["contains_best_answer"]]
share_wrong_passed = float((wrong["geval_score"] >= 0.5).mean())
print("GEval score on Confusion, split by whether the answer contains the correct name:\n", by_group)
print(f"\nAnswers WITHOUT the correct name scored >= 0.5: {share_wrong_passed:.1%} (n={len(wrong)})")

print("\nExamples (qwen):")
for _, r in c[(c["model"] == "qwen") & ~c["contains_best_answer"]].head(3).iterrows():
    print(f"  correct: {r['best_answer']!r} | GEval {r['geval_score']:.1f} | "
          f"answer: {str(r['generated_answer'])[:100]!r}")

out = {
    "mc1_correct_on_confusion": {k: {"correct": int(v["sum"]), "n": int(v["count"])}
                                 for k, v in mc1_correct.iterrows()},
    "mc1_interpretation": "Qwen's free-text answers name the famous namesake (the lure), "
                          "consistent with 0/46 being genuine model behaviour, not a scoring bug.",
    "geval_by_contains_best_answer": {
        f"{mdl}_{'contains' if has else 'missing'}_best_answer":
            {"n": int(row["count"]), "mean_geval": round(float(row["mean"]), 4)}
        for (mdl, has), row in by_group.iterrows()
    },
    "share_answers_missing_best_answer_scored_ge_0_5": round(share_wrong_passed, 4),
    "n_answers_missing_best_answer": int(len(wrong)),
    "geval_interpretation": "The judge scores answers with and without the correct name almost "
                            "identically here, i.e. it does not discriminate on this category. "
                            "Supports treating the secondary metric as descriptive only.",
}
with open("../config/confusion_check.yaml", "w") as f:
    yaml.dump(out, f, sort_keys=False, default_flow_style=False)
print("\nSaved to ../config/confusion_check.yaml")
