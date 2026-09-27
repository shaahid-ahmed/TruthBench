import yaml
import pandas as pd
import numpy as np

mc1_df = pd.read_csv("../data/pilot_mc1_results.csv")
secondary_df = pd.read_csv("../data/pilot_secondary_scored.csv")

mc1_acc = mc1_df.groupby("model")["correct"].mean()
print("MC1 pilot accuracy by model:")
print(mc1_acc)

model_names = mc1_acc.index.tolist()
acc_a, acc_b = mc1_acc.iloc[0], mc1_acc.iloc[1]
mc1_raw_diff = abs(acc_a - acc_b)

pivot = mc1_df.pivot(index="question", columns="model", values="correct")
n_a_right_b_wrong = ((pivot[model_names[0]] == True) & (pivot[model_names[1]] == False)).sum()
n_b_right_a_wrong = ((pivot[model_names[1]] == True) & (pivot[model_names[0]] == False)).sum()
n_discordant = n_a_right_b_wrong + n_b_right_a_wrong

print(f"\nDiscordant pairs in pilot: {n_discordant} (out of {len(pivot)} questions)")
print("NOTE: this is far below the ~25 discordant-pair threshold needed "
      "for a reliable asymptotic McNemar's test (per T6's design), the "
      "pilot is not meant to produce a trustworthy p-value, only a rough "
      "effect-size starting point for power analysis.")

secondary_mean = secondary_df.groupby("model")["geval_score"].mean()
print("\nSecondary (DeepEval) pilot mean score by model:")
print(secondary_mean)
secondary_raw_diff = abs(secondary_mean.iloc[0] - secondary_mean.iloc[1])

pilot_effect_size_record = {
    "note": (
        "Pilot effect sizes at N=15-20 are NOISY estimates, not reliable "
        "findings. Used only as a rough input to S1's power analysis. "
        "Consider running S1's power calculation under both this estimate "
        "and a smaller/larger plausible effect size as a sensitivity check."
    ),
    "mc1_pilot": {
        "n_questions": int(len(pivot)),
        "accuracy_by_model": {k: float(v) for k, v in mc1_acc.items()},
        "raw_accuracy_difference": float(mc1_raw_diff),
        "n_discordant_pairs": int(n_discordant),
        "reliable_for_mcnemar": bool(n_discordant >= 25),
    },
    "secondary_metric_pilot": {
        "n_questions": int(len(secondary_df) // len(model_names)),
        "mean_score_by_model": {k: float(v) for k, v in secondary_mean.items()},
        "raw_score_difference": float(secondary_raw_diff),
        "caveat": "Descriptive/exploratory metric per T11, not used for power analysis, informational only.",
    },
}

with open("../config/pilot_effect_size.yaml", "w") as f:
    yaml.dump(pilot_effect_size_record, f, sort_keys=False, default_flow_style=False)

print("\nSaved to ../config/pilot_effect_size.yaml")
print("\nUse mc1_pilot.raw_accuracy_difference as the primary input to S1's "
      "power analysis. Given n_discordant_pairs is far below 25, treat this "
      "as a lower-confidence starting point and consider testing S1 across "
      "a range of plausible effect sizes rather than trusting this single "
      "number.")