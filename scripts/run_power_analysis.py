import yaml
from scipy.stats import norm
import pandas as pd

ALPHA = 0.05
POWER = 0.80

Z_ALPHA = norm.ppf(1 - ALPHA / 2)
Z_BETA = norm.ppf(POWER)


def required_discordant_pairs(p_split, alpha=ALPHA, power=POWER):
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    p0 = 0.5
    numerator = z_a * (p0 * (1 - p0)) ** 0.5 + z_b * (p_split * (1 - p_split)) ** 0.5
    denominator = abs(p_split - p0)
    n_disc = (numerator / denominator) ** 2
    return n_disc


with open("../config/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

with open("../config/pilot_effect_size.yaml") as f:
    pilot_cfg = yaml.safe_load(f)

TOTAL_POOL = dataset_cfg["total_pool_size"]
USABLE_POOL = dataset_cfg["usable_pool_size"]

p_discordant_values = [0.15, 0.20, 0.25]
p_split_values = [0.55, 0.60, 0.65, 0.70]

rows = []
for p_disc in p_discordant_values:
    for p_split in p_split_values:
        n_disc_needed = required_discordant_pairs(p_split)
        n_total_needed = n_disc_needed / p_disc

        rows.append({
            "p_discordant": p_disc,
            "p_split": p_split,
            "n_discordant_pairs_needed": int(round(n_disc_needed)),
            "n_total_questions_needed": int(round(n_total_needed)),
            "fits_in_total_pool_817": bool(n_total_needed <= TOTAL_POOL),
            "fits_in_usable_pool_605": bool(n_total_needed <= USABLE_POOL),
        })

sensitivity_df = pd.DataFrame(rows)
print("=== Power analysis sensitivity grid (alpha=0.05, power=0.80) ===\n")
print(sensitivity_df.to_string(index=False))

print(f"\nTotal pool available (overall comparison): {TOTAL_POOL}")
print(f"Usable pool available (category-level tests): {USABLE_POOL}")

n_fits_total = sensitivity_df["fits_in_total_pool_817"].sum()
n_fits_usable = sensitivity_df["fits_in_usable_pool_605"].sum()
print(f"\nScenarios where the full pool (817) is sufficient: {n_fits_total} / {len(sensitivity_df)}")
print(f"Scenarios where the usable pool (605) is sufficient: {n_fits_usable} / {len(sensitivity_df)}")

pilot_note = {
    "pilot_p_discordant_observed": round(
        pilot_cfg["mc1_pilot"]["n_discordant_pairs"] / pilot_cfg["mc1_pilot"]["n_questions"], 3
    ),
    "pilot_effect_size_observed": pilot_cfg["mc1_pilot"]["raw_accuracy_difference"],
    "reliable": pilot_cfg["mc1_pilot"]["reliable_for_mcnemar"],
    "used_in_power_analysis": False,
    "reason_not_used": "Only 4 discordant pairs at N=20, too few to trust as a point estimate.",
}

output = {
    "alpha": ALPHA,
    "power_target": POWER,
    "total_pool_size": TOTAL_POOL,
    "usable_pool_size": USABLE_POOL,
    "pilot_estimate_for_reference_only": pilot_note,
    "sensitivity_grid": rows,
}

with open("../config/power_analysis.yaml", "w") as f:
    yaml.dump(output, f, sort_keys=False, default_flow_style=False)

print("\nSaved full sensitivity grid to ../config/power_analysis.yaml")