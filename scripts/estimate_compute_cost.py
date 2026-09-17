import yaml

with open("../config/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

with open("../config/secondary_metric_generation.yaml") as f:
    secondary_cfg = yaml.safe_load(f)

USABLE_POOL_SIZE = dataset_cfg["usable_pool_size"]
TOTAL_POOL_SIZE = dataset_cfg["total_pool_size"]

AVG_MC1_CHOICES_PER_QUESTION = 6

observed = {
    "qwen":    {"requests": 60, "seconds": 85},
    "mistral": {"requests": 60, "seconds": 140},
}

print("MC1 pass time projection\n")

mc1_total_seconds = 0
for name, obs in observed.items():
    sec_per_request = obs["seconds"] / obs["requests"]
    total_requests = TOTAL_POOL_SIZE * AVG_MC1_CHOICES_PER_QUESTION
    projected_seconds = sec_per_request * total_requests
    mc1_total_seconds += projected_seconds

    print(f"{name}:")
    print(f"  observed: {sec_per_request:.3f} s/request")
    print(f"  projected requests for full pool ({TOTAL_POOL_SIZE} questions "
          f"x ~{AVG_MC1_CHOICES_PER_QUESTION} choices): {total_requests}")
    print(f"  projected time: {projected_seconds/60:.1f} minutes "
          f"({projected_seconds/3600:.2f} hours)\n")

print(f"Total MC1 pass (both models): {mc1_total_seconds/60:.1f} minutes "
      f"({mc1_total_seconds/3600:.2f} hours)")

ASSUMED_SECONDS_PER_GENERATION = 8

n_repeats = secondary_cfg["sampling"]["n_repeats"]
subset_size_for_repeats = 30

secondary_main_pass_seconds = TOTAL_POOL_SIZE * 2 * ASSUMED_SECONDS_PER_GENERATION
secondary_repeat_pass_seconds = subset_size_for_repeats * 2 * n_repeats * ASSUMED_SECONDS_PER_GENERATION

secondary_total_seconds = secondary_main_pass_seconds + secondary_repeat_pass_seconds

print(f"\nSecondary free-text pass time projection (ESTIMATE — unmeasured)")
print(f"  Assumed {ASSUMED_SECONDS_PER_GENERATION}s/generation (placeholder, refine after P2b)")
print(f"  Main pass (both models, full pool): {secondary_main_pass_seconds/60:.1f} minutes")
print(f"  Repeated-sampling check (S6, {subset_size_for_repeats} questions x "
      f"{n_repeats} repeats x both models): {secondary_repeat_pass_seconds/60:.1f} minutes")
print(f"  Total secondary pass: {secondary_total_seconds/60:.1f} minutes")

grand_total_seconds = mc1_total_seconds + secondary_total_seconds
print(f"\nGRAND TOTAL (rough)")
print(f"  {grand_total_seconds/3600:.2f} hours of raw compute time")
print(f"  NOTE: this does not include model load time (~15s/load observed), "
      f"pipeline overhead, retries for failed/missing responses (S5/S6 in "
      f"the ticket backlog), or DeepEval scoring time itself.")

estimate = {
    "based_on_observed_timing": observed,
    "usable_pool_size": USABLE_POOL_SIZE,
    "total_pool_size": TOTAL_POOL_SIZE,
    "mc1_projected_hours": round(mc1_total_seconds / 3600, 2),
    "secondary_projected_hours_ESTIMATE": round(secondary_total_seconds / 3600, 2),
    "grand_total_hours_rough": round(grand_total_seconds / 3600, 2),
    "caveats": [
        "Secondary pass timing is a placeholder, not measured — refine after P2b pilot",
        "Does not include model load/unload overhead, retries, or DeepEval scoring time",
        "MC1 choices-per-question assumed at 6 — verify actual average across full dataset",
    ],
}

with open("compute_cost_estimate.yaml", "w") as f:
    yaml.dump(estimate, f, sort_keys=False, default_flow_style=False)

print("\nSaved to compute_cost_estimate.yaml")