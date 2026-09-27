"""Select a stratified pilot sample from the usable pool"""

import pandas as pd
import yaml

PILOT_SIZE = 15
N_CATEGORIES = 3
RANDOM_SEED = 42  # Fixed for reproducibility

df = pd.read_csv("../data/truthfulqa_generation.csv")

with open("../config/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

usable_categories = dataset_cfg["usable_categories"]

# Use the largest usable categories
usable_df = df[df["category_merged"].isin(usable_categories)]
top_categories = (
    usable_df["category_merged"].value_counts().head(N_CATEGORIES).index.tolist()
)

print(f"Pilot categories selected: {top_categories}")

per_category = max(1, PILOT_SIZE // N_CATEGORIES)
pilot_rows = []
for cat in top_categories:
    subset = usable_df[usable_df["category_merged"] == cat]
    sample = subset.sample(n=min(per_category, len(subset)), random_state=RANDOM_SEED)
    pilot_rows.append(sample)

pilot_df = pd.concat(pilot_rows).reset_index(drop=True)

print(f"\nPilot sample: {len(pilot_df)} questions across {pilot_df['category_merged'].nunique()} categories")
print(pilot_df["category_merged"].value_counts())

pilot_df.to_csv("../data/pilot_sample.csv", index=False)
print("\nSaved to ../data/pilot_sample.csv")