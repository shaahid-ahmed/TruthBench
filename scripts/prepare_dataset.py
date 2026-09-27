import os
import yaml
from datasets import load_dataset

MIN_CATEGORY_SIZE = 20
FAMILY_MERGE_MAP = {
    "Indexical Error: Other": "Indexical Error",
    "Indexical Error: Time": "Indexical Error",
    "Indexical Error: Location": "Indexical Error",
    "Indexical Error: Identity": "Indexical Error",

    "Confusion: People": "Confusion",
    "Confusion: Places": "Confusion",
    "Confusion: Other": "Confusion",

    "Misconceptions: Topical": "Misconceptions",
}

os.makedirs("../data", exist_ok=True)

dataset = load_dataset("truthfulqa/truthful_qa", "generation")
df = dataset["validation"].to_pandas()

total_pool_size = len(df)
print(f"Total pool size: {total_pool_size} questions")

df["category_merged"] = df["category"].map(FAMILY_MERGE_MAP).fillna(df["category"])

category_counts = df["category_merged"].value_counts().sort_values(ascending=False)
print("\nCategory sizes (after family merging):")
print(category_counts)

small_categories = category_counts[category_counts < MIN_CATEGORY_SIZE].index.tolist()
usable_categories = category_counts[category_counts >= MIN_CATEGORY_SIZE].index.tolist()

usable_pool_size = df[df["category_merged"].isin(usable_categories)].shape[0]
excluded_pool_size = df[df["category_merged"].isin(small_categories)].shape[0]

print(f"\nCategories still too small (< {MIN_CATEGORY_SIZE}): {len(small_categories)}")
for cat in small_categories:
    print(f"  - {cat}: {category_counts[cat]}")

print(f"\nUsable categories: {len(usable_categories)} / {len(category_counts)} total")
print(f"Usable pool size (post-exclusion): {usable_pool_size} / {total_pool_size}")
print(f"Excluded pool size: {excluded_pool_size}")

df.to_csv("../data/truthfulqa_generation.csv", index=False)
print("\nSaved full dataset (with category_merged column) to ../data/truthfulqa_generation.csv")

dataset_config = {
    "source": "truthfulqa/truthful_qa",
    "config": "generation",
    "split": "validation",
    "total_pool_size": int(total_pool_size),
    "grouping": "family_merge_only",
    "family_merge_map": FAMILY_MERGE_MAP,
    "min_category_size_threshold": MIN_CATEGORY_SIZE,
    "category_sizes_merged": category_counts.to_dict(),
    "excluded_categories": small_categories,
    "usable_categories": usable_categories,
    "usable_pool_size": int(usable_pool_size),
    "excluded_pool_size": int(excluded_pool_size),
}

with open("../config/dataset.yaml", "w") as f:
    yaml.dump(dataset_config, f, sort_keys=False, default_flow_style=False)

print("Saved category breakdown to ../config/dataset.yaml")