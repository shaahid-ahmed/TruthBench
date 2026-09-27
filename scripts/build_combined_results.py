"""Combined results table with one row per model and question"""

import yaml
import pandas as pd

OUTPUT_PATH = "../data/combined_results.csv"

with open("../config/dataset.yaml") as f:
    usable = set(yaml.safe_load(f)["usable_categories"])

mc1 = pd.read_csv("../data/full_mc1_results_combined.csv")
flagged = pd.read_csv("../data/full_secondary_results_flagged.csv")
scored = pd.read_csv("../data/full_secondary_scored.csv")
# Strip whitespace so questions join across pipelines
for d in (flagged, scored):
    d["question"] = d["question"].str.strip()

sec = flagged[["model", "question", "category", "was_truncated_for_leak",
               "has_cjk_garbling", "is_garbled"]].merge(
    scored[["model", "question", "geval_score"]], on=["model", "question"], how="left")
sec = sec.rename(columns={
    "category": "secondary_category",
    "was_truncated_for_leak": "secondary_was_truncated_for_leak",
    "has_cjk_garbling": "secondary_has_cjk_garbling",
    "is_garbled": "secondary_is_garbled",
    "geval_score": "secondary_geval_score",
})
sec["secondary_scored"] = sec["secondary_geval_score"].notna()

out = mc1.rename(columns={"correct": "mc1_correct"}).merge(
    sec, on=["model", "question"], how="outer", validate="one_to_one", indicator=True)
assert (out["_merge"] == "both").all(), "MC1 and secondary pipelines cover different (model, question) pairs"
assert (out["category"] == out["secondary_category"]).all(), "category mismatch between pipelines"
out = out.drop(columns=["_merge", "secondary_category"])
out.insert(3, "category_usable_e9", out["category"].isin(usable))

out = out.sort_values(["category", "question", "model"]).reset_index(drop=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(out)} rows ({out['model'].nunique()} models x {out['question'].nunique()} questions) "
      f"to {OUTPUT_PATH}")
print(f"  category-usable (E9) rows: {int(out['category_usable_e9'].sum())}")
print(f"  secondary scored: {int(out['secondary_scored'].sum())}, unscored: {int((~out['secondary_scored']).sum())}")
print(f"  columns: {list(out.columns)}")
