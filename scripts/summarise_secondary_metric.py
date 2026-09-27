"""Descriptive secondary metric means per model with no tests"""

import yaml
import pandas as pd

FLAGGED_PATH = "../data/full_secondary_results_flagged.csv"
SCORED_PATH = "../data/full_secondary_scored.csv"
OUTPUT_PATH = "../config/secondary_descriptive_results.yaml"

flagged = pd.read_csv(FLAGGED_PATH)
scored = pd.read_csv(SCORED_PATH)
assert not scored.duplicated(["model", "question"]).any()

key = lambda d: set(zip(d["model"], d["question"]))
unscored = sorted(key(flagged) - key(scored))


def describe(s):
    return {
        "n": int(s.count()),
        "mean": round(float(s.mean()), 4) if s.count() else None,
        "median": round(float(s.median()), 4) if s.count() else None,
        "sd": round(float(s.std(ddof=1)), 4) if s.count() > 1 else None,
    }


out = {
    "label": "DESCRIPTIVE / EXPLORATORY ONLY: no test, no correction, no inferential claim",
    "metric": "DeepEval GEval 'Truthfulness', judge microsoft/Phi-3-mini-4k-instruct, score in [0, 1]",
    "n_generated": int(len(flagged)),
    "n_scored": int(len(scored)),
    "n_unscored_judge_failed": len(unscored),
    "unscored_rows": [{"model": m, "question": q} for m, q in unscored],
    "by_model": {},
}

print("T11: secondary metric, DESCRIPTIVE ONLY (no test, no correction)\n")
print(f"{len(scored)} of {len(flagged)} generations scored; "
      f"{len(unscored)} unscored (judge failed, not imputed)\n")
print(f"{'model':<8} {'subset':<18} {'n':>4} {'mean':>7} {'median':>7} {'sd':>7}")
for model, g in scored.groupby("model"):
    n_gen = int((flagged["model"] == model).sum())
    n_garbled_gen = int(flagged.loc[flagged["model"] == model, "is_garbled"].sum())
    block = {
        "garbled_rate_all_generations": round(n_garbled_gen / n_gen, 4),
        "n_unscored": sum(1 for m, _ in unscored if m == model),
        "all_scored_rows": describe(g["geval_score"]),
        "excluding_garbled": describe(g.loc[~g["is_garbled"], "geval_score"]),
        "garbled_only": describe(g.loc[g["is_garbled"], "geval_score"]),
    }
    out["by_model"][model] = block
    for subset in ["all_scored_rows", "excluding_garbled", "garbled_only"]:
        d = block[subset]
        fmt = lambda x: "-" if x is None else f"{x:.3f}"
        print(f"{model:<8} {subset:<18} {d['n']:>4} {fmt(d['mean']):>7} {fmt(d['median']):>7} {fmt(d['sd']):>7}")
    print(f"{model:<8} garbled rate: {block['garbled_rate_all_generations']:.1%} of {n_gen} generations\n")

with open(OUTPUT_PATH, "w") as f:
    yaml.dump(out, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
print(f"Saved to {OUTPUT_PATH}")
