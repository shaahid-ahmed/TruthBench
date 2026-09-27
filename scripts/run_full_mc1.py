import os
import yaml
import torch
import pandas as pd
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from join_category import check_no_duplicate_questions, attach_category

with open("../config/models.yaml") as f:
    model_specs = yaml.safe_load(f)

with open("../config/mc1_scoring.yaml") as f:
    mc1_cfg = yaml.safe_load(f)

device = "mps" if torch.backends.mps.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=mc1_cfg["quantization"]["load_in_4bit"],
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type=mc1_cfg["quantization"]["bnb_4bit_quant_type"],
)

df = check_no_duplicate_questions()

for name, spec in model_specs.items():
    output_path = f"../data/full_mc1_results_{name}.csv"

    if os.path.exists(output_path):
        print(f"{name}: already completed, skipping")
        continue

    print(f"\n FULL MC1 run: {name} (no limit, all questions)")

    hf_model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], quantization_config=bnb_config, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    model = HFLM(pretrained=hf_model, tokenizer=tokenizer)

    try:
        results = lm_eval.simple_evaluate(
            model=model, tasks=["truthfulqa_mc1"], log_samples=True,
            apply_chat_template=mc1_cfg["apply_chat_template"],
            system_instruction=mc1_cfg["system_instruction"],
        )
    except Exception as e:
        print(f"\n!!! {name} run FAILED partway through: {e}")
        raise

    samples = results["samples"]["truthfulqa_mc1"]
    attach_category(samples, df, strict=False)  # Tolerate unmatched categories

    acc = results["results"]["truthfulqa_mc1"].get("acc,none")
    print(f"\n{name} FULL MC1 accuracy ({len(samples)} questions): {acc}")

    rows = {"model": [], "question": [], "category": [], "correct": []}
    for s in samples:
        rows["model"].append(name)
        rows["question"].append(s["doc"]["question"])
        rows["category"].append(s["category"])
        rows["correct"].append(bool(s.get("acc")))

    model_df = pd.DataFrame(rows)
    n_missing_category = model_df["category"].isna().sum()
    if n_missing_category > 0:
        print(f"NOTE: {n_missing_category} row(s) saved with missing category, "
              f"fix manually in {output_path} before running category-level (T6) analysis.")

    model_df.to_csv(output_path, index=False)
    print(f"Saved {name}'s full results to {output_path}")

    del hf_model, tokenizer, model
    if device == "mps":
        torch.mps.empty_cache()

all_model_names = list(model_specs.keys())
all_files_exist = all(os.path.exists(f"../data/full_mc1_results_{name}.csv") for name in all_model_names)

if all_files_exist:
    combined = pd.concat(
        [pd.read_csv(f"../data/full_mc1_results_{name}.csv") for name in all_model_names], ignore_index=True
    )
    combined.to_csv("../data/full_mc1_results_combined.csv", index=False)
    print(f"\nAll models complete. Combined results saved ({len(combined)} rows).")
else:
    missing = [n for n in all_model_names if not os.path.exists(f"../data/full_mc1_results_{n}.csv")]
    print(f"\nNot all models complete: missing: {missing}. Re-run to continue.")