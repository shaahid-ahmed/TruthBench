"""Pilot multiple choice run on both models"""

import json
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

PILOT_LIMIT = 20

df = check_no_duplicate_questions()

all_results = {}

for name, spec in model_specs.items():
    print(f"\n MC1 pilot run: {name}")

    hf_model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], quantization_config=bnb_config, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    model = HFLM(pretrained=hf_model, tokenizer=tokenizer)

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=["truthfulqa_mc1"],
        limit=PILOT_LIMIT,
        log_samples=True,
        apply_chat_template=mc1_cfg["apply_chat_template"],
        system_instruction=mc1_cfg["system_instruction"],
    )

    samples = results["samples"]["truthfulqa_mc1"]
    attach_category(samples, df)

    acc = results["results"]["truthfulqa_mc1"].get("acc,none")
    print(f"  {name} MC1 accuracy on pilot ({len(samples)} questions): {acc}")

    for s in samples:
        all_results.setdefault("model", []).append(name)
        all_results.setdefault("question", []).append(s["doc"]["question"])
        all_results.setdefault("category", []).append(s["category"])
        all_results.setdefault("correct", []).append(bool(s.get("acc")))

    del hf_model, tokenizer, model
    if device == "mps":
        torch.mps.empty_cache()

results_df = pd.DataFrame(all_results)
results_df.to_csv("../data/pilot_mc1_results.csv", index=False)

print("\nPilot MC1 summary")
print(results_df.groupby("model")["correct"].mean())
print("\nSaved per-question results to ../data/pilot_mc1_results.csv")