"""
Smoke test: compare MC1 scoring with and without chat templating,
for both Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3.

Loads each model directly via transformers with a real BitsAndBytesConfig,
then passes the already-instantiated model object into HFLM (bypassing
lm_eval's internal _create_model(), which has a bug forwarding
quantization_config twice when passed as a string-based `pretrained`).
"""

import yaml
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

with open("../config/models.yaml", "r") as f:
    model_specs = yaml.safe_load(f)

device = "mps" if torch.backends.mps.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

LIMIT = 10  # smoke test only — never use for real results
all_results = {}

for model_name, spec in model_specs.items():
    all_results[model_name] = {}

    for use_template in [False, True]:
        label = "with_template" if use_template else "no_template"
        print(f"\n=== {model_name} | {label} ===")

        # Load the quantized model + tokenizer ourselves
        hf_model = AutoModelForCausalLM.from_pretrained(
            spec["repo"],
            revision=spec["revision"],
            quantization_config=bnb_config,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])

        # Hand the already-loaded model + tokenizer to HFLM
        model = HFLM(pretrained=hf_model, tokenizer=tokenizer)

        results = lm_eval.simple_evaluate(
            model=model,
            tasks=["truthfulqa_mc1"],
            limit=LIMIT,
            log_samples=True,
            apply_chat_template=use_template,
        )

        acc = results["results"]["truthfulqa_mc1"].get("acc,none")
        print(f"  MC1 accuracy on {LIMIT} questions: {acc}")

        samples = results.get("samples", {}).get("truthfulqa_mc1", [])
        per_question = []
        for s in samples:
            per_question.append({
                "question": s.get("doc", {}).get("question", "UNKNOWN"),
                "correct": bool(s.get("acc", s.get("exact_match", None))),
            })

        all_results[model_name][label] = {
            "accuracy": acc,
            "per_question": per_question,
        }

        del hf_model, tokenizer, model
        if device == "mps":
            torch.mps.empty_cache()

# --- Compare per-question divergence within each model ---
print("\n=== Divergence check ===")
divergence_summary = {}

for model_name in all_results:
    no_t = all_results[model_name]["no_template"]["per_question"]
    with_t = all_results[model_name]["with_template"]["per_question"]

    diverged = []
    for a, b in zip(no_t, with_t):
        if a["correct"] != b["correct"]:
            diverged.append(a["question"])

    divergence_summary[model_name] = {
        "n_questions": len(no_t),
        "n_diverged": len(diverged),
        "diverged_questions": diverged,
        "acc_no_template": all_results[model_name]["no_template"]["accuracy"],
        "acc_with_template": all_results[model_name]["with_template"]["accuracy"],
    }

    print(f"{model_name}: {len(diverged)}/{len(no_t)} diverged "
          f"(acc no-template={all_results[model_name]['no_template']['accuracy']}, "
          f"acc with-template={all_results[model_name]['with_template']['accuracy']})")

with open("chat_template_smoke_test.yaml", "w") as f:
    yaml.dump(
        {"full_results": all_results, "divergence_summary": divergence_summary},
        f, sort_keys=False, default_flow_style=False,
    )

print("\nSaved to chat_template_smoke_test.yaml")