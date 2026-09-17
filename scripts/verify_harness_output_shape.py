import json
import yaml
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

SYSTEM_INSTRUCTION = "You are a helpful assistant."

LIMIT = 10
all_samples = {}

for name, spec in model_specs.items():
    print(f"\n=== Running structural check on: {name} ===")

    hf_model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], quantization_config=bnb_config, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    model = HFLM(pretrained=hf_model, tokenizer=tokenizer)

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=["truthfulqa_mc1"],
        limit=LIMIT,
        log_samples=True,
        apply_chat_template=mc1_cfg["apply_chat_template"],
        system_instruction=SYSTEM_INSTRUCTION,
    )

    samples = results["samples"]["truthfulqa_mc1"]
    all_samples[name] = samples

    print(f"Got {len(samples)} samples for {name}.")
    print(f"\n--- {name}: doc structure for sample 0 ---")
    print(json.dumps(samples[0]["doc"], indent=2, default=str))

    print(f"\n--- {name}: raw context fed to model (arguments[0][0]) for sample 0 ---")
    try:
        print(samples[0]["arguments"][0][0])
    except (KeyError, IndexError, TypeError) as e:
        print(f"Could not extract arguments[0][0]: {e}")
        print(f"Raw 'arguments' field: {samples[0].get('arguments')}")

    del hf_model, tokenizer, model
    if device == "mps":
        torch.mps.empty_cache()

print("\n\n=== Structural checks (field presence, same across models) ===")

sample0 = next(iter(all_samples.values()))[0]
top_level_keys = list(sample0.keys())
print(f"Top-level keys in a sample: {top_level_keys}")

doc_id_candidates = [k for k in top_level_keys if "id" in k.lower() or k == "doc"]
print(f"Possible ID fields: {doc_id_candidates}")

metric_candidates = [k for k in top_level_keys if k in
                      ("acc", "acc,none", "exact_match", "exact_match,none")]
print(f"Possible correctness/metric fields: {metric_candidates}")

category_present = False
category_location = None
if "category" in top_level_keys:
    category_present = True
    category_location = "top-level"
elif "doc" in sample0 and isinstance(sample0["doc"], dict) and "category" in sample0["doc"]:
    category_present = True
    category_location = "inside 'doc'"

print(f"Category label present: {category_present} (location: {category_location})")

question_present = "question" in str(sample0)
print(f"Question text findable somewhere in sample: {question_present}")

print("\n=== Cross-model context comparison (same question, both models) ===")
model_names = list(all_samples.keys())
if len(model_names) >= 2:
    a_name, b_name = model_names[0], model_names[1]
    try:
        ctx_a = all_samples[a_name][0]["arguments"][0][0]
        ctx_b = all_samples[b_name][0]["arguments"][0][0]
        print(f"{a_name} context (first 300 chars):\n{ctx_a[:300]}\n")
        print(f"{b_name} context (first 300 chars):\n{ctx_b[:300]}\n")
    except (KeyError, IndexError, TypeError) as e:
        print(f"Could not compare contexts: {e}")

verdict = {
    "system_instruction_used": SYSTEM_INSTRUCTION,
    "top_level_keys": top_level_keys,
    "id_field_candidates": doc_id_candidates,
    "metric_field_candidates": metric_candidates,
    "category_present": category_present,
    "category_location": category_location,
    "question_text_present": question_present,
    "recommended_join_strategy": (
        "Use lm-eval's own doc index/id if present and stable across runs. "
        "If not, fall back to joining on exact question text against "
        "data/truthfulqa_generation.csv — verify no duplicate question "
        "text exists in the dataset first, or this join will silently "
        "produce wrong matches."
    ),
}

with open("harness_output_shape_check.yaml", "w") as f:
    yaml.dump(verdict, f, sort_keys=False, default_flow_style=False)

print("\nSaved verdict to harness_output_shape_check.yaml")