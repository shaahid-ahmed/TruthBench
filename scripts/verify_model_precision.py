import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

with open("../config/models.yaml", "r") as f:
    models = yaml.safe_load(f)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

device = "mps" if torch.backends.mps.is_available() else "cpu"
results = {}

# Same quantization config applied to BOTH models — this is the fairness check
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

for name, spec in models.items():
    repo = spec["repo"]
    revision = spec["revision"]

    print(f"Verifying model: {name}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, quantization_config=bnb_config, device_map=device
        )
    except Exception as e:
        print(f"  FAILED to load {name} in 4-bit: {e}")
        results[name] = {"repo": repo, "revision": revision, "load_error": str(e)}
        continue

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)

    model_dtype = next(model.parameters()).dtype
    quant_cfg = getattr(model.config, "quantization_config", None)
    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

    # Quick real generation test, not just a load test — confirms it actually runs, not just loads
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    generated_text = tokenizer.decode(out[0], skip_special_tokens=True)

    results[name] = {
        "repo": repo,
        "revision": revision,
        "actual_dtype": str(model_dtype),
        "quantization_config": str(quant_cfg),
        "parameter_memory_gb": round(mem_gb, 2),
        "generation_test": generated_text,
    }

    print(f"  dtype: {model_dtype}, memory: {mem_gb:.2f} GB, generated: {generated_text!r}")

    del model, tokenizer
    if device == "mps":
        torch.mps.empty_cache()

# --- Cross-model comparison (only if both loaded successfully) ---
names = list(results.keys())
a, b = names[0], names[1]

if "load_error" in results[a] or "load_error" in results[b]:
    print("\nOne or both models FAILED to load in 4-bit — bitsandbytes likely unsupported on this device.")
    quant_match = precision_match = False
else:
    precision_match = results[a]["actual_dtype"] == results[b]["actual_dtype"]
    quant_match = results[a]["quantization_config"] == results[b]["quantization_config"]

comparison = {
    "precision_match": precision_match,
    "quantization_match": quant_match,
    "model_a": a,
    "model_b": b,
}

output = {"models": results, "comparison": comparison}

with open("model_quantization_check.yaml", "w") as f:
    yaml.dump(output, f, sort_keys=False, default_flow_style=False)

print(f"\nSaved results to model_quantization_check.yaml")