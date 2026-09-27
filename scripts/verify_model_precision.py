import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("../config/models.yaml", "r") as f:
    models = yaml.safe_load(f)

DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
device = "mps" if torch.backends.mps.is_available() else "cpu"
results = {}

for name, spec in models.items():
    repo = spec["repo"]
    revision = spec["revision"]
    dtype = DTYPE_MAP[spec["dtype"]]

    print(f"Verifying model: {name}")

    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, torch_dtype=dtype, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)

    model_dtype = next(model.parameters()).dtype
    precision_ok = model_dtype == dtype
    quant_cfg = getattr(model.config, "quantization_config", None)
    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

    results[name] = {
        "repo": repo,
        "revision": revision,
        "requested_dtype": str(dtype),
        "actual_dtype": str(model_dtype),
        "precision_ok": precision_ok,
        "quantization_config": quant_cfg,
        "parameter_memory_gb": round(mem_gb, 2),
    }

    print(f"  precision_ok: {precision_ok}, dtype: {model_dtype}, memory: {mem_gb:.2f} GB")

    del model
    if device == "mps":
        torch.mps.empty_cache()

names = list(results.keys())
a, b = names[0], names[1]

precision_match = results[a]["actual_dtype"] == results[b]["actual_dtype"]
quant_match = results[a]["quantization_config"] == results[b]["quantization_config"]

comparison = {
    "precision_match": precision_match,
    "quantization_match": quant_match,
    "model_a": a,
    "model_b": b,
}

output = {"models": results, "comparison": comparison}

with open("model_precision_check.yaml", "w") as f:
    yaml.dump(output, f, sort_keys=False, default_flow_style=False)

print(f"\nSaved verification results to model_precision_check.yaml")

assert precision_match, f"DTYPE MISMATCH between {a} and {b}"
assert quant_match, f"QUANTIZATION MISMATCH between {a} and {b}"
print("Both models confirmed at matching precision and quantization.")