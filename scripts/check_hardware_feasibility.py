import yaml
import gc
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

with open("../config/models.yaml", "r") as f:
    models = yaml.safe_load(f)

device = "mps" if torch.backends.mps.is_available() else "cpu"

def report_ram(label):
    vm = psutil.virtual_memory()
    print(f"  [{label}] RAM used: {vm.used/1e9:.2f} GB / {vm.total/1e9:.2f} GB ({vm.percent}%)")
    return vm.used / 1e9

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

# Pick one model to stress-test repeatedly — qwen used here, swap if you want mistral instead
name = "qwen"
spec = models[name]
N_CYCLES = 6

print(f"=== Repeated load/unload test: {name}, {N_CYCLES} cycles ===\n")

baseline = report_ram("startup baseline")
cycle_usage = []

for i in range(1, N_CYCLES + 1):
    print(f"\n--- Cycle {i} ---")
    before = report_ram("before load")

    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], quantization_config=bnb_config, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])

    after_load = report_ram("after load")

    inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    after_gen = report_ram("after generation")

    del model, tokenizer, inputs, out
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    after_unload = report_ram("after unload + gc.collect()")

    cycle_usage.append({
        "cycle": i,
        "before_gb": round(before, 2),
        "after_load_gb": round(after_load, 2),
        "after_gen_gb": round(after_gen, 2),
        "after_unload_gb": round(after_unload, 2),
    })

# --- Summary ---
print("\n=== Summary across cycles ===")
for c in cycle_usage:
    print(f"  cycle {c['cycle']}: before={c['before_gb']}GB  after_unload={c['after_unload_gb']}GB")

first_unload = cycle_usage[0]["after_unload_gb"]
last_unload = cycle_usage[-1]["after_unload_gb"]
drift = last_unload - first_unload

print(f"\nBaseline (startup): {baseline:.2f} GB")
print(f"After-unload, cycle 1: {first_unload:.2f} GB")
print(f"After-unload, cycle {N_CYCLES}: {last_unload:.2f} GB")
print(f"Drift across {N_CYCLES} cycles: {drift:+.2f} GB")

if drift > 1.0:
    print("Memory is climbing across cycles — likely a leak. Investigate before running full Phase 2 batch.")
else:
    print("Memory appears stable across repeated load/unload cycles.")

with open("memory_stability_check.yaml", "w") as f:
    yaml.dump({
        "model": name,
        "n_cycles": N_CYCLES,
        "startup_baseline_gb": round(baseline, 2),
        "cycles": cycle_usage,
        "drift_gb": round(drift, 2),
    }, f, sort_keys=False, default_flow_style=False)

print("\nSaved results to memory_stability_check.yaml")