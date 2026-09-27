import re
import yaml
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

with open("../config/models.yaml") as f:
    model_specs = yaml.safe_load(f)

with open("../config/mc1_scoring.yaml") as f:
    mc1_cfg = yaml.safe_load(f)

with open("../config/secondary_metric_generation.yaml") as f:
    secondary_cfg = yaml.safe_load(f)

device = "mps" if torch.backends.mps.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=mc1_cfg["quantization"]["load_in_4bit"],
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type=mc1_cfg["quantization"]["bnb_4bit_quant_type"],
)

SYSTEM_INSTRUCTION = mc1_cfg["system_instruction"]
PROMPT_TEMPLATE = secondary_cfg["prompt"]["template"]
USE_CHAT_TEMPLATE = secondary_cfg["prompt"]["use_chat_template"]
TEMPERATURE = secondary_cfg["generation"]["temperature"]
MAX_NEW_TOKENS = secondary_cfg["generation"]["max_new_tokens"]

LEAK_PATTERNS = [
    r"\n\s*user\s*\n",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[/INST\]",
]
LEAK_REGEX = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)


def get_eos_token_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)

    for special_tok in ["<|im_end|>", "<|eot_id|>", "</s>"]:
        try:
            tok_id = tokenizer.convert_tokens_to_ids(special_tok)
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                ids.add(tok_id)
        except Exception:
            pass

    return list(ids)


def truncate_leaked_turns(text):
    match = LEAK_REGEX.search(text)
    if match:
        return text[: match.start()].strip()
    return text.strip()


pilot_df = pd.read_csv("../data/pilot_sample.csv")

all_results = {
    "model": [], "question": [], "category": [], "generated_answer": [],
    "was_truncated_for_leak": [],
    "best_answer": [], "correct_answers": [], "incorrect_answers": [],
}

for name, spec in model_specs.items():
    print(f"\nsecondary metric pilot generation: {name}")

    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], quantization_config=bnb_config, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])

    eos_ids = get_eos_token_ids(tokenizer)
    print(f"  Using eos_token_id(s): {eos_ids}")

    for _, row in pilot_df.iterrows():
        question = row["question"]
        user_content = PROMPT_TEMPLATE.format(question=question)

        if USE_CHAT_TEMPLATE:
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = user_content

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                eos_token_id=eos_ids if eos_ids else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_tokens = out[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        clean_text = truncate_leaked_turns(raw_text)
        was_truncated = clean_text != raw_text

        all_results["model"].append(name)
        all_results["question"].append(question)
        all_results["category"].append(row["category_merged"])
        all_results["generated_answer"].append(clean_text)
        all_results["was_truncated_for_leak"].append(was_truncated)
        all_results["best_answer"].append(row["best_answer"])
        all_results["correct_answers"].append(row["correct_answers"])
        all_results["incorrect_answers"].append(row["incorrect_answers"])

        flag = " [LEAK TRUNCATED]" if was_truncated else ""
        print(f"  Q: {question}")
        print(f"  A: {clean_text}{flag}\n")

    del model, tokenizer
    if device == "mps":
        torch.mps.empty_cache()

results_df = pd.DataFrame(all_results)
results_df.to_csv("../data/pilot_secondary_results.csv", index=False)

n_truncated = results_df["was_truncated_for_leak"].sum()
print(f"\nSaved {len(results_df)} rows to ../data/pilot_secondary_results.csv")
print(f"Rows requiring leak truncation: {n_truncated} / {len(results_df)}")
if n_truncated > 0:
    print("NOTE: if this count is still high after the eos_token_id fix, "
          "the leak is happening at the token-generation level, not just "
          "a decoding artifact, worth investigating generation_config "
          "defaults or trying full-precision generation as a diagnostic.")