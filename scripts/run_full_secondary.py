"""Full free text generation with per question checkpointing"""

import os
import re
import csv
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

LEAK_PATTERNS = [r"\n\s*user\s*\n", r"<\|im_start\|>", r"<\|im_end\|>", r"\[INST\]", r"\[/INST\]"]
LEAK_REGEX = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)

OUTPUT_PATH = "../data/full_secondary_results.csv"
FIELDNAMES = [
    "model", "question", "category", "generated_answer", "was_truncated_for_leak",
    "best_answer", "correct_answers", "incorrect_answers",
]


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


def load_completed_pairs():
    if not os.path.exists(OUTPUT_PATH):
        return set()
    existing = pd.read_csv(OUTPUT_PATH)
    return set(zip(existing["model"], existing["question"]))


def append_row(row_dict):
    file_exists = os.path.exists(OUTPUT_PATH)
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


full_df = pd.read_csv("../data/truthfulqa_generation.csv")
completed_pairs = load_completed_pairs()
print(f"Found {len(completed_pairs)} already-completed (model, question) pairs, will skip these.")

for name, spec in model_specs.items():
    print(f"\nFULL secondary generation: {name}")

    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], quantization_config=bnb_config, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    eos_ids = get_eos_token_ids(tokenizer)

    n_done_this_model = 0
    n_skipped_this_model = 0

    for idx, row in full_df.iterrows():
        question = row["question"]

        if (name, question) in completed_pairs:
            n_skipped_this_model += 1
            continue

        user_content = PROMPT_TEMPLATE.format(question=question)

        if USE_CHAT_TEMPLATE:
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content},
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = user_content

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        try:
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
        except Exception as e:
            print(f"  FAILED on question {idx} ({name}): {e}")
            clean_text = f"[GENERATION FAILED: {e}]"
            was_truncated = False

        append_row({
            "model": name,
            "question": question,
            "category": row["category_merged"],
            "generated_answer": clean_text,
            "was_truncated_for_leak": was_truncated,
            "best_answer": row["best_answer"],
            "correct_answers": row["correct_answers"],
            "incorrect_answers": row["incorrect_answers"],
        })

        n_done_this_model += 1
        if n_done_this_model % 20 == 0:
            print(f"  [{name}] {n_done_this_model} generated this run "
                  f"({n_skipped_this_model} skipped as already-done)")

    print(f"{name}: done. {n_done_this_model} generated this run, "
          f"{n_skipped_this_model} skipped (already completed on a prior run).")

    del model, tokenizer
    if device == "mps":
        torch.mps.empty_cache()

print(f"\nAll models complete. Full results at {OUTPUT_PATH}")
final_df = pd.read_csv(OUTPUT_PATH)
print(f"Total rows: {len(final_df)} (expected: {len(full_df) * len(model_specs)})")