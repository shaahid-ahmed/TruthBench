"""Repeated sampling variance check for the secondary metric"""

import os
import re
import csv
import yaml
import numpy as np
import pandas as pd

with open("../config/models.yaml") as f:
    model_specs = yaml.safe_load(f)
with open("../config/mc1_scoring.yaml") as f:
    mc1_cfg = yaml.safe_load(f)
with open("../config/secondary_metric_generation.yaml") as f:
    gen_cfg = yaml.safe_load(f)

SYSTEM_INSTRUCTION = mc1_cfg["system_instruction"]
PROMPT_TEMPLATE = gen_cfg["prompt"]["template"]
USE_CHAT_TEMPLATE = gen_cfg["prompt"]["use_chat_template"]
TEMPERATURE = gen_cfg["generation"]["temperature"]
MAX_NEW_TOKENS = gen_cfg["generation"]["max_new_tokens"]
N_REPEATS = gen_cfg["sampling"]["n_repeats"]
N_QUESTIONS = gen_cfg["sampling"]["variance_check_n_questions"]
SEED = gen_cfg["sampling"]["variance_check_seed"]

FLAGGED_PATH = "../data/full_secondary_results_flagged.csv"
SCORED_PATH = "../data/full_secondary_scored.csv"
GEN_PATH = "../data/secondary_variance_generations.csv"
OUT_PATH = "../data/secondary_variance_scored.csv"
FAIL_PATH = "../data/secondary_variance_failures.csv"
SUMMARY_PATH = "../config/secondary_variance_results.yaml"

# Same stop handling as the full generation run
LEAK_PATTERNS = [r"\n\s*user\s*\n", r"<\|im_start\|>", r"<\|im_end\|>", r"\[INST\]", r"\[/INST\]"]
LEAK_REGEX = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)
# Reproduces the saved garbling flag exactly
CJK_REGEX = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯]")

GEN_FIELDS = ["model", "question", "category", "repeat", "generated_answer",
              "was_truncated_for_leak", "has_cjk_garbling", "is_garbled",
              "best_answer", "incorrect_answers"]


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
    return text[: match.start()].strip() if match else text.strip()


def append_csv(path, row, fields):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d["model"], d["question"], d["repeat"]))


def select_subset():
    """Seeded subset of questions scored for both models"""
    flagged = pd.read_csv(FLAGGED_PATH)
    scored = pd.read_csv(SCORED_PATH)
    per_q = scored.groupby("question")["model"].nunique()
    pool = sorted(per_q[per_q == len(model_specs)].index)
    rng = np.random.default_rng(SEED)
    chosen = set(rng.choice(pool, size=N_QUESTIONS, replace=False))
    return flagged[flagged["question"].isin(chosen)], scored[scored["question"].isin(chosen)]


def phase_generate(subset):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=mc1_cfg["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=mc1_cfg["quantization"]["bnb_4bit_quant_type"],
    )
    done = done_keys(GEN_PATH)
    questions = subset.drop_duplicates("question").reset_index(drop=True)

    for name, spec in model_specs.items():
        todo = [(i, r, rep) for i, r in questions.iterrows() for rep in range(1, N_REPEATS)
                if (name, r["question"], rep) not in done]
        if not todo:
            print(f"[A] {name}: all {len(questions) * (N_REPEATS - 1)} generations already done")
            continue
        print(f"[A] {name}: generating {len(todo)} samples")
        model = AutoModelForCausalLM.from_pretrained(
            spec["repo"], revision=spec["revision"], quantization_config=bnb_config, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
        eos_ids = get_eos_token_ids(tokenizer)

        for k, (i, r, rep) in enumerate(todo, start=1):
            user_content = PROMPT_TEMPLATE.format(question=r["question"])
            if USE_CHAT_TEMPLATE:
                messages = [{"role": "system", "content": SYSTEM_INSTRUCTION},
                            {"role": "user", "content": user_content}]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = user_content
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            torch.manual_seed(SEED * 1_000_003 + rep * 10_007 + i)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=TEMPERATURE > 0,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                    eos_token_id=eos_ids or None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            clean = truncate_leaked_turns(raw)
            leak = clean != raw
            cjk = bool(CJK_REGEX.search(clean))
            append_csv(GEN_PATH, {
                "model": name, "question": r["question"], "category": r["category"], "repeat": rep,
                "generated_answer": clean, "was_truncated_for_leak": leak,
                "has_cjk_garbling": cjk, "is_garbled": leak or cjk,
                "best_answer": r["best_answer"], "incorrect_answers": r["incorrect_answers"],
            }, GEN_FIELDS)
            if k % 10 == 0 or k == len(todo):
                print(f"  [{name}] {k}/{len(todo)}")

        del model, tokenizer
        if device == "mps":
            torch.mps.empty_cache()


def phase_judge():
    from deepeval.test_case import LLMTestCase
    from score_full_secondary_deepeval import LocalJudgeLLM, JUDGE_REPO, build_truthfulness_metric

    gens = pd.read_csv(GEN_PATH)
    done = done_keys(OUT_PATH) | done_keys(FAIL_PATH)
    todo = gens[[(m, q, r) not in done for m, q, r in zip(gens["model"], gens["question"], gens["repeat"])]]
    if todo.empty:
        print("[B] all generations already judged (or logged as failed)")
        return
    print(f"[B] judging {len(todo)} generations with {JUDGE_REPO}")
    judge = LocalJudgeLLM(JUDGE_REPO)
    metric = build_truthfulness_metric(judge)
    base = {c: None for c in ["model", "question", "category", "repeat", "is_garbled"]}

    for k, (_, row) in enumerate(todo.iterrows(), start=1):
        key = {c: row[c] for c in base}
        try:
            metric.measure(LLMTestCase(
                input=row["question"], actual_output=str(row["generated_answer"]),
                expected_output=row["best_answer"], context=[str(row["incorrect_answers"])],
            ))
        except Exception as e:
            append_csv(FAIL_PATH, {**key, "error_type": type(e).__name__, "error_message": str(e)[:300]},
                       list(base) + ["error_type", "error_message"])
            print(f"  [{k}/{len(todo)}] JUDGE FAILED ({type(e).__name__})")
            continue
        append_csv(OUT_PATH, {**key, "geval_score": metric.score}, list(base) + ["geval_score"])
        if k % 10 == 0 or k == len(todo):
            print(f"  [{k}/{len(todo)}] judged")


def phase_summarise(repeat0):
    new = pd.read_csv(OUT_PATH)
    r0 = repeat0[["model", "question", "category", "is_garbled", "geval_score"]].assign(repeat=0)
    allr = pd.concat([r0, new[r0.columns]], ignore_index=True)

    # Keep cells with every repeat scored
    n_per_cell = allr.groupby(["model", "question"])["repeat"].nunique()
    complete = n_per_cell[n_per_cell == N_REPEATS].index
    allr = allr.set_index(["model", "question"]).loc[complete].reset_index()

    summary = {
        "description": "Repeated-sampling variance of the secondary GEval score "
                       "(descriptive only; no significance testing).",
        "n_repeats": N_REPEATS, "n_questions_sampled": N_QUESTIONS, "seed": SEED,
        "repeat_0": "the full-run sample; repeats 1+ regenerated by this script",
        "models": {},
    }
    print("\n[C] Repeated-sampling variance (secondary metric, descriptive only)")
    for m, g in allr.groupby("model"):
        per_q_sd = g.groupby("question")["geval_score"].std(ddof=1)
        per_rep_mean = g.groupby("repeat")["geval_score"].mean()
        per_rep_garbled = g.groupby("repeat")["is_garbled"].mean()
        summary["models"][m] = {
            "n_questions_complete": int(g["question"].nunique()),
            "mean_score_all_repeats": float(g["geval_score"].mean()),
            "mean_within_question_sd": float(per_q_sd.mean()),
            "share_questions_score_changed": float((per_q_sd > 0).mean()),
            "mean_score_by_repeat": {int(k): float(v) for k, v in per_rep_mean.items()},
            "range_of_repeat_means": float(per_rep_mean.max() - per_rep_mean.min()),
            "garbled_rate_by_repeat": {int(k): float(v) for k, v in per_rep_garbled.items()},
        }
        s = summary["models"][m]
        print(f"  {m:<8} questions={s['n_questions_complete']}  mean={s['mean_score_all_repeats']:.3f}  "
              f"within-question SD={s['mean_within_question_sd']:.3f}  "
              f"score changed on {s['share_questions_score_changed']:.0%} of questions  "
              f"repeat means={[round(v, 3) for v in per_rep_mean]}  "
              f"garbled by repeat={[round(v, 3) for v in per_rep_garbled]}")

    means = {m: s["mean_score_by_repeat"] for m, s in summary["models"].items()}
    if len(means) == 2:
        a, b = list(means)
        gaps = {r: means[a][r] - means[b][r] for r in means[a] if r in means[b]}
        summary["model_gap_by_repeat"] = {"direction": f"{a} - {b}", **{int(k): float(v) for k, v in gaps.items()}}
        print(f"  gap ({a} - {b}) by repeat: {[round(v, 3) for v in gaps.values()]}")

    with open(SUMMARY_PATH, "w") as f:
        yaml.dump(summary, f, sort_keys=False, default_flow_style=False)
    print(f"\nSaved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    subset, repeat0 = select_subset()
    print(f"S6 subset: {subset['question'].nunique()} questions (seed {SEED}), "
          f"{N_REPEATS} repeats per model (repeat 0 = full run)")
    phase_generate(subset)
    phase_judge()
    phase_summarise(repeat0)
