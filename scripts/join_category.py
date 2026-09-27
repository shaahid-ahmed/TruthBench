import pandas as pd


def check_no_duplicate_questions(csv_path="../data/truthfulqa_generation.csv"):
    df = pd.read_csv(csv_path)
    n_dupes = df["question"].duplicated().sum()
    if n_dupes > 0:
        dupe_questions = df[df["question"].duplicated(keep=False)]["question"].unique()
        raise ValueError(
            f"{n_dupes} duplicate question(s) found. Duplicated: {list(dupe_questions)}"
        )
    print(f"No duplicate questions found across {len(df)} rows. Text-based join is safe.")
    return df


def attach_category(samples, df, question_col="question", category_col="category_merged", strict=True):
    # Strip whitespace on both sides of the join
    lookup = {q.strip(): cat for q, cat in zip(df[question_col], df[category_col])}

    unmatched = []
    for sample in samples:
        q = sample["doc"]["question"].strip()
        category = lookup.get(q)
        if category is None:
            unmatched.append(q)
        sample["category"] = category

    if unmatched:
        msg = f"{len(unmatched)} question(s) could not be matched. Example: {unmatched[0]!r}"
        if strict:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")
            print("Continuing anyway (strict=False), fix manually afterward.")

    n_matched = len(samples) - len(unmatched)
    print(f"Successfully attached category to {n_matched}/{len(samples)} samples.")
    return samples


if __name__ == "__main__":
    df = check_no_duplicate_questions()
    test_samples = [
        {"doc": {"question": df.iloc[0]["question"]}},
        {"doc": {"question": df.iloc[1]["question"]}},
    ]
    attach_category(test_samples, df)
    print(test_samples)