import pandas as pd


def check_no_duplicate_questions(csv_path="../data/truthfulqa_generation.csv"):
    df = pd.read_csv(csv_path)
    n_dupes = df["question"].duplicated().sum()
    if n_dupes > 0:
        dupe_questions = df[df["question"].duplicated(keep=False)]["question"].unique()
        raise ValueError(
            f"{n_dupes} duplicate question(s) found — a text-based join is "
            f"unsafe without disambiguation. Duplicated questions: {list(dupe_questions)}"
        )
    print(f"No duplicate questions found across {len(df)} rows. Text-based join is safe.")
    return df


def attach_category(samples, df, question_col="question", category_col="category_merged"):
    lookup = dict(zip(df[question_col], df[category_col]))

    unmatched = []
    for sample in samples:
        q = sample["doc"]["question"]
        category = lookup.get(q)
        if category is None:
            unmatched.append(q)
        sample["category"] = category

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} question(s) from lm-eval output could not be "
            f"matched to a category. Example: {unmatched[0]!r}. "
            f"Check for whitespace/encoding differences between the harness's "
            f"internal dataset copy and data/truthfulqa_generation.csv."
        )

    print(f"Successfully attached category to all {len(samples)} samples.")
    return samples


if __name__ == "__main__":
    df = check_no_duplicate_questions()

    test_samples = [
        {"doc": {"question": df.iloc[0]["question"]}},
        {"doc": {"question": df.iloc[1]["question"]}},
    ]
    attach_category(test_samples, df)
    print(test_samples)