"""Utilities to download and format MMLU questions."""

from datasets import load_dataset
import pandas as pd

DATASET_ID = "cais/mmlu"
SUBSETS = ("professional_finance", "econometrics", "global_facts")
SUBSET_ALIASES = {"professional_finance": "professional_accounting"}
OPTIONS_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def resolve_subset(subset: str) -> str:
    """Return the subset name compatible with Hugging Face."""
    return SUBSET_ALIASES.get(subset, subset)


def load_questions(subset: str = SUBSETS[0], split: str = "test") -> pd.DataFrame:
    """
    Download the requested MMLU subset and return it as a DataFrame.

    subset: one of SUBSETS; "professional_finance" resolves to "professional_accounting".
    split: data split (e.g., "test", "validation", "dev").
    """
    resolved_subset = resolve_subset(subset)
    try:
        dataset = load_dataset(DATASET_ID, resolved_subset, split=split)
    except ValueError as exc:
        raise ValueError(
            f"Subset '{subset}' is not available in {DATASET_ID}. Use one of {SUBSETS}."
        ) from exc
    return pd.DataFrame(dataset)


def format_question(row: pd.Series) -> str:
    """Return the question and choices in a readable text block."""
    choices = row["choices"]
    return (
        f"Question: {row['question']}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n"
        f"\nCorrect Answer (Index): "
        f"{row['answer']} -> {OPTIONS_MAP[row['answer']]}"
    )
