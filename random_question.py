import random
from typing import Any

from data import OPTIONS_MAP, SUBSETS, format_question, load_questions


def pick_random_question(subset: str = SUBSETS[0], split: str = "test") -> Any:
    df = load_questions(subset=subset, split=split)
    return df.sample(1).iloc[0]


def pick_random_answer(question: Any) -> int:
    return random.randint(0, len(question["choices"]) - 1)


def check_answer(question: Any, guess_index: int) -> bool:
    return guess_index == question["answer"]


def demo(subset: str = SUBSETS[0], split: str = "test") -> bool:
    question = pick_random_question(subset=subset, split=split)
    guess_index = pick_random_answer(question)
    is_correct = check_answer(question, guess_index)

    print(format_question(question))
    print(f"\nRandom guess: {guess_index} -> {OPTIONS_MAP.get(guess_index, '?')}")
    print(f"Is correct? {'yes' if is_correct else 'no'}")

    return is_correct


if __name__ == "__main__":
    demo()
