# evaluate_baseline.py
"""
Evalúa la precisión (accuracy) del baseline LLM (sin RAG) sobre preguntas MMLU.

Uso (ejemplos):
  python evaluate_baseline.py
  python evaluate_baseline.py --subset professional_finance --split test --n 50 --model qwen3:4b
  python evaluate_baseline.py --all-subsets --split validation --n 100 --model gemma3:4b
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import pandas as pd

from data import SUBSETS, load_questions
from answers import baseline_llm_answer
from answers import rag_answer_bm25

def evaluate_df(
    df: pd.DataFrame,
    model: str,
    sleep_s: float = 0.0,
    verbose: bool = True,
    #asnwer=baseline_llm_answer,
    asnwer=rag_answer_bm25,
) -> Tuple[float, int, int]:
    """Devuelve (accuracy, correct, total)."""
    correct = 0
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            pred = asnwer(row, model=model)
            is_ok = int(pred == row["answer"])
            correct += is_ok

            if verbose:
                running_acc = correct / i
                print(f"[{i:>4}/{total}] pred={pred} gold={row['answer']} ok={bool(is_ok)} | acc={running_acc:.3f}")

        except Exception as e:
            # Cuenta como fallo, pero no rompe el batch
            if verbose:
                print(f"[{i:>4}/{total}] ERROR: {e!r} (counted as wrong)")

        if sleep_s > 0:
            time.sleep(sleep_s)

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


def evaluate_subset(
    subset: str,
    split: str,
    n: int | None,
    seed: int,
    model: str,
    sleep_s: float,
    verbose: bool,
) -> Dict[str, float]:
    df = load_questions(subset=subset, split=split)

    if n is not None:
        # muestreo reproducible
        df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    acc, correct, total = evaluate_df(df, model=model, sleep_s=sleep_s, verbose=verbose)

    return {
        "subset": subset,
        "split": split,
        "n": total,
        "correct": correct,
        "accuracy": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default=SUBSETS[0], help=f"Subset a evaluar. Opciones: {SUBSETS}")
    parser.add_argument("--all-subsets", action="store_true", help="Evalúa todos los subsets de SUBSETS.")
    parser.add_argument("--split", type=str, default="train", help="Split: test / validation / dev (según disponibilidad).")
    parser.add_argument("--n", type=int, default=30, help="Número de preguntas a muestrear (None para todo).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para el muestreo.")
    parser.add_argument("--model", type=str, default="qwen3:4b", help="Modelo en Ollama (p.ej. qwen3:4b).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Segundos de pausa entre preguntas (opcional).")
    parser.add_argument("--quiet", action="store_true", help="Menos logs (solo resumen final).")
    args = parser.parse_args()

    verbose = not args.quiet
    n = None if args.n is None or args.n <= 0 else args.n

    subsets = list(SUBSETS) if args.all_subsets else [args.subset]

    results = []
    t0 = time.time()
    for subset in subsets:
        print(f"\n=== Evaluating subset='{subset}' split='test' model='{args.model}' n={n or 'ALL'} ===")
        res = evaluate_subset(
            subset=subset,
            split=args.split,
            n=n,
            seed=args.seed,
            model=args.model,
            sleep_s=args.sleep,
            verbose=verbose,
        )
        results.append(res)
        print(f"-> {subset}: accuracy={res['accuracy']:.3f} ({res['correct']}/{res['n']})")

    elapsed = time.time() - t0

    # Resumen global (promedio ponderado por nº de preguntas)
    total_n = sum(r["n"] for r in results)
    total_correct = sum(r["correct"] for r in results)
    overall_acc = (total_correct / total_n) if total_n else 0.0

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['subset']:<25} acc={r['accuracy']:.3f}  ({int(r['correct'])}/{int(r['n'])})  split={r['split']}")

    print(f"\nOVERALL accuracy={overall_acc:.3f} ({total_correct}/{total_n})")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
