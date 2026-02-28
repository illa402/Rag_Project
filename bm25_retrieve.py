# bm25_retrieve.py
from __future__ import annotations

import pickle
import re
from typing import List, Dict, Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def load_bm25(path: str = "./rag_store/bm25.pkl") -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve_bm25(query: str, top_k: int = 5, store_path: str = "./rag_store/bm25.pkl"):
    store = load_bm25(store_path)
    bm25 = store["bm25"]
    tokens = tokenize(query)

    scores = bm25.get_scores(tokens)  # score por chunk
    # top_k índices por score
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in top_idx:
        results.append(
            {
                "chunk_id": store["chunk_ids"][i],
                "score": float(scores[i]),
                "text": store["texts"][i],
                "metadata": store["metadatas"][i],
            }
        )
    return results


if __name__ == "__main__":
    q = "How to compute portfolio standard deviation correlation covariance"
    hits = retrieve_bm25(q, top_k=3)
    for h in hits:
        print("\n---")
        print(h["chunk_id"], "score=", round(h["score"], 3), "meta=", h["metadata"])
        print(h["text"][:500], "...")
