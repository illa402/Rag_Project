# build_bm25.py
from __future__ import annotations

import json
import os
import pickle
import re
from typing import Dict, List, Iterable

from rank_bm25 import BM25Okapi


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# Tokenizador simple (funciona bien para inglés + números + siglas)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")  # "WACC", "2SLS", "EBITDA", "10-K"


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def build_bm25(
    chunks_jsonl: str = "./rag_store/chunks.jsonl",
    out_path: str = "./rag_store/bm25.pkl",
) -> None:
    chunk_ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict] = []
    corpus_tokens: List[List[str]] = []

    for it in read_jsonl(chunks_jsonl):
        chunk_ids.append(it["chunk_id"])
        texts.append(it["text"])
        metadatas.append(it.get("metadata", {}))
        corpus_tokens.append(tokenize(it["text"]))

    bm25 = BM25Okapi(corpus_tokens)

    payload = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "texts": texts,
        "metadatas": metadatas,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"BM25 guardado en {out_path}")
    print(f"   chunks: {len(chunk_ids)}")


if __name__ == "__main__":
    build_bm25()
