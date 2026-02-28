from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Iterable

import chromadb
import requests


ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = ROOT / "rag_store" / "chunks.jsonl"
CHROMA_DIR = ROOT / "rag_store" / "chroma_ollama"
COLLECTION_NAME = "rag_chunks"


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ollama_embed_batch(
    texts: List[str],
    model: str = "embeddinggemma",
    base_url: str = "http://localhost:11434",
) -> List[List[float]]:
    """
    Llama al endpoint oficial /api/embed. Acepta input como lista (batch).
    Devuelve una lista de embeddings (uno por texto).
    """
    r = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": texts},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama embed error {r.status_code}: {r.text}")

    data = r.json()
    # Ollama devuelve {"embeddings":[[...],[...],...]}
    return data["embeddings"]


def build_vectorstore(
    embed_model: str = "embeddinggemma",
    batch_size: int = 32,
    base_url: str = "http://localhost:11434",
) -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}. Genera chunks.jsonl primero.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Si ya existe la colección, la borramos para evitar IDs duplicados
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []

    # Recorremos chunks y los insertamos por lotes
    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[Dict] = []

    total = 0
    for it in read_jsonl(CHUNKS_PATH):
        batch_ids.append(it["chunk_id"])
        batch_docs.append(it["text"])
        batch_metas.append(it.get("metadata", {}))

        if len(batch_docs) >= batch_size:
            embs = ollama_embed_batch(batch_docs, model=embed_model, base_url=base_url)
            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embs)

            total += len(batch_docs)
            print(f"Indexed {total} chunks...")

            batch_ids, batch_docs, batch_metas = [], [], []

    # último lote
    if batch_docs:
        embs = ollama_embed_batch(batch_docs, model=embed_model, base_url=base_url)
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embs)
        total += len(batch_docs)


    print(f"✅ Done. Stored {total} chunks in {CHROMA_DIR} (collection='{COLLECTION_NAME}').")


if __name__ == "__main__":
    # Cambia embed_model si has instalado otro (qwen3-embedding, all-minilm, etc.)
    build_vectorstore(embed_model="embeddinggemma", batch_size=32)
