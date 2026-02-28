# chunk_pages.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Iterable


ROOT = Path(__file__).resolve().parent
PAGES_PATH = ROOT / "rag_store" / "pages.jsonl"
CHUNKS_PATH = ROOT / "rag_store" / "chunks.jsonl"


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(items: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 250) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    chunks = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + max_chars, n)

        if end < n:
            cut_space = text.rfind(" ", i, end)
            cut_nl = text.rfind("\n", i, end)
            cut = max(cut_space, cut_nl)
            if cut > i + int(max_chars * 0.6):
                end = cut

        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        i = max(0, end - overlap)

    return chunks


def build_chunks(pages_path: Path = PAGES_PATH, out_path: Path = CHUNKS_PATH) -> None:
    chunks_out: List[Dict] = []

    for it in read_jsonl(pages_path):
        source = it.get("source", "unknown.pdf")
        page = int(it.get("page", -1))
        text = it.get("text", "")

        parts = chunk_text(text, max_chars=1800, overlap=250)
        for j, ch in enumerate(parts):
            chunk_id = f"{source}::p{page}::c{j}"
            chunks_out.append(
                {
                    "chunk_id": chunk_id,
                    "text": ch,
                    "metadata": {
                        "source": source,
                        "page": page,
                        "chunk_in_page": j,
                    },
                }
            )

    write_jsonl(chunks_out, out_path)
    print(f"✅ Saved {len(chunks_out)} chunks -> {out_path}")


if __name__ == "__main__":
    build_chunks()
