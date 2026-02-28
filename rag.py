# pdf_ingest.py
from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Union

import fitz  # PyMuPDF


def _clean_page_text(text: str) -> str:
    """
    Limpieza de texto extraído de una página PDF (texto real).
    - Quita guiones de final de línea (hyphenation)
    - Junta líneas dentro de párrafos
    - Normaliza espacios
    """
    if not text:
        return ""

    # Normaliza saltos
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Quita guionado al final de línea: "inter-\nnational" -> "international"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Separa en "bloques" por líneas en blanco (párrafos)
    blocks = re.split(r"\n\s*\n+", text)

    cleaned_blocks = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue

        # Junta saltos simples dentro del bloque
        # (mantiene listas si empiezan por viñeta / numeración)
        lines = [ln.strip() for ln in b.split("\n") if ln.strip()]
        if not lines:
            continue

        joined = []
        for ln in lines:
            # Conserva viñetas/numeración como saltos "suaves" si quieres; aquí las unimos igual.
            joined.append(ln)

        b2 = " ".join(joined)

        # Normaliza espacios
        b2 = re.sub(r"[ \t]+", " ", b2).strip()

        cleaned_blocks.append(b2)

    out = "\n\n".join(cleaned_blocks).strip()

    # Limpieza final: espacios antes de puntuación
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)

    return out


def _detect_repeated_header_footer(pages_lines: List[List[str]], ratio: float = 0.6):
    """
    Heurística simple:
    - header_candidate: primera línea no vacía de cada página
    - footer_candidate: última línea no vacía de cada página
    Si una misma línea aparece en >= ratio de páginas, se elimina.
    """
    if not pages_lines:
        return None, None

    headers = []
    footers = []
    for lines in pages_lines:
        nonempty = [ln.strip() for ln in lines if ln.strip()]
        headers.append(nonempty[0] if nonempty else "")
        footers.append(nonempty[-1] if nonempty else "")

    def most_common(lines: List[str]):
        freq = {}
        for x in lines:
            if not x:
                continue
            freq[x] = freq.get(x, 0) + 1
        if not freq:
            return None, 0
        best = max(freq.items(), key=lambda kv: kv[1])
        return best[0], best[1]

    header, h_count = most_common(headers)
    footer, f_count = most_common(footers)

    n = len(pages_lines)
    header_ok = header if header and (h_count / n) >= ratio else None
    footer_ok = footer if footer and (f_count / n) >= ratio else None

    return header_ok, footer_ok


def ingest_pdf_to_pages(pdf_path: str) -> List[Dict[str, Union[str, int]]]:
    """
    Devuelve una lista:
      [{"source": "<pdf>", "page": 1, "text": "<texto limpio>"}, ...]
    con limpieza básica + eliminación heurística de header/footer repetidos.
    """
    doc = fitz.open(pdf_path)

    # Primero extraemos líneas crudas para detectar headers/footers repetidos
    raw_pages_lines = []
    raw_pages_text = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        t = page.get_text("text") or ""
        raw_pages_text.append(t)
        raw_pages_lines.append(t.replace("\r\n", "\n").replace("\r", "\n").split("\n"))

    header_line, footer_line = _detect_repeated_header_footer(raw_pages_lines, ratio=0.6)

    out = []
    for i, raw_text in enumerate(raw_pages_text, start=1):
        # Elimina header/footer detectados (si aplica)
        lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        lines_stripped = [ln.strip() for ln in lines]

        if header_line:
            # elimina solo si coincide exactamente con la primera línea no vacía
            for idx, ln in enumerate(lines_stripped):
                if ln:
                    if ln == header_line:
                        lines[idx] = ""
                    break

        if footer_line:
            # elimina solo si coincide con la última línea no vacía
            for idx in range(len(lines_stripped) - 1, -1, -1):
                if lines_stripped[idx]:
                    if lines_stripped[idx] == footer_line:
                        lines[idx] = ""
                    break

        cleaned = _clean_page_text("\n".join(lines))

        # opcional: descartar páginas casi vacías
        if cleaned and len(cleaned) >= 20:
            out.append({"source": os.path.basename(pdf_path), "page": i, "text": cleaned})

    doc.close()
    return out


def ingest_folder(folder_path: str, exts=(".pdf",)) -> List[Dict[str, Union[str, int]]]:
    all_pages = []
    for fn in sorted(os.listdir(folder_path)):
        if fn.lower().endswith(exts):
            all_pages.extend(ingest_pdf_to_pages(os.path.join(folder_path, fn)))
    return all_pages


def save_jsonl(items: List[Dict[str, Union[str, int]]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Ejemplo: mete tus PDFs en ./pdfs/
    pages = ingest_folder("./rag_data")
    print(f"Extracted {len(pages)} page-items")
    # Guarda para el siguiente paso (chunking + indexado)
    os.makedirs("./rag_store", exist_ok=True)
    save_jsonl(pages, "./rag_store/pages.jsonl")
    print("Saved: ./rag_store/pages.jsonl")
