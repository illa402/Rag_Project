import json
import re
import requests
import pickle
from pathlib import Path
from typing import Any, Dict, List


OPTIONS_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_TO_INDEX = {v: k for k, v in OPTIONS_MAP.items()}
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_TIMEOUT_S = 300
OLLAMA_EMBED_TIMEOUT_S = 120
OLLAMA_NUM_PREDICT = 128
MAX_CONTEXT_CHARS_PER_HIT = 900
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}

def build_mcq_prompt(row) -> str:
    choices = row["choices"]
    return (
        "You are a multiple-choice test solver.\n"
        "Answer with ONLY one letter: A, B, C, or D.\n"
        "Do not output reasoning or explanations.\n\n"
        f"Question: {row['question']}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        "Correct Answer (Letter):"
        "VERY IMPORTANT: Answer with ONLY the letter.\n"
    )

def extract_letter(text: str) -> str | None:
    if not text:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    tail = lines[-3:]  # últimas 1-3 líneas
    print('tail para extraer letra: ' + str(tail))  # para debug
    for ln in reversed(tail):
        m = re.match(r"^\s*\(?\s*([ABCD])\s*\)?\s*$", ln.upper())
        if m:
            return m.group(1)

    m = re.search(r"\bANSWER\s*(?:IS|:)\s*([ABCD])\b", text.upper())
    if m:
        return m.group(1)

    return None


def _prepare_model_prompt(prompt: str, model: str) -> str:
    if model.lower().startswith("qwen3"):
        return "/no_think\n" + prompt
    return prompt

def ollama_generate(prompt: str, model: str, base_url: str = OLLAMA_BASE_URL) -> str:
    r = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": _prepare_model_prompt(prompt, model),
            "stream": False,
            # "think": False,
            # "reasoning": False,
            "options": {
                "temperature": 0,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
            "think": False,
        },
        timeout=OLLAMA_GENERATE_TIMEOUT_S,
    )
    r.raise_for_status()
    print('respuesta: ' + r.json()["response"])  # para debug
    return r.json()["response"]

def baseline_llm_answer(row, model: str = "qwen3:4b") -> int:
    prompt = build_mcq_prompt(row)
    letter = _ollama_generate_letter(prompt, model=model)
    if letter is None:
        # fallback simple: si no parsea, elige A (o podrías reintentar con otro prompt)
        return 0
    return LETTER_TO_INDEX[letter]

#ADD: RAG (BM25-only) answer function ---


ROOT = Path(__file__).resolve().parent
BM25_PATH = ROOT / "rag_store" / "bm25.pkl"

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

_BM25_STORE: Dict[str, Any] | None = None


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _load_bm25_store() -> Dict[str, Any]:
    global _BM25_STORE
    if _BM25_STORE is None:
        if not BM25_PATH.exists():
            raise FileNotFoundError(f"Missing {BM25_PATH}. Run build_bm25.py first.")
        with BM25_PATH.open("rb") as f:
            _BM25_STORE = pickle.load(f)
    return _BM25_STORE


def _retrieve_bm25(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    store = _load_bm25_store()
    bm25 = store["bm25"]
    scores = bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    hits = []
    for i in top_idx:
        hits.append(
            {
                "chunk_id": store["chunk_ids"][i],
                "score": float(scores[i]),
                "text": store["texts"][i],
                "metadata": store["metadatas"][i],
            }
        )
    return hits


def _extract_letter_strategy2(text: str) -> str | None:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    tail = lines[-3:]
    for ln in reversed(tail):
        m = re.match(r"^\s*\(?\s*([ABCD])\s*\)?\s*$", ln.upper())
        if m:
            return m.group(1)

    m = re.search(r"\bANSWER\s*(?:IS|:)\s*([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    return None


def _build_context(hits: List[Dict[str, Any]], max_chars_per_hit: int = MAX_CONTEXT_CHARS_PER_HIT) -> str:
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        text = (h.get("text") or "").strip()
        if len(text) > max_chars_per_hit:
            text = text[: max_chars_per_hit - 3].rstrip() + "..."
        context_blocks.append(
            f"[{i}] (source={h['metadata'].get('source')}, page={h['metadata'].get('page')})\n{text}"
        )
    return "\n\n".join(context_blocks)


def _ollama_generate(prompt: str, model: str = "qwen3:4b", base_url: str = OLLAMA_BASE_URL) -> str:
    r = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": _prepare_model_prompt(prompt, model),
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
            "think": False,
        },
        timeout=OLLAMA_GENERATE_TIMEOUT_S,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")

    data = r.json()
    return (data.get("response") or "").strip()


def _ollama_generate_letter(prompt: str, model: str = "qwen3:4b", base_url: str = OLLAMA_BASE_URL) -> str | None:
    json_prompt = prompt + "\n\nReturn the best answer using the required structured output."
    r = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": _prepare_model_prompt(json_prompt, model),
            "stream": False,
            "format": ANSWER_SCHEMA,
            "options": {
                "temperature": 0,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
            "think": False,
        },
        timeout=OLLAMA_GENERATE_TIMEOUT_S,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")

    response = (r.json().get("response") or "").strip()
    if not response:
        return None

    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return _extract_letter_strategy2(response)

    answer = str(payload.get("answer", "")).strip().upper()
    return answer if answer in LETTER_TO_INDEX else None


def rag_answer_bm25(row: Any, model: str = "gemma3:4b", top_k_ctx: int = 6) -> int:
    hits = _retrieve_bm25(row["question"], top_k=top_k_ctx)

    ctx = _build_context(hits)

    c = row["choices"]
    prompt = (
        "Use ONLY the provided context to answer the question.\n"
        "Return ONLY one letter: A, B, C, or D. No explanation.\n"
        "Do not output reasoning, tags, or extra words.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {row['question']}\n"
        f"A) {c[0]}\n"
        f"B) {c[1]}\n"
        f"C) {c[2]}\n"
        f"D) {c[3]}\n\n"
        "Answer:"
    )

    letter = _ollama_generate_letter(prompt, model=model)
    return LETTER_TO_INDEX.get(letter, 0)

def rag_answer_hybrid(
    row: Any,
    model: str = "gemma3:4b",
    top_k_bm25: int = 8,
    top_k_ctx: int = 4,
) -> int:
    """Compatibilidad: el modo híbrido queda degradado a BM25-only."""
    top_k_ctx = min(top_k_bm25, top_k_ctx)
    return rag_answer_bm25(row, model=model, top_k_ctx=top_k_ctx)
