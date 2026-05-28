from __future__ import annotations

import hashlib
import re

from .types import PageText

_WORD_RE = re.compile(r"\S+")


def stable_chunk_id(doc_name: str, page_num: int, chunk_index: int) -> str:
    raw = f"{doc_name}|{page_num}|{chunk_index}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"chunk_{digest}"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_page_text(
    page: PageText,
    *,
    start_chunk_index: int,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[tuple[int, str]]:
    text = normalize_ws(page.text)
    if not text:
        return []

    words = [m.group(0) for m in _WORD_RE.finditer(text)]
    if len(words) <= chunk_size_words:
        return [(start_chunk_index, " ".join(words))]

    chunks: list[tuple[int, str]] = []
    step = max(1, chunk_size_words - chunk_overlap_words)
    chunk_index = start_chunk_index
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_size_words)
        chunk_text = " ".join(words[start:end])
        if chunk_text:
            chunks.append((chunk_index, chunk_text))
            chunk_index += 1
        if end >= len(words):
            break
    return chunks

