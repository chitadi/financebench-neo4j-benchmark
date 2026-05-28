from __future__ import annotations

import re

from .config import Settings
from .embeddings import Embedder
from .neo4j_store import Neo4jStore
from .types import RetrievedContext

_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9$%.,/-]*")


def sanitize_fulltext_query(question: str) -> str:
    tokens = []
    for token in _QUERY_TOKEN_RE.findall(question):
        stripped = token.strip(".,")
        if len(stripped) >= 2:
            tokens.append(stripped)
    return " ".join(tokens[:80])


def _rrf_merge(
    vector_rows: list[dict],
    fulltext_rows: list[dict],
    *,
    final_k: int,
) -> list[tuple[str, float, list[str]]]:
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}

    for rank, row in enumerate(vector_rows, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
        sources.setdefault(cid, []).append("vector")

    for rank, row in enumerate(fulltext_rows, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
        sources.setdefault(cid, []).append("fulltext")

    return [
        (cid, score, sources.get(cid, []))
        for cid, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:final_k]
    ]


def retrieve_context(
    question: str,
    *,
    store: Neo4jStore,
    embedder: Embedder,
    settings: Settings,
) -> list[RetrievedContext]:
    query_embedding = embedder.embed_one(question)
    vector_rows = store.vector_search(query_embedding, settings.retrieval_vector_k)
    fulltext_rows = store.fulltext_search(
        sanitize_fulltext_query(question),
        settings.retrieval_fulltext_k,
    )
    merged = _rrf_merge(vector_rows, fulltext_rows, final_k=settings.retrieval_final_k)
    score_by_id = {cid: score for cid, score, _sources in merged}
    sources_by_id = {cid: sources for cid, _score, sources in merged}
    ordered_ids = [cid for cid, _score, _sources in merged]

    expanded = store.expand_neighbors(ordered_ids, settings.retrieval_neighbor_window)
    seen: set[str] = set()
    contexts: list[RetrievedContext] = []
    center_rank = {cid: i for i, cid in enumerate(ordered_ids)}

    def sort_key(row: dict) -> tuple[int, int]:
        center_id = row.get("center_id")
        return (center_rank.get(center_id, 10_000), int(row.get("chunk_index") or 0))

    for row in sorted(expanded, key=sort_key):
        chunk_id = row["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        center_id = row.get("center_id", chunk_id)
        contexts.append(
            RetrievedContext(
                chunk_id=chunk_id,
                doc_name=row["doc_name"],
                page_num=int(row["page_num"]),
                chunk_index=int(row["chunk_index"]),
                text=row["text"],
                score=float(score_by_id.get(center_id, 0.0)),
                sources=sources_by_id.get(center_id, []),
            )
        )
    return contexts


def context_to_prompt(contexts: list[RetrievedContext], max_chars: int = 24_000) -> str:
    parts: list[str] = []
    used = 0
    for i, ctx in enumerate(contexts, start=1):
        header = f"[Source {i}] doc={ctx.doc_name} page={ctx.page_num}\n"
        body = ctx.text.strip()
        block = f"{header}{body}\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining <= len(header) + 200:
                break
            block = f"{header}{body[: remaining - len(header)]}\n"
        parts.append(block)
        used += len(block)
    return "\n".join(parts)

