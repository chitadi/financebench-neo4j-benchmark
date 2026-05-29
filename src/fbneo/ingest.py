from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .chunking import chunk_page_text, stable_chunk_id
from .config import Settings
from .data import load_document_info, load_questions, question_document_info
from .embeddings import Embedder
from .pdf import extract_pages, find_pdf
from .types import Chunk, DocumentMeta, PageText

if TYPE_CHECKING:
    from .neo4j_store import Neo4jStore


@dataclass
class IngestStats:
    documents_seen: int = 0
    documents_ingested: int = 0
    documents_missing_pdf: int = 0
    pages_ingested: int = 0
    chunks_ingested: int = 0
    missing_pdfs: list[str] | None = None


def _embed_in_batches(embedder: Embedder, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed_many(texts[start : start + batch_size]))
    return vectors


def build_chunks(
    pages: list[PageText],
    *,
    settings: Settings,
    embedder: Embedder,
) -> list[Chunk]:
    chunk_specs: list[tuple[str, int, int, str]] = []
    next_chunk_index = 0
    for page in pages:
        page_chunks = chunk_page_text(
            page,
            start_chunk_index=next_chunk_index,
            chunk_size_words=settings.chunk_size_words,
            chunk_overlap_words=settings.chunk_overlap_words,
        )
        for chunk_index, text in page_chunks:
            chunk_specs.append((page.doc_name, page.page_num, chunk_index, text))
        if page_chunks:
            next_chunk_index = page_chunks[-1][0] + 1

    embeddings = _embed_in_batches(embedder, [spec[3] for spec in chunk_specs])
    chunks: list[Chunk] = []
    for (doc_name, page_num, chunk_index, text), embedding in zip(chunk_specs, embeddings):
        chunks.append(
            Chunk(
                chunk_id=stable_chunk_id(doc_name, page_num, chunk_index),
                doc_name=doc_name,
                page_num=page_num,
                chunk_index=chunk_index,
                text=text,
                embedding=embedding,
            )
        )
    return chunks


def load_document_manifest(settings: Settings) -> dict[str, DocumentMeta]:
    docs = load_document_info(settings.document_info_file)
    if settings.questions_file.exists():
        questions = load_questions(settings.questions_file)
        docs = question_document_info(questions, docs)
    return docs


def ingest_documents(
    *,
    settings: Settings,
    store: Neo4jStore,
    embedder: Embedder,
    limit_docs: int | None = None,
    doc_names: list[str] | None = None,
) -> IngestStats:
    docs = load_document_manifest(settings)
    selected = sorted(docs.values(), key=lambda d: d.doc_name)
    if doc_names:
        wanted = set(doc_names)
        selected = [doc for doc in selected if doc.doc_name in wanted]
    if limit_docs is not None:
        selected = selected[:limit_docs]

    stats = IngestStats(documents_seen=len(selected), missing_pdfs=[])
    for meta in selected:
        pdf_path = find_pdf(settings.pdf_dir, meta.doc_name)
        if pdf_path is None:
            stats.documents_missing_pdf += 1
            stats.missing_pdfs.append(meta.doc_name)
            continue

        pages = extract_pages(pdf_path, meta.doc_name)
        chunks = build_chunks(pages, settings=settings, embedder=embedder)
        store.upsert_document(meta, pages, chunks)
        stats.documents_ingested += 1
        stats.pages_ingested += len(pages)
        stats.chunks_ingested += len(chunks)
    return stats
