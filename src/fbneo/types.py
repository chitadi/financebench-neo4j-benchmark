from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentMeta:
    doc_name: str
    company: str = ""
    doc_type: str = ""
    doc_period: str = ""
    doc_link: str = ""
    company_sector_gics: str = ""


@dataclass(frozen=True)
class EvidenceGold:
    doc_name: str
    evidence_page_num: int | None
    evidence_text: str = ""
    evidence_text_full_page: str = ""


@dataclass(frozen=True)
class FinanceBenchQuestion:
    financebench_id: str
    question: str
    answer: str
    doc_name: str
    company: str = ""
    question_type: str = ""
    question_reasoning: str = ""
    justification: str = ""
    evidence: list[EvidenceGold] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PageText:
    doc_name: str
    page_num: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_name: str
    page_num: int
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass
class RetrievedContext:
    chunk_id: str
    doc_name: str
    page_num: int
    chunk_index: int
    text: str
    score: float
    sources: list[str] = field(default_factory=list)

