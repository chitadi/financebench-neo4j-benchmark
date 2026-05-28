from __future__ import annotations

import json
from pathlib import Path

from .types import DocumentMeta, EvidenceGold, FinanceBenchQuestion


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}: {exc}") from exc
    return rows


def load_document_info(path: Path) -> dict[str, DocumentMeta]:
    docs: dict[str, DocumentMeta] = {}
    if not path.exists():
        return docs
    for row in read_jsonl(path):
        doc_name = str(row.get("doc_name", "")).strip()
        if not doc_name:
            continue
        docs[doc_name] = DocumentMeta(
            doc_name=doc_name,
            company=str(row.get("company", "") or ""),
            doc_type=str(row.get("doc_type", "") or ""),
            doc_period=str(row.get("doc_period", "") or ""),
            doc_link=str(row.get("doc_link", "") or ""),
            company_sector_gics=str(
                row.get(
                    "company_sector_gics",
                    row.get("gics_sector", row.get("comany_sector_gics", "")),
                )
                or ""
            ),
        )
    return docs


def _parse_evidence(items: list[dict] | None) -> list[EvidenceGold]:
    evidence: list[EvidenceGold] = []
    for item in items or []:
        page_num = item.get("evidence_page_num")
        try:
            parsed_page = int(page_num) if page_num is not None else None
        except (TypeError, ValueError):
            parsed_page = None
        evidence.append(
            EvidenceGold(
                doc_name=str(item.get("doc_name", "") or ""),
                evidence_page_num=parsed_page,
                evidence_text=str(item.get("evidence_text", "") or ""),
                evidence_text_full_page=str(item.get("evidence_text_full_page", "") or ""),
            )
        )
    return evidence


def load_questions(path: Path) -> list[FinanceBenchQuestion]:
    questions: list[FinanceBenchQuestion] = []
    for row in read_jsonl(path):
        qid = str(row.get("financebench_id", row.get("id", "")) or "")
        questions.append(
            FinanceBenchQuestion(
                financebench_id=qid,
                question=str(row.get("question", "") or ""),
                answer=str(row.get("answer", "") or ""),
                doc_name=str(row.get("doc_name", "") or ""),
                company=str(row.get("company", "") or ""),
                question_type=str(row.get("question_type", "") or ""),
                question_reasoning=str(row.get("question_reasoning", "") or ""),
                justification=str(row.get("justification", "") or ""),
                evidence=_parse_evidence(row.get("evidence") or []),
                raw=row,
            )
        )
    return questions


def merge_question_document_info(
    questions: list[FinanceBenchQuestion],
    docs: dict[str, DocumentMeta],
) -> dict[str, DocumentMeta]:
    merged = dict(docs)
    for q in questions:
        if q.doc_name and q.doc_name not in merged:
            merged[q.doc_name] = DocumentMeta(
                doc_name=q.doc_name,
                company=q.company,
            )
    return merged


def validate_data_files(
    questions_file: Path,
    document_info_file: Path,
    pdf_dir: Path,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not questions_file.exists():
        errors.append(f"Missing questions file: {questions_file}")
    if not document_info_file.exists():
        errors.append(f"Missing document info file: {document_info_file}")
    if not pdf_dir.exists():
        errors.append(f"Missing PDF directory: {pdf_dir}")
    elif not any(pdf_dir.glob("*.pdf")):
        errors.append(f"No PDFs found in: {pdf_dir}")
    return (not errors, errors)
