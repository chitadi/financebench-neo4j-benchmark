from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

import requests

from .config import Settings
from .types import FinanceBenchQuestion, RetrievedContext

_NUM_RE = re.compile(r"-?\$?\d+(?:,\d{3})*(?:\.\d+)?%?")
_SOURCE_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _numbers(text: str) -> list[str]:
    values = []
    for value in _NUM_RE.findall(text):
        values.append(value.replace("$", "").replace(",", "").rstrip("%"))
    return values


def expected_docs(question: FinanceBenchQuestion) -> set[str]:
    docs = {ev.doc_name for ev in question.evidence if ev.doc_name}
    if question.doc_name:
        docs.add(question.doc_name)
    return docs


def expected_pages(question: FinanceBenchQuestion) -> set[tuple[str, int]]:
    pages = {
        (ev.doc_name or question.doc_name, ev.evidence_page_num)
        for ev in question.evidence
        if ev.evidence_page_num is not None and (ev.doc_name or question.doc_name)
    }
    return {(doc, int(page)) for doc, page in pages}


def heuristic_answer_correct(gold: str, answer: str) -> tuple[float, str]:
    gold_n = _norm(gold)
    answer_n = _norm(answer)
    if not gold_n:
        return 0.0, "No gold answer provided."
    if gold_n in answer_n:
        return 1.0, "Gold answer string appears in model answer."

    gold_nums = _numbers(gold)
    answer_nums = _numbers(answer)
    if gold_nums and all(num in answer_nums for num in gold_nums):
        return 1.0, "All numeric values from the gold answer appear in model answer."

    gold_tokens = [t for t in re.findall(r"[a-z0-9]+", gold_n) if len(t) > 2]
    if gold_tokens:
        matched = sum(1 for token in gold_tokens if token in answer_n)
        ratio = matched / len(gold_tokens)
        if ratio >= 0.8:
            return 1.0, "Most important gold-answer tokens appear in model answer."
        return round(ratio, 2), "Partial lexical overlap with the gold answer."

    return 0.0, "Gold answer not found in model answer."


def llm_answer_correct(
    *,
    question: FinanceBenchQuestion,
    answer: str,
    settings: Settings,
) -> tuple[float, str]:
    model = settings.judge_model or settings.llm_model
    if not settings.llm_api_key:
        return heuristic_answer_correct(question.answer, answer)

    prompt = f"""Judge whether the candidate answer is correct for the question.
Return JSON only with keys score and reason. score must be 0, 0.5, or 1.

Question: {question.question}
Gold answer: {question.answer}
Gold justification: {question.justification}
Candidate answer: {answer}
"""
    response = requests.post(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
        return float(data.get("score", 0.0)), str(data.get("reason", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        return heuristic_answer_correct(question.answer, answer)


def score_result(
    *,
    question: FinanceBenchQuestion,
    answer: str,
    contexts: list[RetrievedContext],
    settings: Settings,
) -> dict[str, Any]:
    exp_docs = expected_docs(question)
    exp_pages = expected_pages(question)
    got_docs = {ctx.doc_name for ctx in contexts}
    got_pages = {(ctx.doc_name, ctx.page_num) for ctx in contexts}

    doc_recall = len(exp_docs & got_docs) / len(exp_docs) if exp_docs else 1.0
    doc_precision = len(exp_docs & got_docs) / len(got_docs) if got_docs else 0.0
    page_recall = len(exp_pages & got_pages) / len(exp_pages) if exp_pages else None

    cited_nums = [int(n) for n in _SOURCE_RE.findall(answer)]
    cited_contexts = [contexts[n - 1] for n in cited_nums if 1 <= n <= len(contexts)]
    cited_pages = {(ctx.doc_name, ctx.page_num) for ctx in cited_contexts}
    cited_docs = {ctx.doc_name for ctx in cited_contexts}
    citation_doc_precision = (
        len(exp_docs & cited_docs) / len(cited_docs) if cited_docs else 0.0
    )
    citation_page_recall = (
        len(exp_pages & cited_pages) / len(exp_pages) if exp_pages and cited_pages else None
    )

    if settings.judge_provider == "llm":
        answer_score, answer_reason = llm_answer_correct(
            question=question,
            answer=answer,
            settings=settings,
        )
    else:
        answer_score, answer_reason = heuristic_answer_correct(question.answer, answer)

    return {
        "answer_correct": answer_score,
        "answer_reason": answer_reason,
        "doc_recall": round(doc_recall, 4),
        "doc_precision": round(doc_precision, 4),
        "page_recall": round(page_recall, 4) if page_recall is not None else None,
        "citation_doc_precision": round(citation_doc_precision, 4),
        "citation_page_recall": (
            round(citation_page_recall, 4) if citation_page_recall is not None else None
        ),
        "expected_docs": sorted(exp_docs),
        "retrieved_docs": sorted(got_docs),
        "expected_pages": sorted([{"doc_name": d, "page_num": p} for d, p in exp_pages], key=str),
        "retrieved_pages": sorted([{"doc_name": d, "page_num": p} for d, p in got_pages], key=str),
    }


def context_dicts(contexts: list[RetrievedContext]) -> list[dict[str, Any]]:
    return [asdict(ctx) for ctx in contexts]

