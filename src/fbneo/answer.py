from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import Settings
from .retrieval import context_to_prompt
from .types import RetrievedContext


ANSWER_SYSTEM_PROMPT = """You answer financial QA questions using only the provided context.
Be concise and include the relevant numerical value, unit, period, and company when available.
Cite sources inline using [Source N]. If the context does not support an answer, say so."""

ANSWER_USER_TEMPLATE = """Question:
{question}

Context:
{context}

Answer:"""


@dataclass
class AnswerResult:
    answer: str
    model: str
    token_usage: dict[str, Any] = field(default_factory=dict)
    fallback: bool = False


def _extractive_fallback(question: str, contexts: list[RetrievedContext]) -> AnswerResult:
    if not contexts:
        return AnswerResult(
            answer="I could not find relevant context to answer the question.",
            model="extractive-fallback",
            fallback=True,
        )
    snippets = []
    for i, ctx in enumerate(contexts[:3], start=1):
        text = re.sub(r"\s+", " ", ctx.text).strip()
        snippets.append(f"[Source {i}] {ctx.doc_name} p.{ctx.page_num}: {text[:700]}")
    return AnswerResult(
        answer=(
            "LLM_API_KEY is not configured, so this is a retrieval-only fallback. "
            f"Question: {question}\n\n" + "\n\n".join(snippets)
        ),
        model="extractive-fallback",
        fallback=True,
    )


def answer_question(
    question: str,
    contexts: list[RetrievedContext],
    *,
    settings: Settings,
) -> AnswerResult:
    if not settings.llm_api_key:
        return _extractive_fallback(question, contexts)

    context_text = context_to_prompt(contexts)
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ANSWER_USER_TEMPLATE.format(question=question, context=context_text),
            },
        ],
        "temperature": settings.llm_temperature,
    }
    response = requests.post(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return AnswerResult(
        answer=data["choices"][0]["message"]["content"],
        model=settings.llm_model,
        token_usage=data.get("usage") or {},
        fallback=False,
    )

