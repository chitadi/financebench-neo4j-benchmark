from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .answer import answer_question
from .config import Settings
from .data import load_questions
from .embeddings import Embedder
from .neo4j_store import Neo4jStore
from .retrieval import retrieve_context
from .scoring import context_dicts, score_result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "answer_correct",
        "doc_recall",
        "doc_precision",
        "page_recall",
        "citation_doc_precision",
        "citation_page_recall",
    ]
    summary: dict[str, Any] = {"questions": len(results)}
    for metric in metric_names:
        values = [
            item["scores"].get(metric)
            for item in results
            if item.get("scores", {}).get(metric) is not None
        ]
        summary[metric] = round(sum(values) / len(values), 4) if values else None
    summary["fallback_answers"] = sum(1 for item in results if item.get("answer", {}).get("fallback"))
    summary["avg_latency_s"] = (
        round(sum(item["timing"]["total_s"] for item in results) / len(results), 4)
        if results
        else None
    )
    return summary


def run_eval(
    *,
    settings: Settings,
    store: Neo4jStore,
    embedder: Embedder,
    limit: int | None = None,
) -> Path:
    questions = load_questions(settings.questions_file)
    if limit is not None:
        questions = questions[:limit]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.results_dir / f"{run_id}_{settings.embedding_provider}_{settings.llm_model}.json"

    results: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        t0 = time.monotonic()
        retrieve_start = time.monotonic()
        contexts = retrieve_context(
            question.question,
            store=store,
            embedder=embedder,
            settings=settings,
        )
        retrieve_s = time.monotonic() - retrieve_start

        answer_start = time.monotonic()
        answer = answer_question(question.question, contexts, settings=settings)
        answer_s = time.monotonic() - answer_start

        scores = score_result(
            question=question,
            answer=answer.answer,
            contexts=contexts,
            settings=settings,
        )
        item = {
            "index": index,
            "financebench_id": question.financebench_id,
            "question": question.question,
            "gold_answer": question.answer,
            "question_type": question.question_type,
            "question_reasoning": question.question_reasoning,
            "answer": {
                "text": answer.answer,
                "model": answer.model,
                "fallback": answer.fallback,
                "token_usage": answer.token_usage,
            },
            "contexts": context_dicts(contexts),
            "scores": scores,
            "timing": {
                "retrieval_s": round(retrieve_s, 4),
                "answer_s": round(answer_s, 4),
                "total_s": round(time.monotonic() - t0, 4),
            },
        }
        results.append(item)
        print(
            f"[{index}/{len(questions)}] {question.financebench_id} "
            f"answer={scores['answer_correct']} doc_recall={scores['doc_recall']} "
            f"page_recall={scores['page_recall']} total={item['timing']['total_s']}s"
        )

    payload = {
        "run_id": run_id,
        "config": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "llm_model": settings.llm_model,
            "judge_provider": settings.judge_provider,
            "retrieval_final_k": settings.retrieval_final_k,
            "retrieval_neighbor_window": settings.retrieval_neighbor_window,
        },
        "summary": aggregate(results),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path

