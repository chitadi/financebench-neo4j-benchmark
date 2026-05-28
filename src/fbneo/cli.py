from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import load_settings
from .data import load_questions, validate_data_files
from .pdf import find_pdf


def cmd_validate_data(_args: argparse.Namespace) -> int:
    settings = load_settings()
    ok, errors = validate_data_files(
        settings.questions_file,
        settings.document_info_file,
        settings.pdf_dir,
    )
    if not ok:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    from .data import load_document_info, merge_question_document_info

    questions = load_questions(settings.questions_file)
    docs = merge_question_document_info(questions, load_document_info(settings.document_info_file))
    missing = [doc.doc_name for doc in docs.values() if find_pdf(settings.pdf_dir, doc.doc_name) is None]
    print(f"Questions: {len(questions)}")
    print(f"Documents in manifest: {len(docs)}")
    print(f"PDFs in {settings.pdf_dir}: {len(list(settings.pdf_dir.glob('*.pdf')))}")
    if missing:
        print(f"Missing PDFs for manifest docs: {len(missing)}")
        for doc_name in missing[:20]:
            print(f"  - {doc_name}.pdf")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
        return 1
    print("Data validation passed.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = load_settings()
    from .embeddings import build_embedder
    from .ingest import ingest_documents
    from .neo4j_store import Neo4jStore

    embedder = build_embedder(settings)
    with Neo4jStore(settings) as store:
        store.verify_connectivity()
        if args.reset:
            print("Resetting Neo4j graph...")
            store.reset()
        store.ensure_schema(settings.embedding_dimension)
        stats = ingest_documents(
            settings=settings,
            store=store,
            embedder=embedder,
            limit_docs=args.limit_docs,
            doc_names=args.doc_names,
        )
        store.ensure_schema(settings.embedding_dimension)
        print(json.dumps(asdict(stats), indent=2))
        print("Graph counts:")
        print(json.dumps(store.counts(), indent=2))
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    settings = load_settings()
    from .embeddings import build_embedder
    from .neo4j_store import Neo4jStore
    from .retrieval import retrieve_context
    from .scoring import context_dicts

    embedder = build_embedder(settings)
    with Neo4jStore(settings) as store:
        contexts = retrieve_context(
            args.question,
            store=store,
            embedder=embedder,
            settings=settings,
        )
    if args.json:
        print(json.dumps(context_dicts(contexts), indent=2))
    else:
        for i, ctx in enumerate(contexts, start=1):
            snippet = " ".join(ctx.text.split())[:900]
            print(f"[Source {i}] {ctx.doc_name} page={ctx.page_num} score={ctx.score:.4f}")
            print(snippet)
            print()
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    settings = load_settings()
    from .answer import answer_question
    from .embeddings import build_embedder
    from .neo4j_store import Neo4jStore
    from .retrieval import retrieve_context
    from .scoring import context_dicts

    embedder = build_embedder(settings)
    with Neo4jStore(settings) as store:
        contexts = retrieve_context(
            args.question,
            store=store,
            embedder=embedder,
            settings=settings,
        )
    answer = answer_question(args.question, contexts, settings=settings)
    if args.json:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "answer": asdict(answer),
                    "contexts": context_dicts(contexts),
                },
                indent=2,
            )
        )
    else:
        print(answer.answer)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = load_settings()
    from .embeddings import build_embedder
    from .evaluate import run_eval
    from .neo4j_store import Neo4jStore

    embedder = build_embedder(settings)
    with Neo4jStore(settings) as store:
        output = run_eval(settings=settings, store=store, embedder=embedder, limit=args.limit)
    print(f"Wrote {output}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    report = write_report(Path(args.run))
    print(f"Wrote {report}")
    return 0


def cmd_stats(_args: argparse.Namespace) -> int:
    settings = load_settings()
    from .neo4j_store import Neo4jStore

    with Neo4jStore(settings) as store:
        print(json.dumps(store.counts(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fbneo", description="FinanceBench Neo4j benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data", help="Check local FinanceBench JSONL/PDF files")
    validate.set_defaults(func=cmd_validate_data)

    ingest = sub.add_parser("ingest", help="Parse PDFs and write the V1 lexical graph to Neo4j")
    ingest.add_argument("--limit-docs", type=int, default=None, help="Only ingest the first N docs")
    ingest.add_argument("--doc-names", nargs="*", default=None, help="Only ingest these doc_name values")
    ingest.add_argument("--reset", action="store_true", help="Delete the current graph before ingest")
    ingest.set_defaults(func=cmd_ingest)

    retrieve = sub.add_parser("retrieve", help="Run hybrid retrieval for a question")
    retrieve.add_argument("--question", required=True)
    retrieve.add_argument("--json", action="store_true")
    retrieve.set_defaults(func=cmd_retrieve)

    answer = sub.add_parser("answer", help="Retrieve and answer a question")
    answer.add_argument("--question", required=True)
    answer.add_argument("--json", action="store_true")
    answer.set_defaults(func=cmd_answer)

    eval_cmd = sub.add_parser("eval", help="Run FinanceBench eval")
    eval_cmd.add_argument("--limit", type=int, default=None, help="Run only first N questions")
    eval_cmd.set_defaults(func=cmd_eval)

    report = sub.add_parser("report", help="Generate markdown report for a run JSON")
    report.add_argument("--run", required=True)
    report.set_defaults(func=cmd_report)

    stats = sub.add_parser("stats", help="Print Neo4j graph counts")
    stats.set_defaults(func=cmd_stats)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
