# FinanceBench Neo4j GraphRAG Benchmark

Standalone V1 benchmark for comparing a basic Neo4j GraphRAG context layer on the public FinanceBench 150-question sample.

This repo intentionally does not modify or import the data intelligence platform. It also does not download FinanceBench artifacts. Place the JSONL files under `data/` and PDFs under `pdfs/`.

## V1 Scope

V1 proves the pipeline:

1. read FinanceBench JSONL metadata and PDFs
2. parse PDFs page-by-page
3. build a simple Neo4j lexical graph
4. create full-text and vector indexes
5. retrieve hybrid context
6. answer with Gemini 2.5 Flash through Google's OpenAI-compatible chat endpoint
7. score and report results

Rich finance-specific nodes and relationships are deferred to later versions.

## Project Layout

```text
src/fbneo/    Python package and CLI implementation
tests/        Unit tests
data/         FinanceBench JSONL files
pdfs/         FinanceBench PDFs
results/      Eval outputs
```

## Graph Model

```text
(:Company)-[:HAS_DOCUMENT]->(:Document)
(:Document)-[:HAS_PAGE]->(:Page)
(:Page)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:NEXT_CHUNK]->(:Chunk)
```

Node properties:

- `Company {name}`
- `Document {doc_name, company, doc_type, doc_period, doc_link}`
- `Page {page_id, doc_name, page_num, text}`
- `Chunk {chunk_id, doc_name, page_num, chunk_index, text, embedding}`

`page_num` is zero-based to match FinanceBench evidence page numbers.

## Setup

Python runs the benchmark CLI and installs libraries such as `requests`, `neo4j`, and `pymupdf`.
Docker runs the Neo4j database. You need both for the end-to-end benchmark.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
docker compose up -d
```

Place these files:

```text
data/
  financebench_open_source.jsonl
  financebench_document_information.jsonl
pdfs/
  <doc_name>.pdf
```

Validate local data:

```bash
python -m fbneo validate-data
```

## Commands

Smoke ingest two PDFs:

```bash
python -m fbneo ingest --limit-docs 2 --reset
```

Retrieve:

```bash
python -m fbneo retrieve --question "What is the year end FY2018 net PPNE for 3M?"
```

Answer:

```bash
python -m fbneo answer --question "What is the year end FY2018 net PPNE for 3M?"
```

Run eval:

```bash
python -m fbneo eval --limit 150
```

Generate markdown report:

```bash
python -m fbneo report --run results/<run>.json
```

## Model Configuration

The benchmark defaults are set up for Gemini answers and Qwen embeddings:

```env
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...

EMBEDDING_PROVIDER=openrouter
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=qwen/qwen3-embedding-4b
EMBEDDING_DIMENSION=2560

LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-2.5-flash
```

`EMBEDDING_API_KEY` can be used instead of `OPENROUTER_API_KEY`, and `LLM_API_KEY` can be used instead of `GEMINI_API_KEY`. OpenRouter embeddings are required for ingestion and retrieval. If `LLM_API_KEY` and `GEMINI_API_KEY` are empty, `answer` and `eval` use an extractive fallback after retrieval.

Neo4j vector indexes are created with a fixed embedding dimension. If you change embedding models or dimensions, run a full `python -m fbneo ingest --reset`; schema setup will recreate the vector index when the configured dimension changes.
