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
6. answer with an OpenAI-compatible chat model
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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
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

Default mode uses deterministic local hash embeddings so ingestion can run without model keys. For a real benchmark, configure OpenAI-compatible embeddings and chat:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
LLM_MODEL=gpt-4o-mini
```

If `LLM_API_KEY` is empty, `answer` and `eval` use an extractive fallback so the pipeline still runs.
