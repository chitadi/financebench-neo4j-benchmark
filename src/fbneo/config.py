from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str

    data_dir: Path
    pdf_dir: Path
    results_dir: Path
    questions_file: Path
    document_info_file: Path

    chunk_size_words: int
    chunk_overlap_words: int

    retrieval_vector_k: int
    retrieval_fulltext_k: int
    retrieval_final_k: int
    retrieval_neighbor_window: int

    embedding_provider: str
    embedding_dimension: int
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float

    judge_provider: str
    judge_model: str


def load_settings() -> Settings:
    _load_dotenv()
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    pdf_dir_env = os.environ.get("PDF_DIR")
    if pdf_dir_env:
        pdf_dir = Path(pdf_dir_env)
    elif (data_dir / "pdfs").exists():
        pdf_dir = data_dir / "pdfs"
    else:
        pdf_dir = Path("pdfs")
    results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
    questions_file = Path(os.environ.get("QUESTIONS_FILE", data_dir / "financebench_open_source.jsonl"))
    document_info_file = Path(
        os.environ.get("DOCUMENT_INFO_FILE", data_dir / "financebench_document_information.jsonl")
    )

    return Settings(
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "financebench"),
        neo4j_database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        data_dir=data_dir,
        pdf_dir=pdf_dir,
        results_dir=results_dir,
        questions_file=questions_file,
        document_info_file=document_info_file,
        chunk_size_words=_int_env("CHUNK_SIZE_WORDS", 850),
        chunk_overlap_words=_int_env("CHUNK_OVERLAP_WORDS", 120),
        retrieval_vector_k=_int_env("RETRIEVAL_VECTOR_K", 30),
        retrieval_fulltext_k=_int_env("RETRIEVAL_FULLTEXT_K", 30),
        retrieval_final_k=_int_env("RETRIEVAL_FINAL_K", 8),
        retrieval_neighbor_window=_int_env("RETRIEVAL_NEIGHBOR_WINDOW", 1),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "hash").lower(),
        embedding_dimension=_int_env("EMBEDDING_DIMENSION", 384),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_base_url=os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY", ""),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        llm_temperature=_float_env("LLM_TEMPERATURE", 0.0),
        judge_provider=os.environ.get("JUDGE_PROVIDER", "heuristic").lower(),
        judge_model=os.environ.get("JUDGE_MODEL", ""),
    )
