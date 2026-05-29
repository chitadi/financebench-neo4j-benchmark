import pytest
from pathlib import Path

from fbneo.config import Settings
from fbneo.embeddings import OpenAICompatibleEmbedder, build_embedder


def _settings(provider: str = "openrouter") -> Settings:
    return Settings(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        data_dir=Path("data"),
        pdf_dir=Path("pdfs"),
        results_dir=Path("results"),
        questions_file=Path("data/questions.jsonl"),
        document_info_file=Path("data/docs.jsonl"),
        chunk_size_words=850,
        chunk_overlap_words=120,
        retrieval_vector_k=30,
        retrieval_fulltext_k=30,
        retrieval_final_k=8,
        retrieval_neighbor_window=1,
        embedding_provider=provider,
        embedding_dimension=2560,
        embedding_model="qwen/qwen3-embedding-4b",
        embedding_base_url="https://openrouter.ai/api/v1",
        embedding_api_key="test-key",
        llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        llm_api_key="test-key",
        llm_model="gemini-2.5-flash",
        llm_temperature=0,
        judge_provider="heuristic",
        judge_model="",
    )


def test_build_embedder_requires_openrouter_provider() -> None:
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER=openrouter"):
        build_embedder(_settings(provider="unsupported"))


def test_build_embedder_returns_openai_compatible_embedder_for_openrouter() -> None:
    assert isinstance(build_embedder(_settings()), OpenAICompatibleEmbedder)
