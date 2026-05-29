from __future__ import annotations

from abc import ABC, abstractmethod

import requests

from .config import Settings


class Embedder(ABC):
    dimension: int

    @abstractmethod
    def embed_one(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class OpenAICompatibleEmbedder(Embedder):
    def __init__(self, *, base_url: str, api_key: str, model: str, dimension: int) -> None:
        if not api_key:
            raise ValueError(
                "EMBEDDING_API_KEY or OPENROUTER_API_KEY is required when "
                "EMBEDDING_PROVIDER uses an OpenAI-compatible API"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts, "encoding_format": "float"},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        by_index = sorted(payload["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in by_index]
        if vectors and len(vectors[0]) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(vectors[0])}, "
                f"configured EMBEDDING_DIMENSION={self.dimension}"
            )
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed_many([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider != "openrouter":
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
            "This benchmark uses EMBEDDING_PROVIDER=openrouter."
        )
    return OpenAICompatibleEmbedder(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
