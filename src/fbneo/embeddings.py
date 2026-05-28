from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import requests

from .config import Settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9$%.,-]+")


class Embedder(ABC):
    dimension: int

    @abstractmethod
    def embed_one(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class HashEmbedder(Embedder):
    """Small deterministic embedding for local smoke tests.

    This is not a serious semantic embedding model. It exists so ingestion,
    vector indexes, and hybrid retrieval can be tested before API keys are set.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for raw_token in _TOKEN_RE.findall(text.lower()):
            token = raw_token.strip(".,")
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [round(v / norm, 8) for v in vector]


class OpenAICompatibleEmbedder(Embedder):
    def __init__(self, *, base_url: str, api_key: str, model: str, dimension: int) -> None:
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=openai")
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
            json={"model": self.model, "input": texts},
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
    if settings.embedding_provider == "hash":
        return HashEmbedder(settings.embedding_dimension)
    if settings.embedding_provider == "openai":
        return OpenAICompatibleEmbedder(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider!r}")

