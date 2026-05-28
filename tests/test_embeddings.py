from fbneo.embeddings import HashEmbedder


def test_hash_embedding_dimension_and_stability() -> None:
    embedder = HashEmbedder(16)
    first = embedder.embed_one("net sales 2023")
    second = embedder.embed_one("net sales 2023")
    assert len(first) == 16
    assert first == second

