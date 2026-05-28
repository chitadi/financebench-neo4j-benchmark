from fbneo.neo4j_store import Neo4jStore


def test_fulltext_search_does_not_use_driver_reserved_query_param() -> None:
    captured: dict[str, object] = {}
    store = object.__new__(Neo4jStore)

    def fake_run(cypher: str, **params):
        captured["cypher"] = cypher
        captured["params"] = params
        return []

    store._run = fake_run

    assert store.fulltext_search("net ppne", 5) == []

    assert "$search_query" in captured["cypher"]
    assert captured["params"] == {"search_query": "net ppne", "k": 5}
