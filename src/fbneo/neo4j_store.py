from __future__ import annotations

from dataclasses import asdict
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from .config import Settings
from .types import Chunk, DocumentMeta, PageText


class Neo4jStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _run(self, cypher: str, **params: Any) -> list[dict]:
        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def _existing_vector_dimension(self) -> int | None:
        try:
            rows = self._run(
                """
                SHOW INDEXES
                YIELD name, options
                WHERE name = 'chunk_embedding_vector'
                RETURN options
                """
            )
        except Neo4jError:
            return None
        if not rows:
            return None
        config = (rows[0].get("options") or {}).get("indexConfig") or {}
        dimension = config.get("vector.dimensions")
        if dimension is None:
            return None
        return int(dimension)

    def ensure_schema(self, embedding_dimension: int) -> None:
        statements = [
            "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            (
                "CREATE CONSTRAINT document_doc_name IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.doc_name IS UNIQUE"
            ),
            "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.page_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE INDEX document_company IF NOT EXISTS FOR (d:Document) ON (d.company)",
            "CREATE INDEX chunk_doc_name IF NOT EXISTS FOR (c:Chunk) ON (c.doc_name)",
            "CREATE INDEX chunk_page_num IF NOT EXISTS FOR (c:Chunk) ON (c.page_num)",
            (
                "CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS "
                "FOR (c:Chunk) ON EACH [c.text]"
            ),
        ]
        for statement in statements:
            self._run(statement)
        existing_dimension = self._existing_vector_dimension()
        if existing_dimension is not None and existing_dimension != embedding_dimension:
            self._run("DROP INDEX chunk_embedding_vector IF EXISTS")
        self._run(
            "CREATE VECTOR INDEX chunk_embedding_vector IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dimension}, "
            "`vector.similarity_function`: 'cosine'}}}"
        )
        self._run("CALL db.awaitIndexes(300)")

    def reset(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")

    def upsert_document(self, meta: DocumentMeta, pages: list[PageText], chunks: list[Chunk]) -> None:
        page_rows = [
            {
                "page_id": f"{page.doc_name}::{page.page_num}",
                "doc_name": page.doc_name,
                "page_num": page.page_num,
                "text": page.text,
            }
            for page in pages
        ]
        chunk_rows = [
            {
                **asdict(chunk),
                "page_id": f"{chunk.doc_name}::{chunk.page_num}",
            }
            for chunk in chunks
        ]
        pairs = [
            [chunks[i].chunk_id, chunks[i + 1].chunk_id]
            for i in range(len(chunks) - 1)
            if chunks[i].doc_name == chunks[i + 1].doc_name
        ]

        def write_graph(tx):
            tx.run(
                """
                MATCH (d:Document {doc_name: $doc_name})-[:HAS_PAGE]->(p:Page)
                OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE c
                """,
                doc_name=meta.doc_name,
            )
            tx.run(
                """
                MATCH (d:Document {doc_name: $doc_name})-[:HAS_PAGE]->(p:Page)
                DETACH DELETE p
                """,
                doc_name=meta.doc_name,
            )
            tx.run(
                """
                MERGE (company:Company {name: $company})
                MERGE (doc:Document {doc_name: $doc_name})
                SET doc.company = $company,
                    doc.doc_type = $doc_type,
                    doc.doc_period = $doc_period,
                    doc.doc_link = $doc_link,
                    doc.company_sector_gics = $company_sector_gics
                MERGE (company)-[:HAS_DOCUMENT]->(doc)
                """,
                company=meta.company or "UNKNOWN",
                doc_name=meta.doc_name,
                doc_type=meta.doc_type,
                doc_period=str(meta.doc_period or ""),
                doc_link=meta.doc_link,
                company_sector_gics=meta.company_sector_gics,
            )
            tx.run(
                """
                UNWIND $pages AS row
                MATCH (doc:Document {doc_name: row.doc_name})
                MERGE (page:Page {page_id: row.page_id})
                SET page.doc_name = row.doc_name,
                    page.page_num = row.page_num,
                    page.text = row.text
                MERGE (doc)-[:HAS_PAGE]->(page)
                """,
                pages=page_rows,
            )
            tx.run(
                """
                UNWIND $chunks AS row
                MATCH (page:Page {page_id: row.page_id})
                MERGE (chunk:Chunk {chunk_id: row.chunk_id})
                SET chunk.doc_name = row.doc_name,
                    chunk.page_num = row.page_num,
                    chunk.chunk_index = row.chunk_index,
                    chunk.text = row.text,
                    chunk.embedding = row.embedding
                MERGE (page)-[:HAS_CHUNK]->(chunk)
                """,
                chunks=chunk_rows,
            )
            tx.run(
                """
                UNWIND $pairs AS pair
                MATCH (a:Chunk {chunk_id: pair[0]})
                MATCH (b:Chunk {chunk_id: pair[1]})
                MERGE (a)-[:NEXT_CHUNK]->(b)
                """,
                pairs=pairs,
            )

        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.execute_write(write_graph)

    def fulltext_search(self, query: str, k: int) -> list[dict]:
        if not query.strip():
            return []
        try:
            return self._run(
                """
                CALL db.index.fulltext.queryNodes('chunk_text_fulltext', $query, {limit: $k})
                YIELD node, score
                RETURN node.chunk_id AS chunk_id,
                       node.doc_name AS doc_name,
                       node.page_num AS page_num,
                       node.chunk_index AS chunk_index,
                       node.text AS text,
                       score AS score
                """,
                query=query,
                k=k,
            )
        except Neo4jError:
            return []

    def vector_search(self, embedding: list[float], k: int) -> list[dict]:
        try:
            return self._run(
                """
                CALL db.index.vector.queryNodes('chunk_embedding_vector', $k, $embedding)
                YIELD node, score
                RETURN node.chunk_id AS chunk_id,
                       node.doc_name AS doc_name,
                       node.page_num AS page_num,
                       node.chunk_index AS chunk_index,
                       node.text AS text,
                       score AS score
                """,
                embedding=embedding,
                k=k,
            )
        except Neo4jError:
            return []

    def expand_neighbors(self, chunk_ids: list[str], window: int) -> list[dict]:
        if not chunk_ids:
            return []
        return self._run(
            """
            UNWIND $chunk_ids AS center_id
            MATCH (center:Chunk {chunk_id: center_id})
            MATCH (neighbor:Chunk {doc_name: center.doc_name})
            WHERE neighbor.chunk_index >= center.chunk_index - $window
              AND neighbor.chunk_index <= center.chunk_index + $window
            RETURN center.chunk_id AS center_id,
                   neighbor.chunk_id AS chunk_id,
                   neighbor.doc_name AS doc_name,
                   neighbor.page_num AS page_num,
                   neighbor.chunk_index AS chunk_index,
                   neighbor.text AS text
            ORDER BY center.chunk_id, neighbor.chunk_index
            """,
            chunk_ids=chunk_ids,
            window=window,
        )

    def counts(self) -> dict[str, int]:
        rows = self._run(
            """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count
            ORDER BY label
            """
        )
        return {str(row["label"]): int(row["count"]) for row in rows}
