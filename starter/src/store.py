"""Persist and search chunks with PostgreSQL and pgvector."""
import json
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("Thiếu DATABASE_URL trong .env.")
        if dimensions < 1:
            raise ValueError("dimensions phải lớn hơn 0.")

        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Create the pgvector extension and chunks table."""
        with psycopg.connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS day06_chunks (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding VECTOR({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS day06_chunks_metadata_idx
                ON day06_chunks USING GIN (metadata)
                """
            )

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Replace a strategy's stored chunks with the current corpus."""
        if len(chunks) != len(vectors):
            raise ValueError("Số chunk phải bằng số vector.")

        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Vector có {len(vector)} chiều; cần {self.dimensions}."
                )

        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                "DELETE FROM day06_chunks WHERE strategy = %s",
                (strategy,),
            )

            rows = []
            for chunk, vector in zip(chunks, vectors):
                vector_literal = "[" + ",".join(map(str, vector)) + "]"
                rows.append((
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    Jsonb(chunk.metadata),
                    vector_literal,
                ))

            if rows:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        f"""
                        INSERT INTO day06_chunks (
                            strategy, chunk_id, document_id, source, page,
                            text, metadata, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector({self.dimensions}))
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source = EXCLUDED.source,
                            page = EXCLUDED.page,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        rows,
                    )

    def stats(self, strategy: str) -> dict[str, int]:
        """Row and distinct-ID counts, used to prove re-ingestion adds no duplicates."""
        with psycopg.connect(self.database_url) as connection:
            rows, distinct_ids = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT chunk_id) "
                "FROM day06_chunks WHERE strategy = %s",
                (strategy,),
            ).fetchone()
        return {"rows": rows, "distinct_chunk_ids": distinct_ids}

    def versions(self) -> dict[str, str]:
        with psycopg.connect(self.database_url) as connection:
            postgres = connection.execute("SHOW server_version").fetchone()[0]
            row = connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
        return {"postgresql": postgres, "pgvector": row[0] if row else ""}

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks, optionally constrained by JSON metadata."""
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector có {len(query_vector)} chiều; "
                f"cần {self.dimensions}."
            )
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0.")

        vector_literal = "[" + ",".join(map(str, query_vector)) + "]"
        filter_json = json.dumps(filters or {}, ensure_ascii=False)

        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, document_id, source, page, text, metadata,
                       1 - (embedding <=> %s::vector({self.dimensions})) AS score
                FROM day06_chunks
                WHERE strategy = %s
                  AND metadata @> %s::jsonb
                ORDER BY embedding <=> %s::vector({self.dimensions})
                LIMIT %s
                """,
                (
                    vector_literal,
                    strategy,
                    filter_json,
                    vector_literal,
                    top_k,
                ),
            ).fetchall()

        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]
