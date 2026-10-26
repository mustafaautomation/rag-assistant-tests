"""pgvector-backed store (optional extra: `pip install .[pgvector]`)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from .corpus import Chunk
from .store import Hit


class PgVectorStore:
    """Stores chunks in a table with an HNSW cosine index and searches with `<=>`."""

    def __init__(self, conninfo: str, dim: int, table: str = "rag_chunks") -> None:
        self.dim = dim
        self.table = sql.Identifier(table)
        self.conn = psycopg.connect(conninfo, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self.table))
        self.conn.execute(
            sql.SQL(
                "CREATE TABLE {} (id text PRIMARY KEY, doc_id text NOT NULL, title text NOT NULL,"
                " body text NOT NULL, embedding vector({}) NOT NULL)"
            ).format(self.table, sql.Literal(dim))
        )
        self.conn.execute(
            sql.SQL("CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops)").format(
                self.table
            )
        )

    def add(self, chunks: Sequence[Chunk], vectors: npt.NDArray[np.float32]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self.conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {} (id, doc_id, title, body, embedding)"
                    " VALUES (%s, %s, %s, %s, %s)"
                ).format(self.table),
                [
                    (c.id, c.doc_id, c.title, c.text, v)
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def search(self, query: npt.NDArray[np.float32], k: int) -> list[Hit]:
        if k <= 0:
            return []
        rows = self.conn.execute(
            sql.SQL(
                "SELECT id, doc_id, title, body, 1 - (embedding <=> %s) AS score FROM {}"
                " ORDER BY embedding <=> %s, id LIMIT %s"
            ).format(self.table),
            (query, query, k),
        ).fetchall()
        return [Hit(Chunk(r[0], r[1], r[2], r[3]), float(r[4])) for r in rows]

    def close(self) -> None:
        self.conn.close()
