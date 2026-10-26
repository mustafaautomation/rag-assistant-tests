"""Vector stores. InMemoryStore is used by default; PgVectorStore lives in pgstore.py."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .corpus import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher is closer


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: npt.NDArray[np.float32]) -> None: ...

    def search(self, query: npt.NDArray[np.float32], k: int) -> list[Hit]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: npt.NDArray[np.float32] | None = None

    def add(self, chunks: Sequence[Chunk], vectors: npt.NDArray[np.float32]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks.extend(chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query: npt.NDArray[np.float32], k: int) -> list[Hit]:
        if self._matrix is None or k <= 0:
            return []
        scores = self._matrix @ query
        # stable sort: equal scores keep corpus order, so results are reproducible
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self._chunks[i], float(scores[i])) for i in order]
