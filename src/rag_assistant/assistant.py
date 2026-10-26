"""Ties corpus, embedder, store and answerer together."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .answer import Answer, Answerer, ExtractiveAnswerer
from .corpus import DEFAULT_CORPUS, Chunk, load_chunks
from .embed import Embedder, HashingEmbedder
from .grounding import GroundingReport, check_grounding
from .store import Hit, InMemoryStore, VectorStore


@dataclass
class Result:
    question: str
    hits: list[Hit]
    answer: Answer
    grounding: GroundingReport


class Assistant:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        store: VectorStore,
        answerer: Answerer,
        k: int = 4,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.store = store
        self.answerer = answerer
        self.k = k
        self.store.add(self.chunks, self.embedder.embed([c.embed_text for c in self.chunks]))

    @classmethod
    def default(cls, corpus_dir: Path = DEFAULT_CORPUS, k: int = 4) -> Assistant:
        chunks = load_chunks(corpus_dir)
        embedder = HashingEmbedder().fit([c.embed_text for c in chunks])
        return cls(chunks, embedder, InMemoryStore(), ExtractiveAnswerer(), k=k)

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        query = self.embedder.embed([question])[0]
        return self.store.search(query, k or self.k)

    def ask(self, question: str) -> Result:
        hits = self.retrieve(question)
        answer = self.answerer.answer(question, hits)
        return Result(question, hits, answer, check_grounding(answer.text, hits))
