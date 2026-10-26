"""Answer generators. Both return text in which every sentence ends with [chunk-id] citations."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from .store import Hit
from .text import content_tokens, sentences

REFUSAL = "I can't answer that from the Harbourline help articles."


@dataclass(frozen=True)
class Answer:
    text: str
    refused: bool
    cited: tuple[str, ...]


class Answerer(Protocol):
    def answer(self, question: str, hits: Sequence[Hit]) -> Answer: ...


def _cite(sentence: str, chunk_id: str) -> str:
    """Puts the citation inside the sentence: 'It costs 9.00 [pricing#1].'"""
    if sentence[-1:] in ".!?":
        return f"{sentence[:-1]} [{chunk_id}]{sentence[-1]}"
    return f"{sentence} [{chunk_id}]"


class ExtractiveAnswerer:
    """Deterministic: returns the best-matching sentences from the retrieved chunks, verbatim.

    It refuses when the best retrieval score is below min_score or when no sentence shares at
    least min_overlap content tokens with the question.
    """

    def __init__(
        self, min_score: float = 0.12, min_overlap: int = 2, max_sentences: int = 2
    ) -> None:
        self.min_score = min_score
        self.min_overlap = min_overlap
        self.max_sentences = max_sentences

    def answer(self, question: str, hits: Sequence[Hit]) -> Answer:
        if not hits or hits[0].score < self.min_score:
            return Answer(REFUSAL, True, ())
        q = set(content_tokens(question))
        scored: list[tuple[int, float, int, str, str]] = []
        for rank, hit in enumerate(hits):
            for s in sentences(hit.chunk.text):
                overlap = len(q & set(content_tokens(s)))
                scored.append((overlap, hit.score, -rank, s, hit.chunk.id))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        best = [t for t in scored if t[0] >= self.min_overlap][: self.max_sentences]
        if not best:
            return Answer(REFUSAL, True, ())
        text = " ".join(_cite(s, cid) for _, _, _, s, cid in best)
        return Answer(text, False, tuple(dict.fromkeys(cid for *_, cid in best)))


SYSTEM_PROMPT = (
    "You answer questions about Harbourline Bikes using only the numbered context passages. "
    "End every sentence with the id of the passage that supports it in square brackets, "
    "for example [pricing#1]. Copy prices and numbers exactly. If the passages do not contain "
    f"the answer, reply exactly: {REFUSAL}"
)


class OpenAICompatibleAnswerer:
    """Calls any OpenAI-compatible /chat/completions endpoint. Not used in CI.

    Configure with RAG_LLM_BASE_URL, RAG_LLM_API_KEY and RAG_LLM_MODEL. Keys come from the
    environment only; nothing is read from or written to files.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("RAG_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("RAG_LLM_API_KEY", "")
        self.model = model or os.environ.get("RAG_LLM_MODEL", "")
        if not (self.base_url and self.api_key and self.model):
            raise ValueError("set RAG_LLM_BASE_URL, RAG_LLM_API_KEY and RAG_LLM_MODEL")
        self.client = client or httpx.Client(timeout=60)

    def answer(self, question: str, hits: Sequence[Hit]) -> Answer:
        context = "\n\n".join(f"[{h.chunk.id}] {h.chunk.title}: {h.chunk.text}" for h in hits)
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            },
        )
        resp.raise_for_status()
        text = str(resp.json()["choices"][0]["message"]["content"]).strip()
        refused = text == REFUSAL
        from .grounding import citations  # local import avoids a cycle

        return Answer(text, refused, tuple(dict.fromkeys(citations(text))))
