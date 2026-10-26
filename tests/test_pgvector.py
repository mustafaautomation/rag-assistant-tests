"""Same retrieval through pgvector must match the in-memory store. Needs PGVECTOR_URL."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.pgvector
URL = os.environ.get("PGVECTOR_URL")


@pytest.mark.skipif(not URL, reason="PGVECTOR_URL not set")
def test_pgvector_returns_the_same_top_k_as_memory() -> None:
    from rag_assistant.assistant import Assistant
    from rag_assistant.evals import EVALS_DIR, evaluate_retrieval, load_jsonl
    from rag_assistant.pgstore import PgVectorStore

    assert URL
    mem = Assistant.default()
    store = PgVectorStore(URL, dim=mem.embedder.dim)
    try:
        pg = Assistant(mem.chunks, mem.embedder, store, mem.answerer)
        rows = load_jsonl(EVALS_DIR / "retrieval_set.jsonl")
        for row in rows:
            a = [h.chunk.id for h in mem.retrieve(row["question"], 3)]
            b = [h.chunk.id for h in pg.retrieve(row["question"], 3)]
            assert a == b, row["question"]
            sa = [round(h.score, 4) for h in mem.retrieve(row["question"], 3)]
            sb = [round(h.score, 4) for h in pg.retrieve(row["question"], 3)]
            assert sa == pytest.approx(sb, abs=1e-3)
        assert (
            evaluate_retrieval(pg, rows, 3).recall_at_k
            == evaluate_retrieval(mem, rows, 3).recall_at_k
        )
        assert store.search(mem.embedder.embed(["x"])[0], 0) == []
        with pytest.raises(ValueError):
            store.add(mem.chunks[:2], mem.embedder.embed(["one"]))
    finally:
        store.close()
