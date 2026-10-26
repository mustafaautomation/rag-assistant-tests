from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag_assistant.corpus import Chunk
from rag_assistant.embed import HashingEmbedder, RandomEmbedder
from rag_assistant.store import InMemoryStore


@settings(max_examples=50, deadline=None)
@given(st.text(min_size=1, max_size=200))
def test_hashing_embedder_is_deterministic_and_unit_length(text: str) -> None:
    e = HashingEmbedder(dim=128)
    a, b = e.embed([text])[0], e.embed([text])[0]
    assert np.array_equal(a, b)
    norm = float(np.linalg.norm(a))
    assert norm == pytest.approx(1.0, abs=1e-5) or norm == 0.0


def test_similar_texts_score_higher_than_unrelated() -> None:
    e = HashingEmbedder().fit(["day pass price", "battery range e-bike", "delete account"])
    q, near, far = e.embed(["how much is a day pass", "day pass price", "delete account"])
    assert float(q @ near) > float(q @ far)


def test_idf_downweights_common_terms() -> None:
    docs = ["bike price", "bike battery", "bike account", "bike dock"]
    e = HashingEmbedder().fit(docs)
    assert e.idf["bike"] < e.idf["battery"]


def test_random_embedder_is_deterministic_noise() -> None:
    r = RandomEmbedder(dim=16)
    assert np.array_equal(r.embed(["x"]), r.embed(["x"]))
    assert not np.array_equal(r.embed(["x"]), r.embed(["y"]))
    assert r.embed([]).shape == (0, 16)


def test_store_returns_top_k_by_cosine_with_stable_ties() -> None:
    s = InMemoryStore()
    chunks = [Chunk(f"d#{i}", "d", "T", f"t{i}") for i in range(3)]
    vecs = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    s.add(chunks, vecs)
    hits = s.search(np.array([1, 0], dtype=np.float32), 2)
    assert [h.chunk.id for h in hits] == ["d#0", "d#2"]
    assert s.search(np.array([1, 0], dtype=np.float32), 0) == []
    assert InMemoryStore().search(np.array([1, 0], dtype=np.float32), 3) == []
    with pytest.raises(ValueError):
        s.add(chunks, vecs[:1])
