"""Embedders. HashingEmbedder is deterministic and offline, and is what CI uses."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .text import content_tokens

Vector = npt.NDArray[np.float32]


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]: ...


def _features(text: str) -> list[str]:
    toks = content_tokens(text)
    return toks + [f"{a}_{b}" for a, b in pairwise(toks)]


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    n = int.from_bytes(h, "little")
    return n % dim, (1.0 if (n >> 63) & 1 else -1.0)


class HashingEmbedder:
    """Signed feature hashing of stemmed unigrams and bigrams with TF-IDF weights.

    It is lexical, not semantic: it finds paraphrases only when they share word stems. That is
    the honest baseline the retrieval gate is calibrated against. Call fit() on the corpus to
    learn IDF weights; without it every term weighs the same.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.idf: dict[str, float] = {}
        self._default_idf = 1.0

    def fit(self, texts: Sequence[str]) -> HashingEmbedder:
        df: Counter[str] = Counter()
        for t in texts:
            df.update(set(_features(t)))
        n = len(texts)
        self.idf = {f: math.log((1 + n) / (1 + c)) + 1.0 for f, c in df.items()}
        self._default_idf = math.log(1 + n) + 1.0
        return self

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, count in Counter(_features(text)).items():
                idx, sign = _bucket(feat, self.dim)
                weight = (1.0 + math.log(count)) * self.idf.get(feat, self._default_idf)
                out[row, idx] += sign * weight
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class RandomEmbedder:
    """Deterministic noise. Exists only to prove the retrieval gate fails a broken retriever."""

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        rows = []
        for t in texts:
            digest = hashlib.sha256(f"{self.seed}:{t}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            v = rng.standard_normal(self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)
