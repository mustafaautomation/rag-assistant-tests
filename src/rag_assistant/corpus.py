"""Loads markdown articles and splits them into paragraph chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "corpus"


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#<n>"
    doc_id: str
    title: str
    text: str

    @property
    def embed_text(self) -> str:
        # The title carries the topic for short paragraphs ("It costs 9.00").
        return f"{self.title}. {self.text}"


def load_chunks(corpus_dir: Path = DEFAULT_CORPUS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
    if not chunks:
        raise ValueError(f"no articles found in {corpus_dir}")
    return chunks


def chunk_markdown(doc_id: str, text: str) -> list[Chunk]:
    lines = text.strip().splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{doc_id}: an article must start with a '# Title' line")
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:])
    paragraphs = [" ".join(p.split()) for p in body.split("\n\n") if p.strip()]
    return [Chunk(f"{doc_id}#{i}", doc_id, title, p) for i, p in enumerate(paragraphs)]
