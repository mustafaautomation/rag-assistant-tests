from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.corpus import chunk_markdown, load_chunks
from rag_assistant.text import content_tokens, numbers, sentences, stem


def test_corpus_loads_every_article_as_paragraph_chunks() -> None:
    chunks = load_chunks()
    docs = {c.doc_id for c in chunks}
    assert len(docs) == 12
    assert "readme" not in {d.lower() for d in docs}
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.text and "\n" not in c.text for c in chunks)


def test_chunk_ids_and_titles() -> None:
    chunks = chunk_markdown("doc", "# Title here\n\nFirst para\nwraps.\n\nSecond para.")
    assert [c.id for c in chunks] == ["doc#0", "doc#1"]
    assert chunks[0].text == "First para wraps."
    assert chunks[1].embed_text == "Title here. Second para."


def test_article_without_title_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Title"):
        chunk_markdown("bad", "no heading")
    with pytest.raises(ValueError, match="no articles"):
        load_chunks(tmp_path)


def test_tokens_drop_stopwords_and_stem() -> None:
    assert content_tokens("How do I unlock the bikes?") == ["unlock", "bike"]
    assert stem("passes") == "pass" == stem("pass")
    assert stem("boxes") == "box"
    assert stem("batteries") == "battery"
    # known limitation of a tiny stemmer: past tense and plural do not always meet
    assert stem("charged") == "charg" != stem("charges")
    assert stem("bus") == "bus"


def test_numbers_normalise_trailing_zeros() -> None:
    assert numbers("It costs 9.00, then 2.50 for 30 minutes") == {"9", "2.5", "30"}
    assert numbers("no digits") == set()


def test_sentences_split_on_terminal_punctuation_before_capitals() -> None:
    assert sentences("One. Two costs 2.50. three stays.") == [
        "One.",
        "Two costs 2.50. three stays.",
    ]
