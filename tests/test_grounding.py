"""The grounding checker must flag the ways a generated answer typically goes wrong."""

from __future__ import annotations

import pytest

from rag_assistant.answer import REFUSAL
from rag_assistant.assistant import Assistant
from rag_assistant.grounding import check_grounding, citations, normalise_citations
from rag_assistant.store import Hit


@pytest.fixture(scope="module")
def hits(assistant: Assistant) -> list[Hit]:
    return assistant.retrieve("What does a day pass cost?")


def pricing_id(hits: list[Hit]) -> str:
    return next(h.chunk.id for h in hits if "day pass costs 9.00" in h.chunk.text)


def test_a_faithful_cited_answer_is_grounded(hits: list[Hit]) -> None:
    cid = pricing_id(hits)
    r = check_grounding(f"A day pass costs 9.00 [{cid}].", hits)
    assert r.grounded and not r.refused, r.problems


def test_citation_after_the_full_stop_is_accepted(hits: list[Hit]) -> None:
    cid = pricing_id(hits)
    assert (
        normalise_citations(f"A day pass costs 9.00. [{cid}]") == f"A day pass costs 9.00 [{cid}]."
    )
    assert check_grounding(f"A day pass costs 9.00. [{cid}]", hits).grounded


def test_an_invented_price_is_flagged(hits: list[Hit]) -> None:
    cid = pricing_id(hits)
    r = check_grounding(f"A day pass costs 12.00 [{cid}].", hits)
    assert not r.grounded
    assert any("numbers not in cited passages: 12" in p for p in r.problems)


def test_an_uncited_sentence_is_flagged(hits: list[Hit]) -> None:
    cid = pricing_id(hits)
    r = check_grounding(f"A day pass costs 9.00 [{cid}]. It is the best deal in town.", hits)
    assert not r.grounded
    assert any("no citation" in p for p in r.problems)


def test_citing_a_passage_that_was_not_retrieved_is_flagged(hits: list[Hit]) -> None:
    r = check_grounding("Riders must be at least 16 years old [safety#0].", hits)
    assert any("not retrieved" in p for p in r.problems)


def test_unsupported_wording_is_flagged_even_with_a_valid_citation(hits: list[Hit]) -> None:
    cid = pricing_id(hits)
    r = check_grounding(f"Dogs ride free on weekends with a picnic basket [{cid}].", hits)
    assert not r.grounded
    assert any("wording is supported" in p for p in r.problems)


def test_refusal_and_empty_answers(hits: list[Hit]) -> None:
    assert check_grounding(REFUSAL, hits).refused
    assert not check_grounding("   ", hits).grounded


def test_citation_parser() -> None:
    assert citations("a [pricing#1] b [group-rides#0] c [x]") == ["pricing#1", "group-rides#0"]
