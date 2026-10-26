"""The CI gates, plus proof that they fail when retrieval or answering is broken."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rag_assistant import cli
from rag_assistant.answer import Answer
from rag_assistant.assistant import Assistant
from rag_assistant.corpus import load_chunks
from rag_assistant.embed import RandomEmbedder
from rag_assistant.evals import (
    EVALS_DIR,
    evaluate_answers,
    evaluate_retrieval,
    gate,
    load_jsonl,
    load_thresholds,
)
from rag_assistant.store import Hit, InMemoryStore

RETRIEVAL = load_jsonl(EVALS_DIR / "retrieval_set.jsonl")
ANSWERS = load_jsonl(EVALS_DIR / "answer_set.jsonl")
THRESHOLDS = load_thresholds()


def test_eval_sets_are_well_formed_and_labelled_against_real_articles() -> None:
    docs = {c.doc_id for c in load_chunks()}
    assert len(RETRIEVAL) >= 40 and len({r["id"] for r in RETRIEVAL}) == len(RETRIEVAL)
    for r in RETRIEVAL:
        assert set(r["relevant"]) <= docs, r
    assert {r["answerable"] for r in ANSWERS} == {True, False}
    assert all(r["must_include"] for r in ANSWERS if r["answerable"])


def test_retrieval_meets_the_gate_at_k3(assistant: Assistant) -> None:
    rep = evaluate_retrieval(assistant, RETRIEVAL, k=3)
    assert gate(rep.as_dict(), THRESHOLDS) == [], rep.misses
    assert 0 <= rep.mrr <= rep.hit_rate_at_k <= 1


def test_recall_grows_with_k(assistant: Assistant) -> None:
    r1, r3, r5 = (evaluate_retrieval(assistant, RETRIEVAL, k).recall_at_k for k in (1, 3, 5))
    assert r1 <= r3 <= r5


def test_a_broken_retriever_fails_the_gate() -> None:
    chunks = load_chunks()
    broken = Assistant(chunks, RandomEmbedder(), InMemoryStore(), Assistant.default().answerer)
    failed = gate(evaluate_retrieval(broken, RETRIEVAL, k=3).as_dict(), THRESHOLDS)
    assert any(f.startswith("recall_at_k") for f in failed)


def test_answers_meet_the_gate(assistant: Assistant) -> None:
    rep = evaluate_answers(assistant, ANSWERS)
    assert gate(rep.as_dict(), THRESHOLDS) == [], rep.failures
    assert rep.grounded_rate == 1.0


class Fabricator:
    """Answers confidently with a made-up number, citing the top passage."""

    def answer(self, question: str, hits: Sequence[Hit]) -> Answer:
        return Answer(f"It is 999 [{hits[0].chunk.id}].", False, (hits[0].chunk.id,))


def test_a_fabricating_answerer_fails_grounding_and_refusal_gates() -> None:
    a = Assistant.default()
    a.answerer = Fabricator()
    failed = gate(evaluate_answers(a, ANSWERS).as_dict(), THRESHOLDS)
    assert any(f.startswith("grounded_rate") for f in failed)
    assert any(f.startswith("refusal_correct") for f in failed)


def test_bad_labels_and_sets_are_rejected(assistant: Assistant, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown article"):
        evaluate_retrieval(assistant, [{"id": "x", "question": "q", "relevant": ["nope"]}], 3)
    with pytest.raises(ValueError, match="empty"):
        evaluate_retrieval(assistant, [], 3)
    with pytest.raises(ValueError, match="answerable and unanswerable"):
        evaluate_answers(assistant, [r for r in ANSWERS if r["answerable"]])
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\n{oops\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_jsonl(bad)


def test_cli_eval_commands_write_reports_and_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "r.json"
    assert cli.main(["--k", "3", "eval-retrieval", "--json", str(out)]) == 0
    report: dict[str, Any] = json.loads(out.read_text())
    assert report["k"] == 3 and report["questions"] == len(RETRIEVAL)
    assert cli.main(["eval-answers", "--json", str(tmp_path / "a.json")]) == 0
    assert "answerable_correct" in capsys.readouterr().out


def test_cli_retrieval_gate_fails_with_k1(capsys: pytest.CaptureFixture[str]) -> None:
    # recall@1 is below the k=3 threshold, so the same gate must fail
    assert cli.main(["--k", "1", "eval-retrieval"]) == 1
    assert "GATE FAILED: recall_at_k" in capsys.readouterr().err


def test_cli_ask(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ask", "How many devices can I be signed in on?"]) == 0
    out = capsys.readouterr().out
    assert "three devices" in out and "grounded: True" in out


def test_cli_ask_exits_1_when_answer_is_ungrounded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = Assistant.default

    def fabricating(*args: Any, **kwargs: Any) -> Assistant:
        a = real(*args, **kwargs)
        a.answerer = Fabricator()
        return a

    monkeypatch.setattr(Assistant, "default", staticmethod(fabricating))
    assert cli.main(["ask", "What does a day pass cost?"]) == 1
    assert "numbers not in cited passages: 999" in capsys.readouterr().out


def test_cli_uses_a_custom_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "x.md").write_text("# Ferry\n\nThe ferry leaves every 20 minutes from Pier 4.")
    assert (
        cli.main(
            ["--corpus", str(tmp_path), "ask", "How often does the ferry leave from the pier?"]
        )
        == 0
    )
    assert "20 minutes" in capsys.readouterr().out
