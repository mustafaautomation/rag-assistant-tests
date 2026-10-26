"""Retrieval and answer evaluation against the hand-labelled sets in evals/."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .assistant import Assistant

EVALS_DIR = Path(__file__).resolve().parents[2] / "evals"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{n}: {e}") from e
    return rows


def load_thresholds(path: Path = EVALS_DIR / "thresholds.json") -> dict[str, float]:
    data: dict[str, float] = json.loads(path.read_text(encoding="utf-8"))
    return data


@dataclass
class RetrievalReport:
    k: int
    questions: int
    recall_at_k: float  # mean over questions of |relevant docs in top k| / |relevant docs|
    hit_rate_at_k: float  # share of questions with at least one relevant doc in top k
    mrr: float  # mean reciprocal rank of the first relevant doc (0 if none in top k)
    misses: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_retrieval(assistant: Assistant, rows: list[dict[str, Any]], k: int) -> RetrievalReport:
    if not rows:
        raise ValueError("empty retrieval set")
    known_docs = {c.doc_id for c in assistant.chunks}
    recall_sum = hit_sum = rr_sum = 0.0
    misses = []
    for row in rows:
        relevant = set(row["relevant"])
        unknown = relevant - known_docs
        if unknown:
            raise ValueError(f"{row['id']}: labelled with unknown article(s) {sorted(unknown)}")
        # rank documents by their best chunk, keep the first k distinct documents
        docs: list[str] = []
        for hit in assistant.retrieve(row["question"], k=len(assistant.chunks)):
            if hit.chunk.doc_id not in docs:
                docs.append(hit.chunk.doc_id)
            if len(docs) == k:
                break
        found = relevant & set(docs)
        recall_sum += len(found) / len(relevant)
        hit_sum += 1.0 if found else 0.0
        first = next((i for i, d in enumerate(docs) if d in relevant), None)
        rr_sum += 0.0 if first is None else 1.0 / (first + 1)
        if len(found) < len(relevant):
            misses.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "top": docs,
                    "relevant": sorted(relevant),
                }
            )
    n = len(rows)
    return RetrievalReport(k, n, recall_sum / n, hit_sum / n, rr_sum / n, misses)


@dataclass
class AnswerReport:
    questions: int
    answerable_correct: float  # answerable questions answered with every must_include string
    refusal_correct: float  # unanswerable questions refused
    false_refusals: int
    grounded_rate: float  # answers (incl. refusals) that pass check_grounding
    failures: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_answers(assistant: Assistant, rows: list[dict[str, Any]]) -> AnswerReport:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    if not answerable or not unanswerable:
        raise ValueError("answer set needs answerable and unanswerable questions")
    correct = refused_ok = false_refusals = grounded = 0
    failures = []
    for row in rows:
        result = assistant.ask(row["question"])
        grounded += result.grounding.grounded
        if not result.grounding.grounded:
            failures.append(
                {"id": row["id"], "why": "ungrounded", "problems": result.grounding.problems}
            )
        if row["answerable"]:
            if result.answer.refused:
                false_refusals += 1
                failures.append({"id": row["id"], "why": "refused an answerable question"})
            elif all(s in result.answer.text for s in row["must_include"]):
                correct += 1
            else:
                failures.append(
                    {
                        "id": row["id"],
                        "why": "missing expected facts",
                        "expected": row["must_include"],
                        "answer": result.answer.text,
                    }
                )
        elif result.answer.refused:
            refused_ok += 1
        else:
            failures.append(
                {
                    "id": row["id"],
                    "why": "answered an unanswerable question",
                    "answer": result.answer.text,
                }
            )
    return AnswerReport(
        questions=len(rows),
        answerable_correct=correct / len(answerable),
        refusal_correct=refused_ok / len(unanswerable),
        false_refusals=false_refusals,
        grounded_rate=grounded / len(rows),
        failures=failures,
    )


def gate(report: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """Returns the list of metrics below their threshold (empty means pass)."""
    failed = []
    for metric, minimum in thresholds.items():
        if metric in report and report[metric] < minimum:
            failed.append(f"{metric} {report[metric]:.3f} < {minimum:.3f}")
    return failed
