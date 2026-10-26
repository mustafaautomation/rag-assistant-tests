"""Checks that an answer is supported by the passages it was generated from.

For each sentence of a non-refusal answer:
  * it must cite at least one passage, as [doc#n];
  * every cited passage must be one that was actually retrieved for this question;
  * every number in the sentence must appear in a cited passage (catches invented prices);
  * at least `min_support` of its content tokens must appear in the cited passages.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .answer import REFUSAL
from .store import Hit
from .text import content_tokens, numbers, sentences

CITATION = re.compile(r"\[([a-z0-9-]+#\d+)\]")


TRAILING = re.compile(r"([.!?])((?:\s*\[[a-z0-9-]+#\d+\])+)")


def normalise_citations(text: str) -> str:
    """'Text. [a#1] [b#2]' -> 'Text [a#1] [b#2].' so a citation stays with its sentence."""
    return TRAILING.sub(lambda m: m.group(2) + m.group(1), text)


def citations(text: str) -> list[str]:
    return CITATION.findall(text)


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[str]
    problems: list[str] = field(default_factory=list)
    support: float = 0.0


@dataclass
class GroundingReport:
    grounded: bool
    refused: bool
    sentences: list[SentenceCheck]

    @property
    def problems(self) -> list[str]:
        return [f"{s.sentence!r}: {p}" for s in self.sentences for p in s.problems]


def check_grounding(answer: str, hits: Sequence[Hit], min_support: float = 0.7) -> GroundingReport:
    text = answer.strip()
    if text == REFUSAL:
        return GroundingReport(True, True, [])
    retrieved = {h.chunk.id: h.chunk.embed_text for h in hits}
    checks: list[SentenceCheck] = []
    for raw in sentences(normalise_citations(text)):
        cited = citations(raw)
        claim = CITATION.sub("", raw).strip()
        check = SentenceCheck(claim, cited)
        if not cited:
            check.problems.append("no citation")
        unknown = [c for c in cited if c not in retrieved]
        if unknown:
            check.problems.append(f"cites passages that were not retrieved: {', '.join(unknown)}")
        source = " ".join(retrieved[c] for c in cited if c in retrieved)
        missing_numbers = sorted(numbers(claim) - numbers(source))
        if missing_numbers:
            check.problems.append(f"numbers not in cited passages: {', '.join(missing_numbers)}")
        toks = content_tokens(claim)
        if toks:
            src = set(content_tokens(source))
            check.support = sum(t in src for t in toks) / len(toks)
            if check.support < min_support:
                check.problems.append(f"only {check.support:.0%} of the wording is supported")
        checks.append(check)
    if not checks:
        return GroundingReport(False, False, [SentenceCheck("", [], ["empty answer"])])
    return GroundingReport(all(not c.problems for c in checks), False, checks)
