"""Tokenising helpers shared by the embedder, the extractive answerer and the grounding checker."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    (
        "a an and are as at be been but by can could do does for from get got had has have "
        "how i if in into is it its me my no not of on or our so than that the their them "
        "then there these they this to too up us was we what when where which who why will "
        "with would you your yours about after all also am any because before being both did "
        "each few more most much must only other over same should some such through under "
        "until very while i'm i've don't can't "
    ).split()
)

WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def stem(word: str) -> str:
    """A deliberately tiny suffix stripper; enough to match 'bikes'/'bike', 'passes'/'pass'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    for suffix in ("ing", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)]
    return word


def words(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("\u2019", "'"))


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stemmed tokens without stopwords. Numbers are kept as written."""
    out = []
    for w in words(text):
        if w in STOPWORDS:
            continue
        out.append(w if NUMBER.fullmatch(w) else stem(w))
    return out


def numbers(text: str) -> set[str]:
    """Numbers in text, normalised so '9', '9.0' and '9.00' compare equal."""
    return {_norm_number(n) for n in NUMBER.findall(text)}


def _norm_number(n: str) -> str:
    return n.rstrip("0").rstrip(".") if "." in n else n


SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]
