from __future__ import annotations

import json

import httpx
import pytest

from rag_assistant.answer import REFUSAL, ExtractiveAnswerer, OpenAICompatibleAnswerer
from rag_assistant.assistant import Assistant
from rag_assistant.grounding import check_grounding


def test_extractive_answer_quotes_the_source_with_citations(assistant: Assistant) -> None:
    res = assistant.ask("How long is the one-time sign-in code valid?")
    assert "10 minutes" in res.answer.text
    assert res.answer.cited and all(c.startswith("account#") for c in res.answer.cited)
    assert res.grounding.grounded, res.grounding.problems


def test_out_of_scope_question_is_refused(assistant: Assistant) -> None:
    res = assistant.ask("What is the capital of Australia?")
    assert res.answer.refused and res.answer.text == REFUSAL


def test_low_retrieval_score_refuses_before_reading_passages(assistant: Assistant) -> None:
    strict = ExtractiveAnswerer(min_score=0.99)
    assert strict.answer("day pass price", assistant.retrieve("day pass price")).refused
    assert ExtractiveAnswerer().answer("anything", []).refused


def _mock_llm(reply: str, seen: list[dict[str, object]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"auth": request.headers["authorization"], "body": json.loads(request.content)})
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compatible_provider_sends_context_and_parses_citations(
    assistant: Assistant,
) -> None:
    hits = assistant.retrieve("What does a day pass cost?")
    cid = next(h.chunk.id for h in hits if "day pass costs 9.00" in h.chunk.text)
    seen: list[dict[str, object]] = []
    llm = OpenAICompatibleAnswerer(
        "https://llm.example/v1/",
        "test-key",
        "some-model",
        client=_mock_llm(f"It costs 9.00 [{cid}].", seen),
    )
    ans = llm.answer("What does a day pass cost?", hits)
    assert ans.cited == (cid,) and not ans.refused
    assert check_grounding(ans.text, hits).grounded
    body = seen[0]["body"]
    assert isinstance(body, dict)
    assert seen[0]["auth"] == "Bearer test-key"
    assert body["temperature"] == 0 and cid in body["messages"][1]["content"]


def test_grounding_catches_a_hallucinating_provider(assistant: Assistant) -> None:
    hits = assistant.retrieve("What does a day pass cost?")
    cid = hits[0].chunk.id
    llm = OpenAICompatibleAnswerer(
        "https://x/v1", "k", "m", client=_mock_llm(f"It costs 4.99 [{cid}].", [])
    )
    assert not check_grounding(llm.answer("What does a day pass cost?", hits).text, hits).grounded


def test_provider_refusal_is_recognised(assistant: Assistant) -> None:
    llm = OpenAICompatibleAnswerer("https://x/v1", "k", "m", client=_mock_llm(REFUSAL, []))
    assert llm.answer("q", assistant.retrieve("q")).refused


def test_provider_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("RAG_LLM_BASE_URL", "RAG_LLM_API_KEY", "RAG_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="RAG_LLM_BASE_URL"):
        OpenAICompatibleAnswerer()
