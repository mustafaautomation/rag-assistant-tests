# rag-assistant-tests

[![CI](https://github.com/mustafaautomation/rag-assistant-tests/actions/workflows/ci.yml/badge.svg)](https://github.com/mustafaautomation/rag-assistant-tests/actions/workflows/ci.yml)

A small retrieval-augmented (RAG) assistant with the tests I would want before trusting one: retrieval recall on a hand-labelled set, answer grounding checks, refusal checks, and gates that are proven to fail. Personal open-source project, built with AI coding agents.

CI runs fully offline. No model provider is called and no API key exists in CI.

## What is in it

```
corpus/      12 short help articles for "Harbourline Bikes", a fictional bike-share (self-written)
evals/       hand-written, hand-labelled retrieval and answer sets + CI thresholds
src/rag_assistant/
  corpus.py     markdown -> paragraph chunks with ids like pricing#1
  embed.py      HashingEmbedder (deterministic TF-IDF feature hashing), RandomEmbedder (for negative tests)
  store.py      in-memory cosine store;  pgstore.py: pgvector store (HNSW, cosine)
  answer.py     ExtractiveAnswerer (deterministic, default) and an optional OpenAI-compatible provider
  grounding.py  per-sentence grounding checks
  evals.py      recall@k, hit rate, MRR, answer accuracy, refusal accuracy, grounded rate, gate()
  cli.py        rag ask | eval-retrieval | eval-answers
```

The corpus is invented on purpose: a model cannot answer from memory, and every fact can be checked against one file.

## Example

```
$ uv run rag ask "What does the day pass cost?"
A day pass costs 9.00 and covers unlimited rides of up to 45 minutes each for 24 hours from the first unlock [pricing#1]. Rides longer than 45 minutes on a day pass are charged 1.00 per extra 15 minutes [pricing#1].

grounded: True
retrieved: pricing#1 (0.41), pricing#0 (0.14), group-rides#2 (0.13), refunds#1 (0.12)

$ uv run rag ask "Does Harbourline operate in Paris?"
I can't answer that from the Harbourline help articles.
```

## Grounding checks

For every sentence of a non-refusal answer (`grounding.py`):

- it must cite a passage, e.g. `[pricing#1]`;
- every cited passage must be one that was actually retrieved for this question;
- every number in the sentence must appear in a cited passage, which catches invented prices and times;
- at least 70% of its content words must appear in the cited passages.

Tests feed the checker deliberately bad answers: an invented price, an uncited sentence, a citation to a passage that was not retrieved, and unsupported wording behind a valid citation. Each one must be flagged. A mocked LLM provider that hallucinates a price is also caught.

## Results of the default offline pipeline

Measured locally and gated in CI (`evals/thresholds.json`):

| Metric | Measured | Gate |
|---|---|---|
| recall@3 (42 retrieval questions) | 0.976 | ≥ 0.93 |
| MRR (top 3) | 0.897 | ≥ 0.85 |
| answerable questions answered with the expected facts (16) | 0.938 (15/16) | ≥ 0.90 |
| unanswerable questions refused (6) | 0.833 (5/6) | ≥ 0.80 |
| answers passing the grounding check (22) | 1.000 | = 1.00 |

Known failures, left visible instead of tuned away:

- **r42** "Do I need a password to log in?": the account article says "sign in" and "never asks for a password", but the hashing embedder ranks other articles higher.
- **a09** "minimum battery level" picks the wrong e-bike sentence (the article says "less than 20 percent").
- **u06** "pizza place near Central Quay" is answered from the accessibility article because it shares the station name. A lexical retriever cannot tell that the question is out of scope.

The hashing embedder is lexical, not semantic. Its numbers are a baseline for this tiny corpus, not a claim about RAG quality in general.

## The gates are proven to fail

- A `RandomEmbedder` retriever fails the recall gate (`test_a_broken_retriever_fails_the_gate`).
- An answerer that always states a made-up number fails the grounding and refusal gates.
- In CI, the same retrieval gate is run at k=1 and the job requires it to fail.
- `rag ask` exits 1 when the answer is not grounded.

## pgvector

`PgVectorStore` keeps chunks in Postgres with an HNSW `vector_cosine_ops` index. A CI job against `pgvector/pgvector:0.8.7-pg17` checks that, for every retrieval question, pgvector returns the same top 3 chunks and scores as the in-memory store. The job fails if that test is skipped.

## Optional real model

The `--llm` flag swaps in any OpenAI-compatible `/chat/completions` endpoint. The prompt requires a citation per sentence and an exact refusal string, and the same grounding checks apply to its output.

```bash
export RAG_LLM_BASE_URL=https://api.openai.com/v1 RAG_LLM_API_KEY=... RAG_LLM_MODEL=...
uv run rag ask --llm "What does the day pass cost?"
uv run rag eval-answers --llm
```

Keys are read from the environment only. This path is unit-tested with a mocked HTTP transport, but CI never calls a real provider, so I make no claim about any particular model's scores.

## Run it

```bash
uv sync
uv run pytest --cov                 # all offline tests, coverage gate 90%
uv run rag --k 3 eval-retrieval
uv run rag eval-answers
# pgvector (needs a Postgres with the vector extension):
uv sync --extra pgvector
PGVECTOR_URL=postgresql://postgres:postgres@127.0.0.1:5432/postgres uv run pytest -m pgvector
```

## CI

Every job must pass: **checks** (ruff, ruff format, mypy --strict, pytest with a 90% coverage gate), **evals** (retrieval and answer gates, plus the expected failure at k=1, with JSON reports as artifacts), **pgvector**, and **secrets** (gitleaks over the full history).

## License

MIT. The corpus and evaluation sets are original to this repository.
