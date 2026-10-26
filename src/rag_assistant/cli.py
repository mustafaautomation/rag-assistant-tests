"""Command line: `rag ask`, `rag eval-retrieval`, `rag eval-answers`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assistant import Assistant
from .evals import (
    EVALS_DIR,
    evaluate_answers,
    evaluate_retrieval,
    gate,
    load_jsonl,
    load_thresholds,
)


def _build(args: argparse.Namespace) -> Assistant:
    assistant = (
        Assistant.default(Path(args.corpus), k=args.k)
        if args.corpus
        else Assistant.default(k=args.k)
    )
    if getattr(args, "llm", False):
        from .answer import OpenAICompatibleAnswerer

        assistant.answerer = OpenAICompatibleAnswerer()
    return assistant


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag", description=__doc__)
    p.add_argument("--corpus", help="directory of markdown articles (default: corpus/)")
    p.add_argument("--k", type=int, default=4, help="passages retrieved per question")
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--llm", action="store_true", help="use the OpenAI-compatible provider")
    r = sub.add_parser("eval-retrieval", help="recall@k on evals/retrieval_set.jsonl")
    r.add_argument("--json", type=Path, help="write the report here")
    a = sub.add_parser("eval-answers", help="answer accuracy, refusals and grounding")
    a.add_argument("--json", type=Path)
    a.add_argument("--llm", action="store_true")
    args = p.parse_args(argv)
    assistant = _build(args)

    if args.cmd == "ask":
        res = assistant.ask(args.question)
        print(res.answer.text)
        print(f"\ngrounded: {res.grounding.grounded}")
        for prob in res.grounding.problems:
            print(f"  - {prob}")
        print("retrieved: " + ", ".join(f"{h.chunk.id} ({h.score:.2f})" for h in res.hits))
        return 0 if res.grounding.grounded else 1

    thresholds = load_thresholds()
    if args.cmd == "eval-retrieval":
        rep = evaluate_retrieval(assistant, load_jsonl(EVALS_DIR / "retrieval_set.jsonl"), args.k)
        report = rep.as_dict()
        print(
            f"retrieval k={rep.k} n={rep.questions}: recall@k={rep.recall_at_k:.3f} "
            f"hit_rate@k={rep.hit_rate_at_k:.3f} mrr={rep.mrr:.3f}"
        )
        for m in rep.misses:
            print(f"  miss {m['id']}: {m['question']!r} top={m['top']} relevant={m['relevant']}")
        limits = {k: v for k, v in thresholds.items() if k in ("recall_at_k", "mrr")}
    else:
        arep = evaluate_answers(assistant, load_jsonl(EVALS_DIR / "answer_set.jsonl"))
        report = arep.as_dict()
        print(
            f"answers n={arep.questions}: answerable_correct={arep.answerable_correct:.3f} "
            f"refusal_correct={arep.refusal_correct:.3f} grounded_rate={arep.grounded_rate:.3f}"
        )
        for failure in arep.failures:
            print(f"  {failure}")
        limits = {
            k: v
            for k, v in thresholds.items()
            if k in ("answerable_correct", "refusal_correct", "grounded_rate")
        }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = gate(report, limits)
    for msg in failed:
        print(f"GATE FAILED: {msg}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
