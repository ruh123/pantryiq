"""The agent, end to end: parse -> retrieve -> generate -> guardrail -> log.

Four stages, and only two of them involve a model. That ordering is the design:

    parse      Claude, constrained to a fixed filter schema — it may shape the search
    retrieve   SQL over the exported Gold file — it may not
    generate   Claude, shown only the retrieved facts — it may phrase, not source
    guardrail  deterministic code — every number must trace back to what retrieval returned

Nothing streams to the user, deliberately. The guardrail needs a complete response before any of
it can be trusted, and showing text that is then retracted is worse than waiting for it.

Run:  uv run python -m pantryiq.agent  ["your question"]
"""
from __future__ import annotations

import sys
import time

from pantryiq.agent.claude import get_client
from pantryiq.agent.context import build
from pantryiq.agent.guardrail import guarded_answer
from pantryiq.agent.log import record, summary
from pantryiq.agent.parse import parse

DEMO = [
    "What can I make with chicken, rice and onions?",
    "I want a vegetarian dinner under 500 calories",
]


def ask(question: str, client=None) -> str:
    """One question, one checked answer, one log row."""
    client = client or get_client()
    started = time.perf_counter()

    query = parse(question, client)
    context = build(question, query)
    text, verdict, regenerated = guarded_answer(context, client)
    latency_ms = (time.perf_counter() - started) * 1000

    record(context, text, verdict, regenerated, latency_ms)
    return text


def main() -> None:
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO
    client = get_client()

    for question in questions:
        started = time.perf_counter()
        query = parse(question, client)
        context = build(question, query)
        text, verdict, regenerated = guarded_answer(context, client)
        latency_ms = (time.perf_counter() - started) * 1000
        query_id = record(context, text, verdict, regenerated, latency_ms)

        print(f"\n{'=' * 78}\nQ: {question}\n{'=' * 78}")
        print(f"   retrieved {len(context.recipes)} recipes | "
              f"{len(context.numbers())} quotable numbers | {latency_ms:,.0f} ms")
        print(f"   guardrail: {'PASS' if verdict.passed else 'FAIL'} on "
              f"{verdict.checked} numeric claims"
              f"{'  (regenerated)' if regenerated else ''}"
              f"{'  unsupported: ' + str(list(verdict.unsupported)) if verdict.unsupported else ''}")
        print(f"   logged as {query_id}\n")
        print(text)

    stats = summary()
    print(f"\n{'=' * 78}")
    print(f"agent_query_log: {stats['queries']:,} queries, "
          f"{stats['passed']:,} passed the guardrail first time, "
          f"{stats['claims_checked']:,} numeric claims checked in total.")


if __name__ == "__main__":
    main()
