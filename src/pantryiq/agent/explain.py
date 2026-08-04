"""The quality explainer — a plain-English runbook entry for a failure, drafted by the agent.

Design doc §97 and the Phase-4 gate (§124): *triggered by a failed dbt test or low-confidence
resolution; the agent reads failure metadata and recent logs and drafts a plain-English summary,
stored in a runbook table.* This is the one part of Phase 4 that serves operators rather than
users, and it is the same machinery pointed the other way.

**Two triggers, both read from artifacts that already exist:**

- `target/run_results.json` — dbt's structured result for every model, test and seed in the last
  `dbt build`. Statuses `fail`, `error` and `warn` become findings; `pass` and `success` do not.
- `silver.ingredient_entity_map` — 2,727 of 9,163 strings abstained and 157 more were flagged.
  Ranked by **occurrence**, not by cosine: a string the resolver was unsure about that appears
  4,399 times matters more than one that appears once, and an operator's attention is the scarce
  resource this table exists to direct.

**The explainer is guardrailed exactly like a user-facing answer.** An ops summary that invents a
row count is worse than no summary — it will be believed, and acted on, by someone who does not
have the query in front of them. Every numeric fact is assembled first, the model may only quote
from that ledger, and a summary that fails the check is rejected in favour of the facts
themselves. `AnswerContext.computed` is the ledger: it was built for the week planner's verified
totals and does the same job here.

**Percentages are precomputed into the ledger.** The model will naturally write "2,727 of 9,163
(29.8%)" — correctly — and a guardrail that only knew the two counts would reject a true
sentence. Deriving the percentage is the sort of arithmetic the model is otherwise forbidden, so
it is done in code and handed over as a fact.

**Runbook rows live beside the query log**, in `data/agent_log.duckdb`, for the reason §4.6 gives:
`gold/publish.py` swaps Gold wholesale inside a transaction and the serving copy is read-only, so
an operational table cannot live there without being destroyed by a rebuild.

Run:  uv run python -m pantryiq.agent.explain
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from pantryiq.agent.claude import MODEL, get_client, injection_paragraph, text_of
from pantryiq.agent.context import AnswerContext
from pantryiq.agent.guardrail import Verdict, check
from pantryiq.agent.log import DEFAULT_DB as LOG_DB
from pantryiq.agent.retrieval import PantryQuery

RUN_RESULTS = Path("target/run_results.json")
WAREHOUSE = Path("data/pantryiq.duckdb")
# dbt statuses that mean something needs a human. `pass` and `success` are the other four.
FAILING = ("fail", "error", "warn", "runtime error")
MAX_TOKENS = 2000

RUNBOOK_SCHEMA = """
CREATE TABLE IF NOT EXISTS quality_runbook (
    entry_id               VARCHAR PRIMARY KEY,
    created_at             TIMESTAMPTZ NOT NULL,
    trigger                VARCHAR NOT NULL,   -- dbt_test | low_confidence
    subject                VARCHAR NOT NULL,
    severity               VARCHAR NOT NULL,
    facts                  VARCHAR NOT NULL,   -- JSON: the numeric ledger the summary may quote
    detail                 VARCHAR NOT NULL,   -- JSON: the non-numeric metadata
    summary                VARCHAR,
    guardrail_pass         BOOLEAN NOT NULL,
    numeric_claims_checked INTEGER NOT NULL
)
"""

SYSTEM = (
    "You write runbook entries for the data engineer who will be woken up by this. They know "
    "the system; they do not have the query in front of them.\n\n"
    "Say, in this order and in plain prose: what broke, what it means for the data that is "
    "published, and what to look at first. Three or four sentences. No headings, no bullet "
    "lists, no restating the metadata back.\n\n"
    "Rules:\n"
    "- Every number you write must appear in <facts>. Do not compute new ones, including "
    "percentages — the ones worth quoting are already there.\n"
    "- Do not guess at a cause the metadata does not support. \"The test does not say which "
    "rows failed\" is a useful sentence; an invented reason is not.\n"
    "- Do not invent remediation steps that assume tooling you were not told about. Point at "
    "the table, model or string that is implicated.\n"
    "- If this is a low-confidence resolution rather than a failure, say plainly that nothing "
    "is broken — it is a data-quality item ranked by how often the string occurs.\n\n"
    + injection_paragraph("facts", "failure metadata")
)


@dataclass(frozen=True)
class Finding:
    """One thing worth a runbook entry, with its numeric ledger separated from its prose."""

    trigger: str
    subject: str
    severity: str
    # (label, value) — the ONLY numbers the summary may state. See the module docstring.
    facts: tuple[tuple[str, float], ...] = ()
    detail: dict = field(default_factory=dict)
    # Names the summary will quote whose digits are not measurements — the test's `unique_id`,
    # the column `kcal_per_100g`, an ingredient string like "2% milk".
    identifiers: tuple[str, ...] = ()

    def context(self) -> AnswerContext:
        """The guardrail contract. No recipes — this is the ledger, not a recipe answer."""
        return AnswerContext(question=f"{self.trigger}: {self.subject}",
                             query=PantryQuery(), recipes=(), computed=self.facts,
                             identifiers=self.identifiers)

    def prompt(self) -> str:
        lines = [f"  <trigger>{self.trigger}</trigger>",
                 f"  <subject>{self.subject}</subject>",
                 f"  <severity>{self.severity}</severity>"]
        lines += [f"  <{key}>{value}</{key}>" for key, value in self.detail.items()]
        lines += [f"  <{label}>{value:,}</{label}>" for label, value in self.facts]
        return "<facts>\n" + "\n".join(lines) + "\n</facts>"


def test_arguments(manifest_path: Path) -> dict[str, dict]:
    """Each test's configured kwargs, keyed by `unique_id`, from dbt's manifest.

    The bounds an `accepted_range` test enforces are real facts about the failure and belong in
    the ledger — without them the model writes "outside the configured bounds (0 to 910)", which
    is correct and gets rejected. Read from the manifest rather than parsed out of the mangled
    test name (`..._kcal_per_100g__910__0.b67c7d28cb`), which is not a data structure.
    """
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {uid: (node.get("test_metadata") or {}).get("kwargs") or {}
            for uid, node in manifest.get("nodes", {}).items()
            if node.get("test_metadata")}


def dbt_failures(path: Path | str = RUN_RESULTS,
                 manifest_path: Path | str | None = None) -> list[Finding]:
    """Findings from the last `dbt build`. Empty when the gate was green, which is the norm."""
    path = Path(path)
    if not path.exists():
        return []
    manifest_path = Path(manifest_path) if manifest_path else path.with_name("manifest.json")
    arguments = test_arguments(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    # The propagation count is what tells an operator nothing reached Gold: `dbt build` skips
    # everything downstream of a failed test, and the publish step only swaps on a clean gate.
    # Without this the entry says "a test failed" and leaves the blast radius to be guessed at.
    skipped = sum(1 for result in results if result.get("status") == "skipped")

    findings = []
    for result in results:
        if result.get("status") not in FAILING:
            continue
        failures = result.get("failures")
        facts = [("execution_time_seconds", round(result.get("execution_time") or 0.0, 2)),
                 ("downstream_models_skipped", skipped)]
        if failures is not None:
            facts.insert(0, ("failing_rows", int(failures)))

        kwargs = arguments.get(result["unique_id"], {})
        for name in ("min_value", "max_value"):
            if isinstance(kwargs.get(name), (int, float)) and not isinstance(kwargs[name], bool):
                facts.append((f"configured_{name}", kwargs[name]))
        column = kwargs.get("column_name") or ""
        relation = result.get("relation_name") or "unknown"

        findings.append(Finding(
            trigger="dbt_test",
            subject=result["unique_id"],
            severity="error" if result["status"] in ("error", "runtime error") else "fail",
            facts=tuple(facts),
            detail={"status": result["status"],
                    "relation": relation,
                    "column": column or "not a column test",
                    "message": (result.get("message") or "no message").strip()[:400],
                    "dbt_version": payload.get("metadata", {}).get("dbt_version", "unknown"),
                    "built_at": payload.get("metadata", {}).get("generated_at", "unknown")},
            identifiers=tuple(name for name in (result["unique_id"], relation, column) if name),
        ))
    return findings


def low_confidence_findings(db_path: Path | str = WAREHOUSE, limit: int = 3) -> list[Finding]:
    """The strings the resolver declined, ranked by how many recipe lines they affect."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        total, undecided = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE abstained OR flagged) "
            "FROM silver.ingredient_entity_map").fetchone()
        rows = con.execute("""
            SELECT normalized_text, occurrence_count, frequency_stratum,
                   cosine, confidence, description_raw, abstained, flagged
            FROM silver.ingredient_entity_map
            WHERE abstained OR flagged
            ORDER BY occurrence_count DESC, normalized_text
            LIMIT ?
        """, [limit]).fetchall()
    finally:
        con.close()

    findings = []
    for text, occurrences, stratum, cosine, confidence, best, abstained, flagged in rows:
        findings.append(Finding(
            trigger="low_confidence",
            subject=text,
            severity="warn",
            # The percentage is computed here because the model is forbidden from computing it.
            facts=(("recipe_lines_affected", int(occurrences)),
                   ("best_candidate_cosine", round(cosine or 0.0, 3)),
                   ("fitted_confidence", round(confidence or 0.0, 3)),
                   ("undecided_strings", int(undecided)),
                   ("distinct_strings_total", int(total)),
                   ("undecided_percent", round(100 * undecided / total, 1))),
            detail={"frequency_stratum": stratum,
                    "best_candidate": best or "none",
                    "outcome": "abstained" if abstained else "flagged for review"},
            # Ingredient strings and USDA descriptions carry digits that are not measurements —
            # "2% milk", "Milk, whole, 3.25% milkfat".
            identifiers=tuple(name for name in (text, best) if name),
        ))
    return findings


def explain(finding: Finding, client=None) -> tuple[str, Verdict]:
    """One Claude call producing a runbook paragraph, checked against the finding's ledger."""
    client = client or get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": finding.prompt()}],
        output_config={"effort": "low"},
    )
    summary = text_of(response).strip()
    return summary, check(summary, finding.context())


def facts_only(finding: Finding) -> str:
    """The fallback when the drafted summary quotes something that is not in the ledger.

    A runbook entry that degrades to a list of measurements is still useful. One that degrades
    to a confident wrong sentence is worse than nothing, because it will be acted on.
    """
    stated = "; ".join(f"{label.replace('_', ' ')}: {value:,}" for label, value in finding.facts)
    return (f"[drafted summary rejected by the guardrail — facts only] "
            f"{finding.trigger} on {finding.subject} ({finding.severity}). {stated}.")


def write_runbook(finding: Finding, summary: str, verdict: Verdict,
                  db_path: Path | str = LOG_DB) -> str:
    """Append one runbook row and return its entry_id."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    entry_id = str(uuid.uuid4())
    con = duckdb.connect(str(db_path))
    try:
        con.execute(RUNBOOK_SCHEMA)
        con.execute("INSERT INTO quality_runbook VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [entry_id, datetime.now(timezone.utc), finding.trigger, finding.subject,
                     finding.severity, json.dumps(dict(finding.facts)),
                     json.dumps(finding.detail), summary,
                     verdict.passed, verdict.checked])
    finally:
        con.close()
    return entry_id


def run(run_results: Path | str = RUN_RESULTS, db_path: Path | str = WAREHOUSE,
        client=None) -> list[tuple[Finding, str, Verdict]]:
    """Draft an entry for every current finding, guardrail each, and record them."""
    findings = dbt_failures(run_results) + low_confidence_findings(db_path)
    if not findings:
        return []
    client = client or get_client()
    written = []
    for finding in findings:
        summary, verdict = explain(finding, client)
        if not verdict.passed:
            summary = facts_only(finding)
            verdict = check(summary, finding.context())
        write_runbook(finding, summary, verdict)
        written.append((finding, summary, verdict))
    return written


def main() -> None:
    import sys

    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN_RESULTS
    failures = dbt_failures(results_path)
    print(f"dbt findings from {results_path}: {len(failures)}"
          f"{'  (gate is green — the normal state)' if not failures else ''}")

    entries = run(results_path)
    for finding, summary, verdict in entries:
        print(f"\n{'=' * 78}\n[{finding.trigger}] {finding.subject}  ({finding.severity})")
        print(f"  guardrail: {'PASS' if verdict.passed else 'FAIL'} on {verdict.checked} "
              f"numeric claims\n")
        print(f"  {summary}")

    print(f"\n{'=' * 78}\n{len(entries)} runbook entries written to {LOG_DB}")
    print("  Guardrailed like a user answer: an ops summary that invents a row count is worse")
    print("  than no summary, because someone will act on it without the query in front of them.")


if __name__ == "__main__":
    main()
