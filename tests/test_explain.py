"""The quality explainer — the runbook entry an operator gets woken up by.

The fixtures are REAL dbt artifacts, produced by injecting `prove_gate.py`'s impossible row and
capturing what dbt actually wrote. A hand-written failure would have agreed with whatever this
module expected to see; the point of the fixture is that it does not.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from pantryiq.agent.claude import Refused, injection_paragraph
from pantryiq.agent.explain import (
    SYSTEM,
    dbt_failures,
    explain,
    facts_only,
    low_confidence_findings,
    write_runbook,
)
from pantryiq.agent.guardrail import check

FIXTURES = Path(__file__).parent / "fixtures"
FAILED = FIXTURES / "run_results_failed.json"
WAREHOUSE = Path("data/pantryiq.duckdb")


def fake_client(text: str, stop_reason: str = "end_turn"):
    blocks = [] if stop_reason == "refusal" else [SimpleNamespace(type="text", text=text)]
    response = SimpleNamespace(stop_reason=stop_reason, content=blocks)
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))


@pytest.fixture
def failure():
    return dbt_failures(FAILED)[0]


# ------------------------------------------------------- reading the artifacts

def test_a_green_run_produces_no_findings(tmp_path):
    """The normal state. An explainer that fires on success is noise, and noise gets muted."""
    green = tmp_path / "run_results.json"
    green.write_text(json.dumps({"results": [
        {"unique_id": "test.a", "status": "pass", "execution_time": 0.1},
        {"unique_id": "model.b", "status": "success", "execution_time": 0.2},
    ], "metadata": {}}))

    assert dbt_failures(green) == []


def test_a_missing_artifact_is_not_an_error(tmp_path):
    """dbt may never have run in this checkout. That is not something to wake anyone for."""
    assert dbt_failures(tmp_path / "absent.json") == []


def test_the_real_failure_is_read_with_its_blast_radius(failure):
    """`failing_rows` alone says a test failed. `downstream_models_skipped` is what tells the
    operator nothing reached Gold — dbt skips everything downstream, and publish only swaps on
    a clean gate. Both numbers come from the captured artifact, not from this test."""
    facts = dict(failure.facts)

    assert failure.trigger == "dbt_test"
    assert failure.severity == "fail"
    assert facts["failing_rows"] == 1
    assert facts["downstream_models_skipped"] == 17


def test_the_configured_bounds_come_from_the_manifest_not_the_test_name(failure):
    """The model correctly writes "outside the configured bounds (0 to 910)" and would be
    rejected without these. Read from `test_metadata.kwargs`, because the name they also appear
    in (`..._kcal_per_100g__910__0.b67c7d28cb`) is a mangled string, not a data structure."""
    facts = dict(failure.facts)

    assert facts["configured_min_value"] == 0
    assert facts["configured_max_value"] == 910


def test_bounds_are_simply_absent_when_the_manifest_is_not_there(tmp_path):
    """The manifest is a separate artifact and may not have been written. Missing bounds cost a
    less specific summary, not a crash."""
    copied = tmp_path / "run_results.json"
    copied.write_text(FAILED.read_text())

    facts = dict(dbt_failures(copied).pop().facts)

    assert "configured_max_value" not in facts
    assert facts["failing_rows"] == 1


# ------------------------------------------------------- the guardrail contract

def test_a_column_name_containing_digits_is_not_a_numeric_claim(failure):
    """The bug this caught, and it rejected a perfectly good runbook entry. `kcal_per_100g` is
    the column that failed; quoting it is quoting the metadata, and reading its digits as a
    figure blocked the summary on the single most important trigger the explainer has."""
    text = ("An accepted_range test on canonical_ingredients.kcal_per_100g failed: "
            "1 row falls outside the bounds of 0 to 910, and 17 downstream models were skipped.")

    assert check(text, failure.context()).passed


def test_a_number_that_is_genuinely_absent_is_still_caught(failure):
    """The identifier allowance must not become a hole. 4,200 is in no ledger and no name."""
    verdict = check("The test found 4,200 bad rows.", failure.context())

    assert not verdict.passed
    assert verdict.unsupported == (4200.0,)


def test_the_metadata_is_fenced_as_untrusted():
    """Ingredient strings reaching this prompt come from the recipe corpus."""
    assert injection_paragraph("facts", "failure metadata") in SYSTEM
    assert "<facts>" in SYSTEM


def test_the_prompt_carries_the_facts_and_the_detail(failure):
    prompt = failure.prompt()

    assert "<failing_rows>1</failing_rows>" in prompt
    assert "<trigger>dbt_test</trigger>" in prompt
    assert "canonical_ingredients" in prompt


# --------------------------------------------------------------- degradation

def test_the_facts_only_fallback_passes_its_own_check(failure):
    """A runbook entry that degrades to a list of measurements is still useful. One that degrades
    to a confident wrong sentence is worse than nothing — it will be acted on."""
    text = facts_only(failure)

    assert check(text, failure.context()).passed
    assert "17" in text and "guardrail" in text


def test_a_refusal_is_raised_rather_than_written_to_the_runbook(failure):
    with pytest.raises(Refused):
        explain(failure, fake_client("", stop_reason="refusal"))


def test_a_summary_quoting_an_invented_number_fails_the_check(failure):
    _, verdict = explain(failure, fake_client("The test failed on 9,999 rows."))

    assert not verdict.passed


# ------------------------------------------------------------------ the runbook

def test_an_entry_is_written_with_its_verdict(tmp_path, failure):
    text = facts_only(failure)
    verdict = check(text, failure.context())

    entry_id = write_runbook(failure, text, verdict, db_path=tmp_path / "log.duckdb")

    con = duckdb.connect(str(tmp_path / "log.duckdb"), read_only=True)
    try:
        row = con.execute("SELECT entry_id, trigger, severity, guardrail_pass, facts "
                          "FROM quality_runbook").fetchone()
    finally:
        con.close()

    assert row[:4] == (entry_id, "dbt_test", "fail", True)
    assert json.loads(row[4])["downstream_models_skipped"] == 17


# ------------------------------------------------------- low-confidence trigger

@pytest.mark.skipif(not WAREHOUSE.exists(), reason="warehouse not present")
def test_low_confidence_findings_are_ranked_by_how_many_lines_they_affect():
    """Not by cosine. A string the resolver was unsure about that appears 1,811 times matters
    more than one appearing once, and operator attention is the scarce resource here."""
    findings = low_confidence_findings(limit=5)

    affected = [dict(finding.facts)["recipe_lines_affected"] for finding in findings]

    assert affected == sorted(affected, reverse=True)
    assert all(finding.trigger == "low_confidence" for finding in findings)


@pytest.mark.skipif(not WAREHOUSE.exists(), reason="warehouse not present")
def test_the_percentage_is_precomputed_because_the_model_may_not_compute_it():
    """The model will write "2,884 of 9,163 (31.5%)" — correctly. A ledger holding only the two
    counts would reject a true sentence, and deriving it is exactly the arithmetic the model is
    forbidden elsewhere."""
    facts = dict(low_confidence_findings(limit=1)[0].facts)

    expected = round(100 * facts["undecided_strings"] / facts["distinct_strings_total"], 1)

    assert facts["undecided_percent"] == expected
