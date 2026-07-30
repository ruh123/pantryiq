"""The 2.6 viability comparison. The reference standard is the whole ballgame here, so that is
what these tests pin."""
import json

from pantryiq.er.adjudicator_value import ADJUDICATOR, compare, paired_rows

DESCRIPTIONS = {"1": "Milk, whole", "2": "Milk, nonfat", "3": "Cream, heavy"}
KCAL = {"1": 61.0, "2": 34.0, "3": 340.0}


def rows_for(texts):
    """Resolver always picks '1', adjudicator always picks '2', gold says '1'."""
    return [{"text": text, "gold": "1", "resolver": "1", "adjudicator": "2",
             "confidence": 0.9} for text in texts]


def test_compare_scores_against_the_reference_not_the_gold_label(capsys):
    """The bug this guards: filtering rows to those that HAVE a human judgment does not change
    what they are scored against. An earlier version scored every section against `gold` and
    printed a self-consistency figure under the heading 'the only valid comparison'."""
    rows = rows_for(["egg"])

    compare(rows, {"egg": "1"}, DESCRIPTIONS, KCAL, "vs gold")
    against_gold = capsys.readouterr().out
    compare(rows, {"egg": "2"}, DESCRIPTIONS, KCAL, "vs human")
    against_human = capsys.readouterr().out

    # Same rows, opposite verdicts — because the reference flipped.
    assert "resolver 100.0%  adjudicator   0.0%" in against_gold
    assert "resolver   0.0%  adjudicator 100.0%" in against_human


def test_compare_drops_rows_the_reference_does_not_cover():
    """A reference with no entry for a string cannot score it; silently treating a miss as a
    failure would understate whichever arm happened to be right."""
    rows = rows_for(["egg", "flour", "salt"])

    # Only 'egg' is covered, so n must be 1 — asserted via the printed count.
    import io
    import contextlib
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        compare(rows, {"egg": "1"}, DESCRIPTIONS, KCAL, "partial")

    assert "(n=1 strings" in buffer.getvalue()


def test_compare_handles_an_empty_comparison(capsys):
    compare([], {"egg": "1"}, DESCRIPTIONS, KCAL, "nothing")

    assert "no comparable strings" in capsys.readouterr().out


def test_paired_rows_skips_no_match_and_unjudged_strings(tmp_path, monkeypatch):
    """no-match strings have no entity to compare, and a string the adjudicator never saw would
    otherwise be scored against a missing judgment."""
    (tmp_path / "sample.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"normalized_text": "egg", "split": "tune"},
        {"normalized_text": "bisquick", "split": "tune"},
        {"normalized_text": "unjudged", "split": "tune"},
        {"normalized_text": "held", "split": "holdout"},
    ]) + "\n")
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"normalized_text": "egg", "fdc_id": "1"},
        {"normalized_text": "bisquick", "fdc_id": "no-match"},
        {"normalized_text": "unjudged", "fdc_id": "1"},
        {"normalized_text": "held", "fdc_id": "1"},
    ]) + "\n")
    (tmp_path / "ensemble.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"normalized_text": "egg", "annotator": ADJUDICATOR, "fdc_id": "2"},
        {"normalized_text": "bisquick", "annotator": ADJUDICATOR, "fdc_id": "2"},
        {"normalized_text": "held", "annotator": ADJUDICATOR, "fdc_id": "2"},
    ]) + "\n")
    monkeypatch.setattr("pantryiq.er.adjudicator_value.resolve",
                        lambda *a, **k: [("egg", "1", 0.9, 0.8, False),
                                         ("bisquick", "1", 0.9, 0.8, False),
                                         ("unjudged", "1", 0.9, 0.8, False)])

    rows = paired_rows(tmp_path, tmp_path / "unused.duckdb")

    assert [row["text"] for row in rows] == ["egg"]  # no-match, unjudged and holdout all dropped
