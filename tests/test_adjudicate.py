"""Blind adjudication: the A/B order is randomized but reproducible, the verdict decodes back
to the right pass, and neither label file is touched."""
import json

from pantryiq.er.adjudicate import (
    AMBIGUOUS,
    NO_MATCH_DISPLAY,
    load_verdicts,
    make_verdict,
    option_text,
    order_for,
    pending,
)
from pantryiq.er.labeling import append_label, make_label

DESCRIPTIONS = {"1": "Milk, canned, evaporated", "2": "Milk, canned, evaporated, nonfat"}


def test_the_ab_order_is_randomized_but_reproducible():
    """Fixed order would let the labeler learn 'a is always the gold label' within a few rows."""
    orders = {text: order_for(text) for text in
              ("evaporated milk", "cider", "tortilla", "strawberry", "spiral pasta", "meat")}

    assert len(set(orders.values())) == 2, "every string got the same side"
    assert orders == {text: order_for(text) for text in orders}, "not reproducible"


def test_the_verdict_decodes_to_the_pass_that_produced_it():
    """The record must survive re-derivation: picking the displayed side the annotator holds
    has to resolve to 'claude' whichever side that was."""
    for text in ("evaporated milk", "cider", "tortilla", "strawberry"):
        annotator_side = "a" if order_for(text) else "b"
        human_side = "b" if order_for(text) else "a"

        assert make_verdict(text, "1", "2", annotator_side)["verdict"] == "claude"
        assert make_verdict(text, "1", "2", human_side)["verdict"] == "human"


def test_the_record_keeps_the_mapping_for_audit():
    record = make_verdict("evaporated milk", "1", "2", "a")

    assert record["shown_first"] in ("claude", "human")
    assert (record["shown_first"] == "claude") == order_for("evaporated milk")
    assert record["claude_fdc_id"] == "1" and record["human_fdc_id"] == "2"


def test_ambiguous_records_no_position():
    record = make_verdict("cider", "1", "2", AMBIGUOUS)

    assert record["verdict"] == AMBIGUOUS
    assert record["chose_position"] is None


def test_option_text_hides_the_id_while_judging_but_shows_energy():
    """The fdc_id is a tell — the gold labels are keyed by it. Energy is decision-relevant."""
    shown = option_text("1", DESCRIPTIONS, {"1": 134.0})

    assert "134 kcal/100g" in shown
    assert "(1)" not in shown
    assert option_text("no-match", DESCRIPTIONS, {}) == NO_MATCH_DISPLAY


def test_pending_skips_what_is_already_settled(tmp_path):
    append_label(tmp_path / "labels.jsonl", make_label("evaporated milk", "1", "candidate"))
    append_label(tmp_path / "labels.jsonl", make_label("cider", "1", "candidate"))
    append_label(tmp_path / "relabel.jsonl", make_label("evaporated milk", "2", "candidate"))
    append_label(tmp_path / "relabel.jsonl", make_label("cider", "2", "candidate"))
    (tmp_path / "adjudication.jsonl").write_text(
        json.dumps(make_verdict("cider", "1", "2", "a")) + "\n")

    assert [row[0] for row in pending(tmp_path, DESCRIPTIONS)] == ["evaporated milk"]


def test_adjudicating_never_touches_the_label_files(tmp_path):
    labels, relabels = tmp_path / "labels.jsonl", tmp_path / "relabel.jsonl"
    append_label(labels, make_label("evaporated milk", "1", "candidate"))
    append_label(relabels, make_label("evaporated milk", "2", "candidate"))
    before = (labels.read_bytes(), relabels.read_bytes())

    path = tmp_path / "adjudication.jsonl"
    path.write_text(json.dumps(make_verdict("evaporated milk", "1", "2", "b")) + "\n")

    assert (labels.read_bytes(), relabels.read_bytes()) == before
    assert load_verdicts(path)["evaporated milk"]["verdict"] in ("claude", "human")
    assert load_verdicts(tmp_path / "absent.jsonl") == {}
