"""Blind relabel: the draw is frozen, the gold labels are never touched, agreement is credited
the same way scoring credits duplicate USDA descriptions."""
import json

import duckdb
import pytest

from pantryiq.er.labeling import NO_MATCH, append_label, label_rows, load_labels, make_label
from pantryiq.er.relabel import (
    agreement,
    draw,
    ensure_manifest,
    paths,
    population,
    report,
    rows_to_judge,
    same_entity,
    wilson,
)

DESCRIPTIONS = {
    "1": "Egg, whole, raw, fresh",
    "2": "Eggnog",
    "9": "Egg, whole, raw, fresh",  # the Foundation/SR Legacy twin of "1"
}


@pytest.fixture
def con(tmp_path):
    connection = duckdb.connect(str(tmp_path / "t.duckdb"))
    connection.execute("CREATE SCHEMA silver")
    connection.execute(
        "CREATE TABLE silver.usda_foods (fdc_id VARCHAR, description_raw VARCHAR, "
        "kcal_per_100g DOUBLE, is_deprioritized BOOLEAN)"
    )
    connection.execute(
        "CREATE TABLE silver.ingredient_candidates (normalized_text VARCHAR, fdc_id VARCHAR, "
        "rank INTEGER, cosine_search DOUBLE, cosine_desc DOUBLE)"
    )
    for fdc_id, description in DESCRIPTIONS.items():
        connection.execute("INSERT INTO silver.usda_foods VALUES (?,?,?,?)",
                           [fdc_id, description, 143.0, False])
    for rank, (fdc_id, cosine) in enumerate([("1", 0.80), ("2", 0.70)]):
        connection.execute("INSERT INTO silver.ingredient_candidates VALUES ('egg',?,?,?,?)",
                           [fdc_id, rank, cosine, cosine])
    return connection


def labels_of(**by_text) -> dict[str, dict]:
    """{'egg': ('1', 'claude')} -> label records."""
    return {text: {"normalized_text": text, "fdc_id": fdc_id, "labeler": labeler}
            for text, (fdc_id, labeler) in by_text.items()}


def test_draw_only_takes_annotator_labeled_strings():
    """Human-labeled rows are already ground truth — relabeling them measures nothing."""
    labels = labels_of(egg=("1", "claude"), flour=("2", "human"), sugar=("1", "claude"))

    assert population(labels) == ["egg", "sugar"]
    assert draw(labels, size=2) == ["egg", "sugar"]


def test_draw_reproduces_from_the_seed_regardless_of_file_order():
    """The draw has to be checkable by someone who suspects it was re-rolled."""
    forward = labels_of(**{name: ("1", "claude") for name in "abcdefgh"})
    reversed_order = dict(reversed(list(forward.items())))

    assert draw(forward, size=3, seed=7) == draw(reversed_order, size=3, seed=7)
    assert draw(forward, size=3, seed=7) != draw(forward, size=3, seed=8)


def test_the_manifest_is_frozen_once_written(tmp_path):
    """A re-rollable draw is a re-rollable result — the first draw is the only draw."""
    path = tmp_path / "labels.jsonl"
    for name in "abcdefgh":
        append_label(path, {**make_label(name, "1", "candidate"), "labeler": "claude"})

    first = ensure_manifest(tmp_path)
    for name in "ijklmnop":  # the population changes underneath it
        append_label(path, {**make_label(name, "2", "candidate"), "labeler": "claude"})

    assert ensure_manifest(tmp_path)["strings"] == first["strings"]
    assert first["population_fingerprint"] == json.loads(
        (tmp_path / "relabel_manifest.json").read_text())["population_fingerprint"]


def test_a_later_pass_never_redraws_a_string_an_earlier_pass_saw(tmp_path):
    """A string judged twice is not a fresh measurement — pass 2 exists to be out-of-sample."""
    path = tmp_path / "labels.jsonl"
    for name in "abcdefghijklmnop":
        append_label(path, {**make_label(name, "1", "candidate"), "labeler": "claude"})

    first = ensure_manifest(tmp_path, number=1)
    second = ensure_manifest(tmp_path, number=2)

    assert set(first["strings"]).isdisjoint(second["strings"])
    assert second["excluded_as_already_drawn"] == len(first["strings"])


def test_each_pass_records_the_guide_it_was_judged_under(tmp_path):
    """Pass 1 measured against a guide with no tiebreak for the case most disagreement fell
    into; comparing the two rates is only meaningful if that is on the record."""
    path = tmp_path / "labels.jsonl"
    for name in "abcdefghijklmnop":
        append_label(path, {**make_label(name, "1", "candidate"), "labeler": "claude"})

    assert ensure_manifest(tmp_path, 1)["guide"] != ensure_manifest(tmp_path, 2)["guide"]


def test_reporting_survives_a_manifest_frozen_before_guide_was_recorded(tmp_path):
    """Pass 1's manifest predates the `guide` field. Reading it must not crash, and the frozen
    file must not be rewritten to add one."""
    # Built and closed here rather than via the `con` fixture: report() opens the database
    # read-only, which conflicts with a read-write handle on the same file in-process.
    setup = duckdb.connect(str(tmp_path / "t.duckdb"))
    setup.execute("CREATE SCHEMA silver")
    setup.execute("CREATE TABLE silver.usda_foods (fdc_id VARCHAR, description_raw VARCHAR)")
    setup.execute("INSERT INTO silver.usda_foods VALUES ('1','Egg, whole, raw'),('2','Eggnog')")
    setup.close()

    (tmp_path / "relabel_manifest.json").write_text(json.dumps({"size": 1, "strings": ["egg"]}))
    append_label(tmp_path / "labels.jsonl", {**make_label("egg", "1", "candidate"),
                                             "labeler": "claude"})
    append_label(tmp_path / "relabel.jsonl", {**make_label("egg", "2", "candidate"),
                                              "labeler": "human"})
    before = (tmp_path / "relabel_manifest.json").read_bytes()

    report(tmp_path, tmp_path / "t.duckdb", number=1)

    assert (tmp_path / "relabel_manifest.json").read_bytes() == before


def test_pass_one_keeps_its_original_filenames(tmp_path):
    """Pass 1's manifest and judgments are committed — renaming them orphans the record."""
    assert paths(tmp_path, 1) == (tmp_path / "relabel_manifest.json", tmp_path / "relabel.jsonl")
    assert paths(tmp_path, 2) == (tmp_path / "relabel_manifest2.json", tmp_path / "relabel2.jsonl")


def test_rows_to_judge_carries_no_gold_label():
    """Blindness is structural: the judging loop is handed sample rows, never label records."""
    sample = [{"normalized_text": "egg", "frequency_stratum": "head", "occurrence_count": 9,
               "example_lines": ["2 eggs"]}]

    rows = rows_to_judge(sample, {"strings": ["egg"]}, done={})

    assert rows == sample
    assert "fdc_id" not in rows[0]


def test_rows_to_judge_skips_what_is_already_relabeled(tmp_path):
    sample = [{"normalized_text": "egg"}, {"normalized_text": "flour"}]

    rows = rows_to_judge(sample, {"strings": ["egg", "flour"]}, done={"egg": {}})

    assert [row["normalized_text"] for row in rows] == ["flour"]


def test_relabeling_never_touches_the_gold_labels(con, tmp_path, monkeypatch):
    """The gold set is irreplaceable judgment; this pass measures agreement, it does not edit."""
    gold = tmp_path / "labels.jsonl"
    append_label(gold, {**make_label("egg", "2", "candidate"), "labeler": "claude"})
    before = gold.read_bytes()
    relabel = tmp_path / "relabel.jsonl"
    rows = [{"normalized_text": "egg", "frequency_stratum": "head", "occurrence_count": 9,
             "example_lines": ["2 eggs"]}]
    monkeypatch.setattr("builtins.input", lambda *_: "")  # ENTER accepts the top candidate

    quit_early = label_rows(con, rows, relabel, extra={"labeler": "human", "pass": "blind"})

    assert quit_early is False
    assert gold.read_bytes() == before
    judged = load_labels(relabel)["egg"]
    assert judged["fdc_id"] == "1" and judged["labeler"] == "human"


def test_agreement_credits_the_duplicate_description(con):
    """94 descriptions exist twice; picking the other twin is not a disagreement."""
    assert same_entity("1", "9", DESCRIPTIONS)
    assert not same_entity("1", "2", DESCRIPTIONS)


def test_no_match_agrees_only_with_no_match():
    assert same_entity("no-match", "no-match", DESCRIPTIONS)
    assert not same_entity("no-match", "1", DESCRIPTIONS)
    assert not same_entity("1", "no-match", DESCRIPTIONS)


def test_agreement_counts_only_strings_judged_twice():
    original = labels_of(egg=("1", "claude"), flour=("2", "claude"))
    relabeled = labels_of(egg=("9", "human"), flour=("1", "human"), salt=("1", "human"))

    agreed, compared, disagreements = agreement(original, relabeled, DESCRIPTIONS)

    assert (agreed, compared) == (1, 2)  # egg agrees via the twin; salt was never in gold
    assert disagreements == [("flour", "2", "1")]


def test_wilson_brackets_the_estimate_and_stays_in_bounds():
    low, high = wilson(27, 30)
    assert low < 27 / 30 < high
    assert (0.0, 0.0) == wilson(0, 0)

    perfect_low, perfect_high = wilson(30, 30)
    assert perfect_high == 1.0
    assert perfect_low < 1.0  # 30/30 is not proof of 100% — the interval must stay open


def test_two_ids_missing_from_the_description_map_do_not_count_as_agreeing():
    """`first is not None and first == second` — drop the None guard and two unknown ids both
    map to None, so `None == None` reports agreement between two entities nobody can name. This
    is the agreement primitive behind the relabel rate, the ensemble, and the entity headline,
    so a false True there inflates all three."""
    assert same_entity("known-a", "known-b", {"known-a": "Butter", "known-b": "Butter"})
    assert not same_entity("ghost-a", "ghost-b", {})
    assert not same_entity("ghost-a", "ghost-b", {"other": "Butter"})


def test_no_match_never_agrees_with_an_entity_but_does_with_itself():
    """The null class is a real judgment, not a missing one: two annotators both saying "no USDA
    entity exists" agree, while "no-match" against any entity is a genuine disagreement."""
    description = {"1": "Butter", "2": "Butter"}

    assert same_entity(NO_MATCH, NO_MATCH, description)
    assert not same_entity(NO_MATCH, "1", description)
    assert not same_entity("1", NO_MATCH, description)
