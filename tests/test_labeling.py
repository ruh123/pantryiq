"""Labeling CLI: persistence, resume, search, and the anti-drift guards."""
import json

import duckdb
import pytest

from pantryiq.er.labeling import (
    NO_MATCH,
    append_label,
    candidates_for,
    load_labels,
    load_sample,
    make_label,
    search_foods,
    verify_sample,
)


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
        "rank INTEGER)"
    )
    connection.execute("CREATE TABLE silver.distinct_ingredient_strings (normalized_text VARCHAR)")
    for fdc_id, description, kcal, flag in [
        ("1", "Egg, whole, raw, fresh", 143.0, False),
        ("2", "Eggnog", 88.0, False),
        ("3", "Babyfood, egg yolk", None, True),
        ("4", "Cheese, cheddar", 403.0, False),
    ]:
        connection.execute("INSERT INTO silver.usda_foods VALUES (?,?,?,?)",
                           [fdc_id, description, kcal, flag])
    for rank, fdc_id in enumerate(["2", "1", "3"]):
        connection.execute("INSERT INTO silver.ingredient_candidates VALUES ('egg',?,?)",
                           [fdc_id, rank])
    connection.execute("INSERT INTO silver.distinct_ingredient_strings VALUES ('egg')")
    return connection


@pytest.fixture
def gold(tmp_path):
    """A sample directory whose manifest matches its sample."""
    from pantryiq.er.gold import holdout_fingerprint

    out = tmp_path / "gold"
    out.mkdir()
    rows = [
        {"normalized_text": "egg", "occurrence_count": 9, "frequency_stratum": "head",
         "split": "tune", "example_lines": ["2 eggs"]},
    ]
    (out / "sample.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "manifest.json").write_text(
        json.dumps({"holdout_fingerprint": holdout_fingerprint(rows)})
    )
    return out


def test_candidates_come_back_in_rank_order(con):
    rows = candidates_for(con, "egg")
    assert [r[1] for r in rows] == ["Eggnog", "Egg, whole, raw, fresh", "Babyfood, egg yolk"]
    assert rows[0][4] == 0  # rank travels with the row so labels can record it


def test_search_finds_foods_outside_the_candidate_list(con):
    """Without this, recall@k would be 100% by construction."""
    results = search_foods(con, "cheddar")
    assert [r[1] for r in results] == ["Cheese, cheddar"]


def test_search_requires_all_tokens(con):
    assert search_foods(con, "egg whole") == [("1", "Egg, whole, raw, fresh", 143.0, False, None)]
    assert search_foods(con, "egg nonexistent") == []


def test_search_is_generic_first(con):
    """Shortest description first, matching the base-form labeling convention."""
    assert [r[1] for r in search_foods(con, "egg")][0] == "Eggnog"


def test_search_ignores_blank_query(con):
    assert search_foods(con, "   ") == []


def test_labels_round_trip_and_resume(tmp_path):
    path = tmp_path / "labels.jsonl"
    append_label(path, make_label("egg", "1", "candidate", 1))
    append_label(path, make_label("love", NO_MATCH, "candidate"))

    labels = load_labels(path)
    assert labels["egg"]["fdc_id"] == "1"
    assert labels["egg"]["candidate_rank"] == 1
    assert labels["love"]["fdc_id"] == NO_MATCH
    assert load_labels(tmp_path / "absent.jsonl") == {}


def test_a_relabel_supersedes_the_earlier_judgment(tmp_path):
    path = tmp_path / "labels.jsonl"
    append_label(path, make_label("egg", "2", "candidate", 0))
    append_label(path, make_label("egg", "1", "search"))

    assert load_labels(path)["egg"]["fdc_id"] == "1"


def test_verify_sample_accepts_a_matching_sample(con, gold):
    verify_sample(load_sample(gold), con, gold)


def test_verify_sample_rejects_a_moved_holdout(con, gold):
    (gold / "manifest.json").write_text(json.dumps({"holdout_fingerprint": "tampered"}))

    with pytest.raises(SystemExit, match="frozen holdout moved"):
        verify_sample(load_sample(gold), con, gold)


def test_verify_sample_rejects_strings_orphaned_by_a_parser_change(con, gold):
    """The failure this guards: normalization changes, and every label silently detaches."""
    con.execute("DELETE FROM silver.distinct_ingredient_strings")

    with pytest.raises(SystemExit, match="Normalization changed after sampling"):
        verify_sample(load_sample(gold), con, gold)
