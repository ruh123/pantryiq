"""Labeling CLI: persistence, resume, search, and the anti-drift guards."""
import json

import duckdb
import pytest

from pantryiq.er.labeling import (
    NO_MATCH,
    append_label,
    candidates_for,
    display_score,
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
        "rank INTEGER, cosine_search DOUBLE, cosine_desc DOUBLE)"
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
    # Generation order puts Eggnog first — exactly the cosine failure the display must undo.
    for rank, (fdc_id, cosine) in enumerate([("2", 0.77), ("1", 0.70), ("3", 0.66)]):
        connection.execute("INSERT INTO silver.ingredient_candidates VALUES ('egg',?,?,?,?)",
                           [fdc_id, rank, cosine, cosine])
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


def test_display_puts_the_base_form_first(con):
    """Raw cosine ranked Eggnog above the actual egg; the display order must not."""
    rows = candidates_for(con, "egg")
    assert rows[0][1] == "Egg, whole, raw, fresh"
    assert rows[0][4] == 1  # the GENERATION rank travels with the row, not the display position


def test_display_deduplicates_descriptions(con):
    """94 descriptions exist twice (Foundation + SR Legacy); duplicates waste display slots."""
    con.execute("INSERT INTO silver.usda_foods VALUES ('9','Eggnog',88.0,false)")
    con.execute("INSERT INTO silver.ingredient_candidates VALUES ('egg','9',3,0.77,0.77)")

    descriptions = [row[1] for row in candidates_for(con, "egg")]
    assert descriptions.count("Eggnog") == 1


def test_display_score_prefers_plain_modifiers_over_facet_count():
    """The ordinary entry often has MORE facets, so facet count must not be penalized."""
    plain = display_score("egg", 0.7, "Egg, whole, raw, fresh", False, True)
    variant = display_score("egg", 0.7, "Egg, white, dried", False, True)
    assert plain > variant


def test_display_score_prefers_the_leading_facet():
    """USDA leads with the food itself, so 'Egg, ...' should beat a higher-cosine 'Eggnog'."""
    assert display_score("egg", 0.70, "Egg, raw", False, True) > \
        display_score("egg", 0.75, "Eggnog", False, True)


def test_display_score_penalizes_deprioritized():
    assert display_score("onion", 0.7, "Onions, raw", False, True) > \
        display_score("onion", 0.7, "Onions, raw", True, True)


def test_search_finds_foods_outside_the_candidate_list(con):
    """Without this, recall@k would be 100% by construction."""
    results = search_foods(con, "cheddar")
    assert results[0][1] == "Cheese, cheddar"


def test_search_falls_back_to_fuzzy_on_a_near_miss(con):
    """The labeler has to guess USDA's vocabulary; a strict substring search punishes typos."""
    results = search_foods(con, "chedder")  # misspelled
    assert "Cheese, cheddar" in [r[1] for r in results]


def test_search_prefers_exact_substring_matches_first(con):
    results = search_foods(con, "egg whole")
    assert results[0][1] == "Egg, whole, raw, fresh"


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
