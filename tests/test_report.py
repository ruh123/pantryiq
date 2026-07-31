"""The 2.5 holdout report. `picks` produces the §7 headline and had no test at all."""
import pyarrow as pa

from pantryiq.er.report import picks


def _table(rows):
    """rows = (split, normalized_text, fdc_id, rank), in table order."""
    return pa.table({
        "split": [row[0] for row in rows],
        "normalized_text": [row[1] for row in rows],
        "fdc_id": [row[2] for row in rows],
        "rank": [row[3] for row in rows],
    })


# Splits INTERLEAVED on purpose: `scores` is indexed split-locally while the table is indexed
# globally, so any fixture where the two happen to coincide cannot test the remap.
INTERLEAVED = _table([
    ("tune",    "t1", "A", 0),
    ("holdout", "h1", "P", 0),
    ("tune",    "t1", "B", 1),
    ("holdout", "h1", "Q", 1),
    ("tune",    "t2", "C", 0),
    ("holdout", "h2", "R", 0),
    ("holdout", "h2", "S", 1),
])


def test_scores_are_indexed_split_locally_not_by_table_position():
    """`picks` remaps each table-global row index to its position *within the split*, because
    `scores` comes from a model that only ever saw that split. Get the remap wrong and every
    string is silently scored against a different string's model output — the report still
    prints, and every number in §7 is quietly meaningless. There was no test for it.

    Holdout rows sit at global indices 1, 3, 5, 6 and split-local 0, 1, 2, 3."""
    scores = [0.10, 0.90, 0.20, 0.80]  # split-local: h1 -> (P=0.10, Q=0.90), h2 -> (R=0.20, S=0.80)

    result = {text: (best, baseline, score) for text, best, baseline, score in
              picks(INTERLEAVED, "holdout", scores)}

    assert result["h1"][0] == "Q"      # 0.90 beats 0.10
    assert result["h2"][0] == "S"      # 0.80 beats 0.20
    assert result["h1"][2] == 0.90     # the returned score is the winning one
    assert result["h2"][2] == 0.80


def test_only_rows_from_the_requested_split_are_returned():
    """Leaking tune rows into the holdout report would inflate it with strings the model was
    fitted on — the exact contamination the frozen split exists to prevent."""
    assert {row[0] for row in picks(INTERLEAVED, "holdout", [0.1, 0.9, 0.2, 0.8])} == {"h1", "h2"}
    assert {row[0] for row in picks(INTERLEAVED, "tune", [0.5, 0.6, 0.7])} == {"t1", "t2"}


def test_the_baseline_arm_is_the_lowest_rank_not_the_first_row():
    """The baseline is "top embedding cosine" = rank 0, and it is the comparison arm of §7's
    headline negative result. Reading table order instead would compare the model against
    whichever candidate happened to be stored first."""
    reordered = _table([
        ("holdout", "h1", "Q", 1),   # rank 1 stored FIRST
        ("holdout", "h1", "P", 0),
    ])

    _, _, baseline, _ = picks(reordered, "holdout", [0.9, 0.1])[0]

    assert baseline == "P"
