"""Candidate generation: exact top-k, head-noun blocking, and the recall@k ceiling."""
import numpy as np
import pyarrow as pa

from pantryiq.er.candidates import build_head_noun_index, recall_at_k, top_k_exact


def _normalized(rows):
    vectors = np.array(rows, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_top_k_exact_matches_brute_force():
    """Chunking must not change the answer — this is what makes the ceiling exact, not ANN."""
    rng = np.random.default_rng(0)
    queries = _normalized(rng.normal(size=(37, 16)))
    candidates = _normalized(rng.normal(size=(200, 16)))

    indices, scores = top_k_exact(queries, candidates, k=5, chunk=8)

    expected = np.argsort(-(queries @ candidates.T), axis=1)[:, :5]
    assert np.array_equal(indices, expected)
    assert np.allclose(scores, np.sort(queries @ candidates.T, axis=1)[:, ::-1][:, :5], atol=1e-6)


def test_top_k_handles_k_larger_than_corpus():
    vectors = _normalized(np.random.default_rng(1).normal(size=(3, 8)))
    indices, _ = top_k_exact(vectors, vectors, k=99)
    assert indices.shape == (3, 3)


def test_top_k_is_ordered_by_descending_similarity():
    queries = _normalized(np.array([[1.0, 0.0]]))
    candidates = _normalized(np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))

    indices, scores = top_k_exact(queries, candidates, k=3)

    assert list(indices[0]) == [2, 1, 0]
    assert scores[0][0] > scores[0][1] > scores[0][2]


def test_head_noun_index_is_generic_first():
    """'lamb' matches hundreds of entries; the least-qualified must surface first."""
    names = [
        "lamb new zealand imported frozen composite trimmed retail cuts raw",
        "lamb raw",
        "beef ground raw",
    ]
    index = build_head_noun_index(names)

    assert index["lamb"] == [1, 0]  # the 2-word entry before the 11-word one
    assert index["beef"] == [2]


def _candidate_table(rows):
    """rows = (text, fdc_id, generator, rank); the generator sets that membership flag.

    Membership is per generator, not "whichever found it first" — recording a single winning
    method makes per-generator recall unmeasurable.
    """
    return pa.table({
        "normalized_text": [r[0] for r in rows],
        "fdc_id": [r[1] for r in rows],
        "from_embed_search": [r[2] == "embed_search" for r in rows],
        "from_embed_desc": [r[2] == "embed_desc" for r in rows],
        "from_token_head": [r[2] == "token_head" for r in rows],
        "rank": [r[3] for r in rows],
    })


def test_recall_at_k_counts_only_within_k():
    # Ranks are dense per string, as build_candidates emits them.
    candidates = _candidate_table([
        ("egg", "100", "embed_search", 0),
        ("egg", "200", "embed_search", 1),
        ("sugar", "300", "embed_search", 0),
        ("sugar", "301", "embed_search", 1),
        ("sugar", "302", "embed_search", 2),
        ("sugar", "400", "token_head", 3),
    ])

    # egg's gold sits at rank 1, sugar's at rank 3.
    labels = {"egg": "200", "sugar": "400"}
    assert recall_at_k(candidates, labels, k=2) == 0.5
    assert recall_at_k(candidates, labels, k=4) == 1.0
    assert recall_at_k(candidates, labels, k=1) == 0.0


def test_recall_at_k_repacks_ranks_when_filtering_by_method():
    """Filtered recall asks 'if this generator ran alone and returned k, would it hit?'

    So a token_head candidate sitting at overall rank 40 is that generator's rank-0 result.
    """
    candidates = _candidate_table(
        [("egg", str(i), "embed_search", i) for i in range(40)]
        + [("egg", "gold", "token_head", 40)]
    )
    labels = {"egg": "gold"}

    assert recall_at_k(candidates, labels, k=5) == 0.0  # not in the overall top 5
    assert recall_at_k(candidates, labels, k=5, method="from_token_head") == 1.0


def test_recall_at_k_excludes_the_null_class():
    """no-match strings have no entity to retrieve; counting them would deflate the ceiling."""
    candidates = _candidate_table([("egg", "100", "embed_search", 0)])
    labels = {"egg": "100", "love": "no-match"}

    assert recall_at_k(candidates, labels, k=5) == 1.0


def test_recall_at_k_attributes_per_method():
    candidates = _candidate_table([
        ("egg", "100", "embed_search", 0),
        ("egg", "200", "token_head", 1),
    ])
    labels = {"egg": "200"}

    assert recall_at_k(candidates, labels, k=5) == 1.0
    assert recall_at_k(candidates, labels, k=5, method="from_token_head") == 1.0
    assert recall_at_k(candidates, labels, k=5, method="from_embed_search") == 0.0


def test_recall_at_k_missing_string_is_a_miss():
    candidates = _candidate_table([("egg", "100", "embed_search", 0)])
    assert recall_at_k(candidates, {"nonexistent": "999"}, k=5) == 0.0
