"""Candidate generation: narrow 8,187 USDA entities to a few dozen per ingredient string.

**No ANN index.** With 9,325 distinct strings against 8,187 candidates, the full similarity
matrix is a 9,325x8,187 matmul — seconds in numpy. An hnsw index would add a dependency and
approximation error to buy nothing, and exact top-k makes the recall@k ceiling a property of
the *representation* rather than of an index's recall. Chunked so peak memory stays small.

Three methods contribute, and every candidate records which one found it:

- `embed_search`  — cosine over `search_text` (facets reversed: "raw whole egg")
- `embed_desc`    — cosine over `description_raw` ("Egg, whole, raw")
- `token_head`    — candidates sharing the string's head noun, most generic first

The two embedding variants are a genuine A/B, but it cannot be settled before labels exist.
So labeling sees the UNION of all three — maximally inclusive, so the measured ceiling is not
capped by a premature choice — and `recall_at_k` is then a groupby over `method` afterwards.

Run directly:  uv run python -m pantryiq.er.candidates
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

DEFAULT_DB = Path("data/pantryiq.duckdb")
DEFAULT_CACHE = Path("data/embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"  # locked in the brief §4
# Generation k is deliberately generous and separate from what the labeling CLI displays.
# Stored candidates set the measurable recall ceiling, so capping them at the ~25 a human
# comfortably scans would cap the ceiling too: at k=25 the correct entry for "flour"
# (rank 34) fell outside the candidate set entirely, for the 6th most common ingredient.
TOP_K = 50
TOKEN_BLOCK_MAX = 50
CHUNK = 512


def load_model(name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: list[str], label: str, model=None, cache_dir: Path | str = DEFAULT_CACHE):
    """L2-normalized embeddings, cached on disk by content hash so re-runs are instant."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
    path = cache_dir / f"{label}-{MODEL_NAME}-{digest}.npy"
    if path.exists():
        return np.load(path)

    model = model or load_model()
    vectors = model.encode(
        texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)
    np.save(path, vectors)
    return vectors


def top_k_exact(queries, candidates, k: int = TOP_K, chunk: int = CHUNK):
    """Exact top-k cosine per query. Vectors are pre-normalized, so a dot product is cosine."""
    k = min(k, candidates.shape[0])
    indices = np.empty((queries.shape[0], k), dtype=np.int32)
    scores = np.empty((queries.shape[0], k), dtype=np.float32)
    for start in range(0, queries.shape[0], chunk):
        sims = queries[start:start + chunk] @ candidates.T
        partial = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(partial.shape[0])[:, None]
        ordered = np.argsort(-sims[rows, partial], axis=1)
        picked = partial[rows, ordered]
        indices[start:start + chunk] = picked
        scores[start:start + chunk] = sims[rows, picked]
    return indices, scores


def build_head_noun_index(canonical_names: list[str]) -> dict[str, list[int]]:
    """token -> candidate rows containing it, each block ordered most-generic-first.

    Generic-first matters: "lamb" matches 296 entries, and the labeling convention wants the
    least-qualified one. Fewest words is a decent proxy for least-qualified.
    """
    index: dict[str, list[int]] = {}
    for row, name in enumerate(canonical_names):
        for token in set(name.split()):
            index.setdefault(token, []).append(row)
    lengths = [len(name.split()) for name in canonical_names]
    for token, rows in index.items():
        rows.sort(key=lambda row: (lengths[row], row))
        index[token] = rows[:TOKEN_BLOCK_MAX]
    return index


def build_candidates(db_path: Path | str = DEFAULT_DB, k: int = TOP_K,
                     cache_dir: Path | str = DEFAULT_CACHE) -> pa.Table:
    """Generate the candidate union for every distinct ingredient string."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        foods = con.execute(
            "SELECT fdc_id, description_raw, search_text, canonical_name "
            "FROM silver.usda_foods ORDER BY fdc_id"
        ).fetchall()
        strings = [
            row[0] for row in con.execute(
                "SELECT normalized_text FROM silver.distinct_ingredient_strings "
                "ORDER BY normalized_text"
            ).fetchall()
        ]
    finally:
        con.close()

    fdc_ids = [f[0] for f in foods]
    descriptions = [f[1] for f in foods]
    search_texts = [f[2] for f in foods]
    canonical_names = [f[3] for f in foods]

    model = load_model()
    query_vectors = embed(strings, "strings", model, cache_dir)
    search_vectors = embed(search_texts, "usda_search", model, cache_dir)
    desc_vectors = embed(descriptions, "usda_desc", model, cache_dir)

    search_idx, search_scores = top_k_exact(query_vectors, search_vectors, k)
    desc_idx, desc_scores = top_k_exact(query_vectors, desc_vectors, k)
    head_index = build_head_noun_index(canonical_names)

    out: dict[str, list] = {key: [] for key in
                            ("normalized_text", "fdc_id", "from_embed_search", "from_embed_desc",
                             "from_token_head", "cosine_search", "cosine_desc", "rank")}
    for query_row, string in enumerate(strings):
        # MEMBERSHIP per generator, not "whichever found it first". Recording a single winning
        # method makes the A/B unmeasurable: whichever generator is consulted first claims
        # nearly every candidate, and the others appear to contribute ~nothing.
        by_search = {int(row) for row in search_idx[query_row]}
        by_desc = {int(row) for row in desc_idx[query_row]}
        head_noun = string.split()[-1] if string.split() else ""
        by_token = set(head_index.get(head_noun, []))

        union = by_search | by_desc | by_token
        # Tie-break on fdc_id, matching `resolve.top_picks`'s ORDER BY. Without it the order of
        # equal cosines falls out of `set` iteration — a hash-layout detail, not an ordering —
        # and 5.2% of strings tie at the top, so `rank = 0` (the §7 baseline arm) was decided
        # by hash order.
        rows = sorted(union, key=lambda row: (
            -float(query_vectors[query_row] @ search_vectors[row]), fdc_ids[row]))
        for rank, candidate_row in enumerate(rows):
            out["normalized_text"].append(string)
            out["fdc_id"].append(fdc_ids[candidate_row])
            out["from_embed_search"].append(candidate_row in by_search)
            out["from_embed_desc"].append(candidate_row in by_desc)
            out["from_token_head"].append(candidate_row in by_token)
            out["cosine_search"].append(
                float(query_vectors[query_row] @ search_vectors[candidate_row]))
            out["cosine_desc"].append(
                float(query_vectors[query_row] @ desc_vectors[candidate_row]))
            out["rank"].append(rank)
    return pa.table(out)


def alias_groups(descriptions: dict[str, str]) -> dict[str, set[str]]:
    """fdc_id -> every fdc_id sharing its `description_raw`.

    Guide rule 6: 94 descriptions exist twice, once as Foundation and once as SR Legacy, with
    different ids and slightly different values. Either is correct, so scoring on raw id
    equality understates every metric. Only ids with a twin appear here.
    """
    by_description: dict[str, set[str]] = {}
    for fdc_id, description in descriptions.items():
        by_description.setdefault(description, set()).add(fdc_id)
    return {fdc_id: group for group in by_description.values() if len(group) > 1
            for fdc_id in group}


def recall_at_k(candidates: pa.Table, labels: dict[str, str], k: int,
                method: str | None = None,
                aliases: dict[str, set[str]] | None = None) -> float:
    """Share of labeled strings whose gold entity appears within the top-k candidates.

    This is the ceiling on everything downstream: a scorer cannot pick what was never shown.
    `labels` maps normalized_text -> gold fdc_id; strings labeled `no-match` belong to the
    null class and are excluded from the denominator. Pass `method` — one of the
    `from_embed_search` / `from_embed_desc` / `from_token_head` membership columns — to
    attribute recall to a single generator.

    Pass `aliases` (from `alias_groups`) to apply the rule-6 duplicate credit. Without it a
    prediction is penalized for choosing the other member of a Foundation/SR Legacy pair, which
    is not an error.

    "Top-k" means the first k rows *after* any method filter, not rank < k. The two agree for
    the unfiltered table (ranks are dense), and the filtered reading is the meaningful one:
    it answers "if this generator ran alone and returned k, would the answer be in there?"
    """
    flags = candidates.column(method).to_pylist() if method else None
    by_string: dict[str, list[tuple[int, str]]] = {}
    for index, (text, fdc_id, rank) in enumerate(zip(
        candidates.column("normalized_text").to_pylist(),
        candidates.column("fdc_id").to_pylist(),
        candidates.column("rank").to_pylist(),
    )):
        if flags is not None and not flags[index]:
            continue
        by_string.setdefault(text, []).append((rank, fdc_id))

    hits = considered = 0
    for text, gold in labels.items():
        if gold == "no-match":
            continue
        considered += 1
        accept = aliases.get(gold, {gold}) if aliases else {gold}
        if any(fdc_id in accept for _, fdc_id in sorted(by_string.get(text, []))[:k]):
            hits += 1
    return hits / considered if considered else 0.0


def write_candidates(candidates: pa.Table, db_path: Path | str = DEFAULT_DB) -> Path:
    db_path = Path(db_path)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("candidate_rows", candidates)
        con.execute(
            "CREATE OR REPLACE TABLE silver.ingredient_candidates AS SELECT * FROM candidate_rows"
        )
    finally:
        con.close()
    return db_path


def main() -> None:
    candidates = build_candidates()
    write_candidates(candidates)
    strings = candidates.column("normalized_text").to_pylist()
    distinct = len(set(strings))
    print(f"silver.ingredient_candidates: {candidates.num_rows:,} rows "
          f"for {distinct:,} strings ({candidates.num_rows / distinct:.1f} candidates each)")
    for column in ("from_embed_search", "from_embed_desc", "from_token_head"):
        count = sum(candidates.column(column).to_pylist())
        print(f"  contributed by {column[5:]:13}: {count:7,} "
              f"({100 * count / candidates.num_rows:.1f}% of rows; generators overlap)")
    print("\nrecall@k needs labels — see scripts/er_metrics or docs/er_metrics.md.")


if __name__ == "__main__":
    main()
