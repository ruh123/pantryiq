# Code-review backlog (2026-07-24)

> **Status update 2026-07-31.** All three P0s and four P1s below are **FIXED** — verified in a
> later review, not assumed: the empty-overwrite guard (`recipes.py:91`, `usda.py:143`), `_get_page`
> retry coverage (5 new tests), idempotency tests that re-run with *different* input, the
> `isinstance(body, list)` check, the `Retry-After` 60s cap, and the `iceberg_scan(?)` bound
> parameter. README Status and regen commands are updated too.
>
> **Still genuinely open:** no `fdc_id` sort in `usda.py`; no ragged-row guard in
> `recipes.build_bronze_table`; no `requests.Session`; `scan_with_duckdb` still untested against a
> genuinely multi-parquet table; stale `pyproject.toml` comment; no `ruff format --check` in CI;
> the in-memory materialization that needs a streaming rewrite at the scale gate.
>
> The Phase-2 ER-plan review this file asked for **has since been run** — four reviews, recorded
> in `er_metrics.md` §13.

From a 3-subagent review of the Phase-1 Bronze implementation. **Two of three reviewers
finished** (code-correctness, test-quality); the **Phase-2 ER-plan reviewer failed on a
session limit — re-run it next session.** Design-level P0s (unit→gram conversion, servings,
blocking recall, calibration, gold stratification) are already tracked in `phase2_plan.md` /
the Master Prompt and are not repeated here.

Baseline at review time: `pytest` 10/10 green, `ruff` clean, no secrets tracked, pushed
through commit `aaa438a`. Bronze: `raw_recipes` (15K) + `raw_usda_foods` (8,187).

## P0 — fix before trusting the pipeline
- **Empty/short fetch silently WIPES a Bronze table.** `recipes.py:88` / `usda.py:140`
  `table.overwrite(data)` uses default `ALWAYS_TRUE`; a 0-row `data` atomically replaces the
  table with 0 rows, no error (repro'd: re-run with `foods=[]` → 0 rows). Trigger: transient
  empty API page (HTTP 200), wrong `dataType`, `limit=0`, wrong-but-existing CSV path.
  **Fix:** refuse to overwrite when `data.num_rows == 0` (or below a tolerance vs current count); raise loudly. Optional `--allow-empty`.
- **`_get_page` retry/backoff is 100% untested.** The riskiest code (network/retries) has no
  coverage — `test_fetch_foods_paginates_and_stops` monkeypatches `_get_page` out.
  **Fix:** unit-test with a fake `requests.get` (+ `sleep=lambda _:None`): 429→retry-then-200, `Retry-After: "3"` honored, 404→raises now, 5×503→`RuntimeError`.
- **Idempotency tests can't tell overwrite from a stale no-op.** Both idempotency tests re-run
  the identical fixture and assert only `num_rows`. A broken overwrite that no-ops still passes.
  **Fix:** second run with DIFFERENT input (e.g. `n=5`); assert new count/content wins (5 vs append=12 vs stale=7).

## P1 — important
- **`resp.json()` not validated as a list.** `usda.py:65→88` — a non-list 200 body (`{"error":…}`) makes `yield from` iterate dict keys → `AttributeError` in `build_bronze_table`. **Fix:** `isinstance(body, list)` check in `_get_page`; raise with payload logged.
- **`Retry-After` sleep is uncapped.** `usda.py:67-68` — `Retry-After: 999999` stalls ~11 days. **Fix:** `min(int(retry_after), 60)`.
- **`iceberg_scan('{metadata}')` is f-string SQL.** `catalog.py:41` — a warehouse path with a quote (`/Users/O'Brien/…`) breaks parsing. **Fix:** bound parameter `iceberg_scan(?)`.
- **`overwrite` is not a real idempotency key.** `recipes.py:88` / `usda.py:140` — correct only because each table is single-source today; a second source would be deleted. **Fix:** if multi-source, `overwrite(data, overwrite_filter=EqualTo("source", SOURCE))`; else document the single-source assumption.
- **`build_bronze_table` materializes the whole corpus in memory.** `recipes.py:50-72` — fine at 15K, several GB at 2.2M. Gated behind the scale-gate. **Fix:** batched/streaming writer for recipes before full corpus (USDA is fine at ~8K).
- **USDA Bronze not reproducible across days.** Live pull, no `fdc_id` sort, no recorded count; Foundation grows over time. **Fix:** record pull-date + expected count in README; sort by `fdc_id`.
- **README stale + missing regen commands.** Says "Phase 1 in progress"; both tables are landed. Regen entrypoints only in docstrings. **Fix:** update Status; add `uv run python -m pantryiq.ingestion.{recipes,usda}` + expected counts (15,000 / 8,187); optional `make ingest`.
- **`recipes.build_bronze_table` lacks the ragged-row guard the inspector has** (`len(row) < 7`). Untested: `row.get("") or str(i)` fallback, short rows, `limit` truncation; usda `""` fallbacks for missing `fdcId`/`dataType`. **Fix:** guard + tests.
- **`scan_with_duckdb` only tested on single-file tables.** Real `raw_recipes` is multi-parquet after overwrites; the manifest-follow path is unexercised. **Fix:** test with two appends before the scan.

## P2 — minor / polish
- `INSTALL iceberg; LOAD iceberg;` + new connection on every `scan_with_duckdb` call (`catalog.py:38-39`) — install once at init / `LOAD` only.
- `.replace("file://","")` fragile (Windows `/C:/…`, S3 passthrough); use `urllib.parse.urlparse().path`. Will need rework at S3/MinIO deploy.
- `ingested_at = datetime.now()` → byte-non-deterministic re-runs (acceptable lineage).
- `row.get("")` assumes empty leading header; a UTF-8 BOM breaks it (use `utf-8-sig` + assert headers).
- `raw_payload` stored as JSON string (deliberate raw-preservation; Silver uses `json_extract`).
- No `requests.Session` → ~43 handshakes per USDA pull.
- Exactly-full last page → one extra empty request; retry sleeps before the final raise. Both cosmetic.
- `inspect_recipenlg.py` QTY regex: `"3.5% milk"` still leaks (`\d+\.\d+` fires before the `%` guard); `".5 cup"` missed; possessive `\d++` needs 3.11 (comment it).
- CI: add `ruff format --check`; consider a `pytest --cov` report (no gate). No Python matrix needed (3.11 pinned).
- Warehouse accumulates orphan parquet across re-runs (no `expire_snapshots`); document "delete `data/lakehouse` for a clean regen" / `make clean`.
- Stale `pyproject.toml` comment (lines 14-15) lists deps already added.

## Positive confirmations (probed, not bugs)
- Mid-pagination failure is safe: generator fully materialized before any write → a late `RuntimeError` leaves the prior table intact (no half-write; no resume either).
- Retry exhaustion + non-retryable statuses handled correctly.
- Iceberg `overwrite` is snapshot-atomic (worst case = orphan files, not corruption).

## Decision questions to resolve next session
1. **Empty-overwrite guard:** 0-row/tolerance guard (recommended) vs `--allow-empty`.
2. **Multi-source intent:** will `raw_recipes`/`raw_usda_foods` ever hold >1 source? (drives overwrite-filter vs documenting single-source).
3. **Ragged/malformed recipe rows:** skip (like the inspector) or land raw with `None` fields? (Bronze is source-preserving — a policy call).
4. **15K subset:** deterministic first-N (current) vs seeded random sample (representative + deterministic) — affects ER-metric bias.
5. **USDA reproducibility:** record pull-date + count (recommended) vs pin/sort snapshot.
6. **`scan_with_duckdb` hardening** (bound param + `urlparse`): now vs deploy-time.
7. **QTY regex:** fix decimal-% / leading-decimal, or freeze the Phase-1 diagnostic as-is.
8. **CI scope:** add `ruff format --check` + coverage, or keep minimal.
9. **Scale rewrite (streaming):** schedule at the scale-gate vs track now.
