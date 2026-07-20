# PantryIQ — Agent Operating Brief (Master Prompt)

> **How to use this document.** This is the persistent brief for building PantryIQ. If you are an AI coding agent working on this project, read this file **and** `PantryIQ_Design_Doc.md` at the start of every session. The decisions in §4 are **locked** — do not re-open or re-litigate them; if you think one is wrong, say so explicitly and wait, don't silently deviate. The full technical spec lives in the design doc; this brief adds the locked choices, the working cadence, and the phase gates.

---

## 1. Mission & thesis

Build a **verified recipe data platform first, then a thin AI layer that can only speak from that verified data.**

The failure mode of LLM recipe apps is confident, ungrounded claims about nutrition, cost, and ingredients. PantryIQ's entire reason to exist is to make that impossible by construction: real data is ingested, cleaned, and resolved to canonical entities with *measured* accuracy, and the AI is constrained to answer only from those grounded rows — with a **deterministic** check on every numeric claim it makes.

**The two things that make this more than "a recipe app":**
1. **Entity resolution** (free-text ingredients → canonical USDA entities) with reported precision/recall. This is the headline.
2. **A deterministic numeric guardrail** that verifies the model's output against source data in code — not another LLM trusting itself.

Plus a demo-able quality signal: **`data_trust_score`** (min ingredient-resolution confidence per recipe) surfaced directly in the product.

## 2. What to protect above all else

- **Finish entity resolution and measure it.** If time runs short, the AI layer may be simpler — the ER component and its metrics may not be cut.
- **Treat the guardrail as its own tested component.** Log and report its catch rate; never assume it's perfect.
- **Lead with the ER problem and metrics** in the README and any demo — not the chat UI. The chat UI is the least differentiating part.

## 3. Architecture at a glance

```
Sources                     Bronze (Iceberg)        Resolution              Silver + Gold (DuckDB, dbt)     Serving
-------                     ----------------        ----------              --------------------------     -------
RecipeNLG         --batch->  raw_recipes    \                                                          
USDA FDC API      --api--->  raw_usda_foods   +--> Entity Resolution --> dbt models --> Gold tables --> AI Agent --> Streamlit UI
Open Food Facts   --api--->  raw_off_products/     (parse/normalize/       (nutrition,     (verified      (structured  (chat +
                                                    block/score/route)      cost, tags,     rows)          retrieval +  trust
                                                                            trust_score)                   guardrail)   score)

   [ Bronze = Apache Iceberg via a REST catalog (PyIceberg writes, DuckDB reads) ]
   [ Silver + Gold = native DuckDB tables managed by dbt — NOT Iceberg ]
   [ Airflow orchestrates every arrow; dbt tests + Great Expectations gate Bronze->Silver->Gold ]
```

Layer responsibilities (see design doc §4 for full column lists):

| Layer | Storage | Managed by | Purpose |
|---|---|---|---|
| **Bronze** | Apache Iceberg | PyIceberg (write), DuckDB (read) | Immutable raw landing zone; raw payloads + source + `ingested_at`; always re-derivable |
| **Silver** | Native DuckDB | dbt | Typed/cleaned per-source; `ingredient_entity_map` with method + confidence |
| **Gold** | Native DuckDB | dbt | Canonical, analytics-ready; `recipe_nutrition` (+`data_trust_score`), tags, `agent_query_log` |

## 4. Locked technical decisions

These were chosen deliberately. Do not change them without an explicit new decision.

| Area | Decision | Notes / rationale |
|---|---|---|
| Warehouse (Silver+Gold) | **DuckDB** (native tables) | Zero infra, fits single-machine batch; keep SQL **Snowflake-portable** so a swap stays cheap. DuckDB 1.5+. |
| Bronze storage | **Apache Iceberg** (raw layer only) | Open-table-format story where it earns its keep: schema evolution, time travel, ACID, re-derivability. **One Iceberg REST catalog**; PyIceberg writes, DuckDB's `iceberg` extension reads. Local filesystem warehouse to start; S3/MinIO only at deploy. |
| Why not Iceberg for Silver/Gold | `dbt-duckdb` → Iceberg materialization is still rough | Native DuckDB tables keep dbt smooth **and** serving fast (Gold is on the query hot path; Bronze never is). |
| Transformation | **dbt-core + dbt-duckdb** | Testable models; Bronze Iceberg read as sources. |
| Orchestration | **Airflow** (local / docker-compose) | DAG: `ingest_sources → run_entity_resolution → dbt_run → dbt_test → quality_gate`; idempotent, retryable. |
| Data quality gate | **dbt tests + Great Expectations** | A failed gate **blocks** promotion to Gold — bad data cannot silently reach the AI. |
| Bulk ingredient matching | **Local `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) + Jaro-Winkler** | Free, on-machine; no API cost for the majority of matches. |
| LLM | **Anthropic Claude** | Reserved for **mid-confidence ER adjudication + final generation only**. Structured output / tool use. |
| Recipe dataset | **RecipeNLG** | Full original ingredient lines *with* quantities + messy raw text (+ `NER` name column) — chosen to keep the ER metric honest and enable per-recipe nutrition/cost. **Switched from Food.com/Kaggle on 2026-07-20** after the readable export was confirmed to store ingredients as pre-cleaned names without quantities. ~2.2M recipes → sample ~15K for the subset phase. |
| Nutrition source | **USDA FoodData Central API** | Canonical set = **Foundation + SR Legacy (~8k)**; **exclude Branded** (millions). Free key, rate-limited — paginate + backoff + cache; pull once. |
| Products | **Open Food Facts** (sampled) | Third source; sampled ingest. |
| Cost data | **Static curated reference table** | No live pricing (explicit non-goal); label it as a stand-in. |
| Serving UI | **Streamlit** | Chat UI; surfaces verified numbers + `data_trust_score` inline. |
| Deployment | **Docker → one managed cloud** (Cloud Run / Fly.io) | Deploy the **Streamlit app + a read-only Gold DuckDB file**. Batch pipeline, Airflow, and the Iceberg catalog run **locally** — do not host the whole lakehouse in the cloud. |
| IaC / CI-CD | **Minimal** — GitHub Actions + light Terraform or native cloud config | Container-first; not a full multi-service cloud deployment. |
| Scale strategy | **Subset-first (~15K recipes) → full (100–500K)** once ER metrics are green | De-risks the ML core; keeps early LLM spend low. |
| Working style | **Tight step-by-step** (see §5) | Propose → wait for approval → next; verify each step. |

## 5. How to work (operating cadence)

Follow **the Karpathy Principles** (the user's global `CLAUDE.md`). In short:

1. **Think before coding** — state assumptions; if multiple interpretations exist, surface them; if a simpler path exists, say so; if unclear, stop and ask.
2. **Simplicity first** — minimum code that solves the problem; nothing speculative; no abstractions for single-use code.
3. **Surgical changes** — touch only what the task requires; match existing style; don't refactor what isn't broken.
4. **Goal-driven execution** — turn each task into a verifiable goal ("write a test that reproduces the bug, then make it pass") and loop until verified.

**Build cadence for this project (tight step-by-step):**
- Work in **small, reviewable increments**. Propose the next concrete step, then **stop and wait for approval** before implementing it.
- After each step, **show the verification** (test output, row counts, a metric) — don't claim done without evidence.
- Build and validate on the **~15K subset** through Phases 1–2; only scale to full data once ER metrics are acceptable.
- Keep each phase to its **definition of done** below — resist building ahead.

## 6. Build phases & definition of done

Mapped to the design doc's milestones, adjusted for subset-first. Each phase ends at a gate.

**Phase 0 — Foundations**
- Repo scaffold; Python 3.11+ env (uv or poetry); pre-commit/lint; move design doc + this brief into `docs/`; README skeleton; `docker-compose` skeleton; `.env` handling for `USDA_API_KEY` and `ANTHROPIC_API_KEY` (never commit secrets).
- **DoD:** `make setup` (or equivalent) works from clean; CI lints; empty pipeline scaffold runs.

**Phase 1 — Ingestion (~Week 1, on the ~15K subset)**
- Stand up the Iceberg REST catalog + local warehouse. Write `bronze.raw_recipes` (RecipeNLG ~15K subset), `bronze.raw_usda_foods` (FDC pull), `bronze.raw_off_products` (OFF sample) via PyIceberg. Preserve raw payloads unmodified + `source` + `ingested_at` lineage. Log row counts. Confirm DuckDB can read all three back.
- **First task (dataset resolved):** recipe source = RecipeNLG (§9 risk 1). Confirm its `ingredients` (raw lines) + `NER` (names) columns on a ~15K sample; note the absence of a servings field for Phase-3 nutrition.
- **DoD:** three Bronze Iceberg tables populated with lineage; row counts logged; DuckDB round-trip read verified.

**Phase 2 — Entity resolution (~Week 2, the core, on subset)**
- Pipeline: **parse** (quantity/unit/ingredient) → **normalize** (lowercase, strip filler, singularize) → **block** (category/first-token) → **score** (Jaro-Winkler + local embeddings, top-k) → **route by confidence**: high → auto (`rule`/`fuzzy`); mid → **Claude adjudication** (present raw text + top-3, structured pick or "no match", method=`llm`); low → flag unresolved. Write `silver.ingredient_entity_map` with `match_method`, `confidence_score`, `reviewed_flag`.
- **Hand-label 200–300 random raw strings**; compute **precision / recall / F1 overall and per method**, plus **% resolved without an LLM call**.
- **DoD:** end-to-end resolution on the subset; metrics measured and written up; entity map populated.
- **Scale gate:** once metrics are acceptable, scale ingestion + resolution to the **full 100–500K**.

**Phase 3 — Transformation, orchestration, quality (~Week 3, full scale)**
- dbt Silver→Gold: `canonical_ingredients`, `recipe_ingredients_resolved`, `recipe_nutrition` (+ `data_trust_score` = min ingredient confidence per recipe), `recipe_tags`. Join resolved ingredients to USDA nutrients, sum to recipe level, join cost table for cost/serving, derive dietary tags from composition rules.
- Airflow DAG wiring every step; idempotent + retryable tasks.
- dbt tests + Great Expectations: nutrient sanity bounds, referential integrity (every resolved ingredient → a real canonical entity), confidence thresholds. **Quality gate blocks promotion.**
- **DoD:** Gold populated at full scale; DAG runs end-to-end; a deliberately bad row is **blocked** by the gate (prove it).

**Phase 4 — AI agent (~Week 4)**
- **Retrieval:** SQL over Gold ranking candidates by ingredient overlap + constraint fit (structured retrieval, not free-text vector search). **Context assembly:** package top-N real nutrition/cost/ingredient data into a structured object. **Generation:** Claude given *only* that context (structured output/tool use) — cannot introduce outside facts. **Guardrail:** deterministic code extracts every numeric claim and checks it against the source context; mismatch → correct or regenerate; log catch rate. **Multi-step planning:** constraint selection (e.g. "week under $50 and 2,000 cal/day") solved **in code**; the LLM only narrates. **Quality explainer:** on a failed dbt test / low-confidence resolution, agent drafts a plain-English summary into `gold.agent_query_log` / a runbook table.
- **DoD:** agent answers pantry+constraint queries **in <5s** using only verified facts; guardrail catches injected numeric errors (test it); week-planner respects budget + calories; `agent_query_log` populated.

**Phase 5 — Serving & deployment (~Week 5)**
- Streamlit chat UI surfacing verified numbers + `data_trust_score`. Dockerize. Deploy the app + read-only Gold to Cloud Run / Fly.io. GitHub Actions CI/CD. README (**leads with the ER problem + headline metrics**), architecture diagram, demo video.
- **DoD:** live URL works; CI green; README leads with entity resolution and reports the metrics from §7.

## 7. Success metrics to protect (report these)

- **Entity-resolution precision / recall / F1** — the headline metric.
- **Raw ingredient strings → canonical entities resolved** — scale metric.
- **% of resolutions needing an LLM call** — cost-efficiency metric.
- **% of recipes with complete verified nutrition coverage.**
- **Guardrail catch rate during testing** — trust/safety metric.

## 8. Cost & LLM discipline

- The **bulk** of matching is local embeddings + fuzzy — **no API calls.** Claude is only for the mid-confidence long tail + final generation.
- **Never** let the LLM do arithmetic or constraint-satisfaction — that stays in code; the model only narrates.
- Batch/cache Claude adjudication calls where possible; log token spend; keep a soft budget and report the % handled without an LLM.

## 9. Known risks & open questions — resolve early

1. **Recipe ingredient quantities — RESOLVED (dataset switched 2026-07-20).** The Food.com Kaggle path failed twice: the file provided was the tokenized `PP_recipes` export (integer token IDs only, no readable text), and Food.com's readable `RAW_recipes.csv` stores `ingredients` as pre-cleaned names *without* quantities — which would make ER artificially easy and block per-serving nutrition. **Decision: recipe source switched to RecipeNLG**, whose `ingredients` are full original lines with quantities ("2 cups flour, sifted") plus an `NER` name column. **New Phase-3 caveat:** RecipeNLG has **no servings and no nutrition columns**, so per-recipe totals are computable from quantities, but "per-serving" needs a servings figure — parse/estimate it, or report per-recipe + per-100g and state the limitation. It's ~2.2M recipes (~2GB); use the subset-first ~15K sample. **Confirmed 2026-07-20** on a 500-row sample: 93.8% of ingredient lines carry a leading quantity, NER gives clean names (but is coarse — e.g. "evaporated milk" → "milk", so build the Phase-2 gold labels from raw lines, not NER), columns = `title/ingredients/directions/link/source/NER` (no servings/nutrition).
2. **USDA FDC rate limits.** Default ~1,000 req/hr on an `api.data.gov` key — paginate, back off, cache, pull once. Exclude Branded foods from the canonical set.
3. **Guardrail is not assumed perfect.** It is its own tested component; log its catch rate rather than trusting it.
4. **Iceberg catalog overhead.** One REST catalog + local FS warehouse to start; add MinIO/S3 only at deploy. Don't let catalog infra eat entity-resolution time. (Note: Iceberg-at-Bronze adds **no** serving latency — Bronze is off the query hot path — only a small, amortized batch cost.)
5. **dbt-into-Iceberg is rough** — that is *why* Silver/Gold are native DuckDB. Do not try to materialize dbt models into Iceberg.
6. **Timeline / scope creep.** Protect the ER core above everything; simplify the AI layer before cutting ER.

## 10. Non-goals (out of scope by design)

- No live/real-time grocery pricing — a static curated cost table stands in.
- No user accounts, auth, or persistent history — single-session use only.
- No real or sensitive personal data — public data only; privacy/security requirements are intentionally minimal (state this as a scope limitation, not an oversight).

## 11. Kickoff instruction (first action when this brief is handed to an agent)

**Do not start building.** First:
1. Confirm you've read this brief and `PantryIQ_Design_Doc.md`.
2. Restate the §4 locked decisions in one line to confirm alignment (flag anything you'd challenge).
3. Inspect the environment: Python/uv, Docker, whether the Food.com dataset, USDA key, and `ANTHROPIC_API_KEY` are present.
4. Propose the **single first concrete step** — either Phase-0 scaffold or the Phase-1 Food.com quantity check — as a small, reviewable increment.
5. **Stop and wait for approval.** Then proceed one small step at a time, showing verification after each.
