# PantryIQ — Technical Design Document

## 1. Problem statement

Recipe recommendation systems built on LLMs alone produce plausible-sounding but unverified nutrition, cost, and ingredient claims, because the model has no grounded source of truth to check itself against. PantryIQ solves this by building a real, verified data platform first, and only then exposing an AI layer that is constrained to answer from that verified data.

## 2. Goals and non-goals

**Goals**
- Ingest recipe, nutrition, and product data from multiple public sources into a governed lakehouse.
- Resolve inconsistent, free-text ingredient descriptions into canonical entities with measurable precision/recall.
- Compute accurate, derived nutrition and cost data per recipe.
- Serve a natural-language meal-planning agent whose factual claims are verified against the underlying data before being shown to a user.
- Produce a working, deployed demo with real, reportable metrics.

**Non-goals (explicitly out of scope, worth stating in a real design doc)**
- Real-time grocery price integration (a static/curated reference table is used instead — live pricing APIs are largely paid/gated).
- Personalized user accounts, auth, or persistent user history (single-session use only).
- Handling any real personal or sensitive user data — this is public data only, so privacy/security requirements are intentionally minimal compared to a production enterprise system.

## 3. System architecture

```
Sources                Bronze                 Resolution            Silver/Gold                 Serving
--------                ------                 ----------            -----------                 -------
Recipe1M+/   --batch-->  raw_recipes   \                                                    
Food.com                                 \                                                  
                                           +--> Entity Resolution --> canonical mapping --> dbt models --> Gold tables --> AI Agent --> Chat UI
USDA FDC API --api-->    raw_usda_foods  /        (rules, fuzzy,        (silver.ingredient_    (nutrition,
                                                    LLM fallback)         entity_map)            cost, tags)
Open Food     --api-->   raw_off_products /
Facts

                                                                    [ Airflow orchestrates every arrow above ]
                                                                    [ dbt tests + Great Expectations gate Bronze->Silver->Gold ]
```

## 4. Data model

**Bronze (raw, source-preserving)**

| Table | Key columns |
|---|---|
| `bronze.raw_recipes` | recipe_id, source, ingested_at, raw_payload (JSON), source_url |
| `bronze.raw_usda_foods` | fdc_id, ingested_at, raw_payload (JSON) |
| `bronze.raw_off_products` | product_id, source, ingested_at, raw_payload (JSON) |

**Silver (typed, cleaned, per-source)**

| Table | Key columns |
|---|---|
| `silver.recipes` | recipe_id, title, instructions, servings, source |
| `silver.recipe_ingredient_lines` | recipe_id, line_raw, parsed_quantity, parsed_unit, parsed_ingredient_text |
| `silver.usda_foods` | fdc_id, canonical_name, category, nutrients (normalized nutrient table) |
| `silver.ingredient_entity_map` | raw_ingredient_text, canonical_id, match_method (`rule`/`fuzzy`/`llm`), confidence_score, reviewed_flag |

**Gold (canonical, analytics-ready)**

| Table | Key columns |
|---|---|
| `gold.canonical_ingredients` | canonical_id, fdc_id, display_name, category |
| `gold.recipe_ingredients_resolved` | recipe_id, canonical_id, quantity, unit |
| `gold.recipe_nutrition` | recipe_id, total_calories, protein_g, carbs_g, fat_g, cost_per_serving, data_trust_score |
| `gold.recipe_tags` | recipe_id, tag |
| `gold.agent_query_log` | query_id, user_input, retrieved_recipe_ids, generated_response, guardrail_pass, numeric_claims_checked, timestamp |

`data_trust_score` is a computed field (e.g., the minimum ingredient-resolution confidence across a recipe's ingredients) — it's a novel, demo-able way to expose data quality directly in the product, not just in a monitoring dashboard.

## 5. Component design

### 5.1 Ingestion
Batch job loads the recipe dataset once (large historical load). Scheduled jobs pull USDA FoodData Central (paginated, rate-limited, retried with backoff) and a sample of Open Food Facts. All raw payloads are preserved unmodified in Bronze with source and timestamp — cleaning never happens on the way in, only in Silver, so you can always re-derive from source of truth.

### 5.2 Entity resolution (the core component)
1. **Parse**: regex/lightweight-grammar parser splits each raw ingredient line into quantity, unit, and ingredient text.
2. **Normalize**: lowercase, strip filler words ("fresh," "chopped," "large"), singularize.
3. **Block**: group candidate USDA foods by category/first-token to avoid comparing every raw string against all ~8,000 canonical foods.
4. **Score**: fuzzy string similarity (e.g., Jaro-Winkler) or embedding cosine similarity within each block, producing top-k candidates with scores.
5. **Route by confidence**:
   - High score → auto-accept (method = `rule`/`fuzzy`).
   - Mid score → LLM adjudication: present the raw text and top-3 candidates, ask the model to pick the correct match or return "no match" (method = `llm`).
   - Low score → flagged unresolved for manual review.
6. **Evaluate**: hand-label a random sample (200-300 raw strings) with the correct canonical mapping; compute precision/recall/F1 overall and per method. This is the headline metric for the whole project.

### 5.3 Transformation (dbt)
Silver → Gold models join resolved ingredients to USDA nutrient data, sum to recipe-level nutrition, join the cost reference table for cost-per-serving, and derive dietary tags from ingredient composition rules.

### 5.4 Orchestration & data quality
Airflow DAG: `ingest_sources` → `run_entity_resolution` → `dbt_run` → `dbt_test` → `quality_gate`. dbt tests and Great Expectations checks enforce nutrient sanity bounds, referential integrity (every resolved ingredient maps to a real canonical entity), and confidence thresholds. A failed quality gate blocks promotion to Gold, not just an alert — bad data should not be able to silently reach the AI layer.

### 5.5 AI agent
1. **Retrieval**: given pantry items + constraints, a SQL query against Gold ranks candidate recipes by ingredient overlap and constraint fit. This is structured retrieval against real rows, not open-ended vector search over free text.
2. **Context assembly**: package the top-N candidates' real nutrition/cost/ingredient data into a structured object passed to the model.
3. **Generation**: the LLM is given only that structured context (function-calling / structured output) and asked to recommend and explain — it cannot introduce facts outside the provided context by design.
4. **Guardrail**: after generation, every numeric claim in the model's output is programmatically extracted and checked against the source context. Mismatches are corrected or trigger regeneration. This check is deterministic code, not another LLM call trusting itself.
5. **Multi-step planning**: for requests like "plan my week under $50 and 2,000 cal/day," a simple constraint-selection pass in code picks a feasible set of recipes, and the LLM only narrates the result — arithmetic and constraint-satisfaction stay in code, not in the model.
6. **Quality explainer**: triggered by a failed dbt test or low-confidence resolution; the agent reads failure metadata and recent logs and drafts a plain-English summary, stored in `gold.agent_query_log`/a runbook table.

## 6. Technology choices and rationale

| Decision | Choice | Why |
|---|---|---|
| Warehouse | Snowflake (trial) or DuckDB | Matches your existing resume claims (Snowflake); DuckDB if you want zero infrastructure cost |
| Transformation | dbt-core | Free, matches existing resume experience, testable models |
| Orchestration | Airflow | Already on your resume — reuse rather than learn a new tool for this project |
| Fuzzy matching | Jaro-Winkler / embedding similarity | Cheap, deterministic, avoids unnecessary LLM calls on the majority of cases |
| LLM use | Reserved for ambiguous entity resolution + final generation only | Cost-aware design — the project should demonstrate judgment about when an LLM is (and isn't) the right tool |

## 7. Non-functional requirements

- **Scale**: 100K-500K recipes, ~8,000 canonical USDA foods; batch entity resolution should complete in hours, not days, on a single machine.
- **Cost**: minimize LLM spend by routing only the ambiguous long tail to the model; track and report the % of resolutions handled without an LLM call as a cost-efficiency metric.
- **Latency**: agent end-to-end response target under ~5 seconds (structured retrieval + 1-2 LLM calls).
- **Reliability**: idempotent, retryable Airflow tasks; quality gates block bad data from propagating.
- **Security/privacy**: minimal by design — only public data is used, no real user data is stored. Worth noting explicitly as a scope limitation relative to a real enterprise system handling regulated data.

## 8. Milestones

| Week | Deliverable | Definition of done |
|---|---|---|
| 1 | Ingestion | Bronze populated from all 3 sources with source lineage; row counts logged |
| 2 | Entity resolution | Resolution pipeline complete; precision/recall measured on hand-labeled sample |
| 3 | Transformation, orchestration, quality | Gold tables populated; Airflow DAG running end-to-end; quality gate enforced |
| 4 | AI agent | Retrieval + generation + guardrail working; multi-step planning and quality-explainer agent functional |
| 5 | Serving & deployment | Chat UI deployed live; Docker/Terraform/CI-CD in place; README, architecture diagram, demo video published |

## 9. Success metrics

- Entity resolution precision/recall (headline metric)
- Raw ingredient strings resolved → canonical entity count (scale metric)
- % of resolutions requiring an LLM call (cost-efficiency metric)
- % of recipes with complete verified nutrition coverage
- Guardrail catch rate during testing (trust/safety metric)

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Entity resolution accuracy too low on messy long-tail text | LLM adjudication fallback; tune confidence thresholds against the labeled sample |
| Project stays partially finished | Protect entity resolution completion above all else — it's the differentiator; the AI layer can be simpler if time runs short |
| Guardrail doesn't catch a hallucination | Treat the guardrail as its own tested component; log and report its catch rate rather than assuming it's perfect |
| Reviewers see it as "just a recipe app" | Lead with the entity resolution problem and metrics in the README and in conversation, not the chat UI |

## 11. Next step

Start Week 1 (ingestion) once you're ready. This document, plus the earlier project plan, can live in the repo as `docs/DESIGN.md` — having a real design doc in the repo is itself a small additional signal of seniority.
