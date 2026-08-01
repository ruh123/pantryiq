-- The curated cost reference, as a Gold model.
--
-- **THESE PRICES ARE A STAND-IN, NOT SOURCED DATA.** Live grocery pricing is an explicit
-- non-goal (Master Prompt §4, §10); this is a hand-curated table of typical US retail prices so
-- cost-per-recipe is demonstrable at all. Unlike `gram_reference.csv` — whose values are USDA's
-- own published weights — nobody can verify these against a source, and any figure derived from
-- them must say so.
--
-- Coverage is ~75% of converted ingredient MASS across 98 entities, which is the right
-- denominator: cost is per kilogram. `cost_coverage` downstream reports it per recipe.
--
-- Prices are per entity, not per category, and an entity is priced for WHAT IT IS. Where a
-- resolution error routes volume to an odd entity (a fast-food burger, a babyfood jar), that
-- entity is still priced as itself — the error stays visible as a resolution error instead of
-- being laundered into the cost table.
--
-- Reads a dbt SEED rather than `read_csv('data/...')`. The seed path resolves relative to the
-- PROJECT, not the process working directory (dbt-duckdb hands relative paths to the OS, so the
-- old form silently built against an empty warehouse when run from elsewhere), and `read_csv` is
-- DuckDB-only — it was the least Snowflake-portable line in the layer, against §4's brief.

select
    fdc_id      as canonical_id,
    usd_per_kg,
    description as priced_as,
    note        as pricing_note
from {{ ref('cost_reference') }}
