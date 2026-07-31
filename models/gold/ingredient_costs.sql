-- The curated cost reference, as a Gold model.
--
-- **THESE PRICES ARE A STAND-IN, NOT SOURCED DATA.** Live grocery pricing is an explicit
-- non-goal (Master Prompt §4, §10); this is a hand-curated table of typical US retail prices so
-- cost-per-recipe is demonstrable at all. Unlike `gram_reference.csv` — whose values are USDA's
-- own published weights — nobody can verify these against a source, and any figure derived from
-- them must say so.
--
-- Coverage is 75.4% of converted ingredient MASS across 98 entities, which is the right
-- denominator: cost is per kilogram, so a gram of flour matters as much as a gram of saffron
-- costs differently. `cost_coverage` downstream reports it per recipe.
--
-- Prices are per entity, not per category, and an entity is priced for WHAT IT IS. Where a
-- resolution error routes volume to an odd entity (a fast-food burger, a babyfood jar), that
-- entity is still priced as itself — the error stays visible as a resolution error instead of
-- being laundered into the cost table.

select
    fdc_id      as canonical_id,
    usd_per_kg,
    description as priced_as,
    note        as pricing_note
-- Columns are declared rather than sniffed: the note field contains commas, and an
-- auto-detected delimiter silently read the whole header as one column.
from read_csv(
    'data/cost_reference.csv',
    header = true,
    delim = ',',
    quote = '"',
    columns = {'fdc_id': 'VARCHAR', 'usd_per_kg': 'DOUBLE',
               'description': 'VARCHAR', 'note': 'VARCHAR'}
)
