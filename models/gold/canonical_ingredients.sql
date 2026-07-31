-- The canonical USDA entity set, as the rest of Gold refers to it.
--
-- One row per food, carrying the per-100g nutrients everything downstream scales. Kept as its
-- own model rather than joined inline so referential integrity is testable: every resolved
-- ingredient must point at a row that exists here.

select
    fdc_id            as canonical_id,
    description_raw   as display_name,
    canonical_name,
    category,
    kcal_per_100g,
    protein_g         as protein_g_per_100g,
    fat_g             as fat_g_per_100g,
    carb_g            as carb_g_per_100g
from {{ source('silver', 'usda_foods') }}
