-- The canonical USDA entity set, as the rest of Gold refers to it.
--
-- One row per food, carrying the per-100g nutrients everything downstream scales. Kept as its
-- own model rather than joined inline so referential integrity is testable: every resolved
-- ingredient must point at a row that exists here.

-- `category` and `food_category` are different things and both are kept. `category` is a derived
-- leading facet of the description ("beef", "bread", "millet flour"); `food_category` is USDA's
-- own food group, one of 25 closed values, which is what `recipe_tags` allows against. An earlier
-- version of that model tested `category like '%dairy%'` and matched zero rows for exactly this
-- reason.

select
    foods.fdc_id            as canonical_id,
    foods.description_raw   as display_name,
    foods.canonical_name,
    foods.category,
    groups.food_category,
    diet.is_vegetarian,
    diet.is_vegan,
    diet.is_gluten_free,
    foods.kcal_per_100g,
    foods.protein_g         as protein_g_per_100g,
    foods.fat_g             as fat_g_per_100g,
    foods.carb_g            as carb_g_per_100g
from {{ source('silver', 'usda_foods') }} as foods
left join {{ source('silver', 'usda_food_categories') }} as groups
    on groups.fdc_id = foods.fdc_id
left join {{ source('silver', 'entity_dietary_flags') }} as diet
    on diet.fdc_id = foods.fdc_id
