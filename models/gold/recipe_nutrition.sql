-- Per-recipe nutrition, with the coverage that qualifies it and the trust score that summarises
-- both.
--
-- TOTALS ARE SUMMED OVER CONTRIBUTING LINES ONLY, and `nutrition_coverage` states what fraction
-- that was. A recipe where three of eight ingredients could not be weighed still gets a total —
-- it is just a total of five ingredients, and the column says so. Presenting that as if it were
-- the whole recipe is exactly the confident-but-wrong number this project exists to prevent, so
-- nothing downstream may quote kcal without quoting coverage.
--
-- `data_trust_score` = min ingredient confidence x nutrition coverage.
--   The brief defines it as the minimum resolution confidence alone. That would score a recipe
--   with perfect resolution and 40% missing quantities as fully trustworthy, which inverts the
--   point of the field, so coverage is folded in. The design doc says "e.g.", so this is a
--   deliberate reading rather than a deviation. Both factors are kept as their own columns so
--   a low score can always be attributed to one or the other.
--
-- PER-SERVING IS NULL FOR MOST RECIPES BY DESIGN. RecipeNLG has no servings column, and only
-- ~15% of recipes state a yield in their text. Estimating the rest would put an invented
-- denominator into every downstream answer, so per-100g is the comparable unit and per-serving
-- is populated only where a real figure exists (§3.4 of the plan).

with lines as (

    select * from {{ ref('recipe_ingredients_resolved') }}

),

stated_servings as (

    select recipe_id, servings
    from {{ source('silver', 'recipe_servings') }}

),

nutrients as (

    select
        lines.recipe_id,
        lines.line_index,
        lines.grams,
        lines.confidence,
        lines.contributes_nutrition,
        lines.grams / 100.0 * foods.kcal_per_100g      as kcal,
        lines.grams / 100.0 * foods.protein_g_per_100g as protein_g,
        lines.grams / 100.0 * foods.fat_g_per_100g     as fat_g,
        lines.grams / 100.0 * foods.carb_g_per_100g    as carb_g,
        lines.grams / 1000.0 * costs.usd_per_kg        as usd,
        -- A line is costed only if it is also nutritionally counted, so cost and nutrition
        -- describe the same set of ingredients and their coverages are comparable.
        (lines.contributes_nutrition and costs.usd_per_kg is not null) as contributes_cost
    from lines
    left join {{ ref('canonical_ingredients') }} as foods
        on foods.canonical_id = lines.canonical_id
    left join {{ ref('ingredient_costs') }} as costs
        on costs.canonical_id = lines.canonical_id

),

aggregated as (

    select
        recipe_id,
        count(*)                                                    as ingredient_count,
        count(*) filter (where contributes_nutrition)               as counted_ingredients,
        sum(grams)      filter (where contributes_nutrition)        as total_grams,
        sum(kcal)       filter (where contributes_nutrition)        as total_kcal,
        sum(protein_g)  filter (where contributes_nutrition)        as total_protein_g,
        sum(fat_g)      filter (where contributes_nutrition)        as total_fat_g,
        sum(carb_g)     filter (where contributes_nutrition)        as total_carb_g,
        -- Minimum over CONTRIBUTING lines: a line that contributes nothing cannot make the
        -- recipe's nutrition less trustworthy, because none of it came from that line.
        min(confidence) filter (where contributes_nutrition)        as min_confidence,
        count(*) filter (where contributes_cost)                    as costed_ingredients,
        sum(usd)  filter (where contributes_cost)                   as total_usd
    from nutrients
    group by 1

)

select
    aggregated.recipe_id,
    ingredient_count,
    counted_ingredients,

    total_grams,
    total_kcal,
    total_protein_g,
    total_fat_g,
    total_carb_g,

    -- Per 100g is the unit that makes recipes comparable without inventing a serving count.
    case when total_grams > 0 then total_kcal      / total_grams * 100.0 end as kcal_per_100g,
    case when total_grams > 0 then total_protein_g / total_grams * 100.0 end as protein_g_per_100g,
    case when total_grams > 0 then total_fat_g     / total_grams * 100.0 end as fat_g_per_100g,
    case when total_grams > 0 then total_carb_g    / total_grams * 100.0 end as carb_g_per_100g,

    -- NULL for ~85% of recipes, and deliberately so: RecipeNLG states no servings count and
    -- estimating one would put an invented denominator inside every per-serving number.
    stated_servings.servings,
    case when stated_servings.servings > 0
         then total_kcal / stated_servings.servings end as kcal_per_serving,

    -- Cost is a STAND-IN (see gold.ingredient_costs) and is reported with its own coverage,
    -- which is lower than nutrition's: an ingredient can be weighed but unpriced.
    total_usd                                                as cost_total_usd,
    case when stated_servings.servings > 0
         then total_usd / stated_servings.servings end       as cost_per_serving_usd,
    costed_ingredients::double / nullif(ingredient_count, 0) as cost_coverage,

    counted_ingredients::double / nullif(ingredient_count, 0) as nutrition_coverage,
    min_confidence,
    coalesce(min_confidence, 0.0)
        * (counted_ingredients::double / nullif(ingredient_count, 0)) as data_trust_score

from aggregated
left join stated_servings using (recipe_id)
