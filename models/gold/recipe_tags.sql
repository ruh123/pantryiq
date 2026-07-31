-- Dietary tags derived from what a recipe actually resolves to, one row per (recipe, tag).
--
-- Tags are stated as ABSENCE OF A DISQUALIFIER, and only for recipes whose ingredients are
-- fully accounted for. A recipe with an unresolved ingredient cannot be called vegan: the thing
-- we could not identify is exactly the thing that might disqualify it. So `nutrition_coverage`
-- must be 1.0 before any tag is emitted — silence is the safe answer, and a false "vegan" tag
-- is the kind of claim that matters to someone.
--
-- Matching is on the USDA category and description, which is coarse. These are a demo-able
-- convenience, NOT an allergen guarantee, and the README must say so.

with lines as (

    select
        resolved.recipe_id,
        lower(coalesce(foods.category, ''))     as category,
        lower(coalesce(foods.display_name, '')) as display_name
    from {{ ref('recipe_ingredients_resolved') }} as resolved
    left join {{ ref('canonical_ingredients') }} as foods
        on foods.canonical_id = resolved.canonical_id

),

complete_recipes as (

    select recipe_id
    from {{ ref('recipe_nutrition') }}
    where nutrition_coverage = 1.0

),

flags as (

    select
        lines.recipe_id,

        bool_or(
            category like '%beef%' or category like '%pork%' or category like '%poultry%'
            or category like '%sausage%' or category like '%lamb%' or category like '%finfish%'
            or category like '%shellfish%'
            or display_name like 'beef,%' or display_name like 'pork,%'
            or display_name like 'chicken,%' or display_name like 'turkey,%'
            or display_name like 'fish,%' or display_name like 'bacon%'
        ) as has_meat,

        bool_or(
            category like '%dairy%' or category like '%egg%'
            or display_name like 'milk,%' or display_name like 'butter,%'
            or display_name like 'cheese,%' or display_name like 'cream,%'
            or display_name like 'yogurt,%' or display_name like 'egg,%'
        ) as has_animal_product,

        bool_or(
            category like '%cereal grain%' or category like '%baked%'
            or display_name like 'wheat%' or display_name like 'bread,%'
            or display_name like 'flour%' or display_name like '%barley%'
            or display_name like '%rye%'
        ) as has_gluten_source

    from lines
    join complete_recipes using (recipe_id)
    group by 1

)

select recipe_id, 'vegetarian' as tag from flags where not has_meat
union all
select recipe_id, 'vegan' as tag from flags where not has_meat and not has_animal_product
union all
select recipe_id, 'gluten-free' as tag from flags where not has_gluten_source
