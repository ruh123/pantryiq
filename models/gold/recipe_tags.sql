-- Dietary tags derived from what a recipe actually resolves to, one row per (recipe, tag).
--
-- Tags are stated as ABSENCE OF A DISQUALIFIER, and only for recipes whose ingredients are
-- fully accounted for. A recipe with an unresolved ingredient cannot be called vegan: the thing
-- we could not identify is exactly the thing that might disqualify it. So `nutrition_coverage`
-- must be 1.0 before any tag is emitted — silence is the safe answer.
--
-- **MATCHING IS ON `display_name`, NOT `category`.** An earlier version tested
-- `category like '%cereal grain%'`, `'%dairy%'`, `'%finfish%'`, `'%shellfish%'` — all four
-- matched ZERO rows, because `silver.usda_foods.category` is a derived leading-comma facet
-- ("beef", "bread", "millet flour"), not USDA's food-group vocabulary. The tags therefore rested
-- on a handful of left-anchored `display_name` patterns and leaked badly: 40 of 429 gluten-free
-- recipes contained gluten (pie crust, chow mein noodles, cake mix) and 9 of 170 vegan recipes
-- contained honey, gelatin or shrimp. Patterns below are unanchored and cover the food families
-- that actually appear.
--
-- Still coarse, and still NOT an allergen guarantee: it is substring matching over USDA
-- descriptions, so a novel product name can slip through. The README says so.

with lines as (

    select
        resolved.recipe_id,
        lower(coalesce(foods.display_name, '')) as name
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

        -- Flesh of any animal, including the fish and shellfish families the old category
        -- predicates silently missed entirely.
        bool_or(
            name like '%beef%' or name like '%pork%' or name like '%chicken%'
            or name like '%turkey%' or name like '%lamb%' or name like '%veal%'
            or name like '%bacon%' or name like '%sausage%' or name like '%ham,%'
            or name like '%fish%' or name like '%salmon%' or name like '%tuna%'
            or name like '%shrimp%' or name like '%crustacean%' or name like '%mollusk%'
            or name like '%crab%' or name like '%lobster%' or name like '%clam%'
            or name like '%oyster%' or name like '%anchov%' or name like '%bison%'
            or name like '%venison%' or name like '%duck%' or name like '%goose%'
            or name like '%liver%' or name like '%bologna%' or name like '%pepperoni%'
            or name like '%salami%' or name like '%frankfurter%' or name like '%meat%'
            or name like '%broth%' or name like '%gelatin%' or name like '%lard%'
        ) as has_meat,

        -- Anything from an animal at all. Honey and gelatin are the ones people miss.
        bool_or(
            name like '%milk%' or name like '%butter,%' or name like '%butter %'
            or name like '%cheese%' or name like '%cream%' or name like '%yogurt%'
            or name like '%egg%' or name like '%honey%' or name like '%mayonnaise%'
            or name like '%custard%' or name like '%whey%' or name like '%casein%'
            or name like '%ghee%' or name like '%ice cream%'
        ) as has_animal_product,

        -- Wheat, barley, rye, and the prepared foods made from them.
        bool_or(
            name like '%wheat%' or name like '%flour%' or name like '%bread%'
            or name like '%barley%' or name like '%rye%' or name like '%bulgur%'
            or name like '%couscous%' or name like '%semolina%' or name like '%farina%'
            or name like '%macaroni%' or name like '%noodle%' or name like '%spaghetti%'
            or name like '%pasta%' or name like '%cracker%' or name like '%cookie%'
            or name like '%cake%' or name like '%pie crust%' or name like '%crust,%'
            or name like '%crouton%' or name like '%pretzel%' or name like '%graham%'
            or name like '%biscuit%' or name like '%tortilla, flour%' or name like '%bagel%'
            or name like '%muffin%' or name like '%doughnut%' or name like '%pastry%'
            or name like '%stuffing%' or name like '%oats%' or name like '%malt%'
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
