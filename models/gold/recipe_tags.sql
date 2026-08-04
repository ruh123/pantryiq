-- Dietary tags derived from what a recipe actually resolves to, one row per (recipe, tag).
--
-- Tags are stated only for recipes whose ingredients are fully accounted for. A recipe with an
-- unresolved ingredient cannot be called vegan: the thing we could not identify is exactly the
-- thing that might disqualify it. So `nutrition_coverage` must be 1.0 before any tag is emitted —
-- silence is the safe answer.
--
-- **THIS IS AN ALLOWLIST OVER USDA FOOD GROUPS, NOT A DENYLIST OVER NAMES.** The distinction is
-- the whole model. An earlier version tagged a recipe vegetarian when no ingredient's
-- `display_name` matched any of 33 meat substrings, which cannot support a safety claim: every
-- product name nobody anticipated is a silent false positive. Two were found by joining Phase 4's
-- recipe titles against the tags — a signal that did not exist when the predicate was written:
--
--   recipenlg:9231  "Meat Loaf"  vegetarian   `hamburger`  -> "BURGER KING, Hamburger"
--   recipenlg:10493 "Meat Loaf"  gluten-free  `quaker oat` -> "Cereals, QUAKER, MultiGrain Oatmeal"
--
-- Neither string contains a listed token ('%ham,%' is comma-anchored so it will not match
-- `graham`; the gluten list had `oats`, not `oatmeal`). Measured against titles, 13 of 604
-- vegetarian and 9 of 147 vegan recipes named meat in their own title — a lower bound, since it
-- only catches recipes whose title gives them away.
--
-- USDA publishes exactly 25 food groups across these 8,187 foods (99.0% carry one). A closed
-- vocabulary can be reasoned about once and audited; an open set of product names cannot. So a
-- tag now requires EVERY ingredient to sit in an allowed group. "Fast Foods" and "Breakfast
-- Cereals" are simply not on the relevant lists, and both failures above decline by construction.
--
-- **The name patterns are kept as a SECOND filter, not a replacement.** Groups are coarse and
-- some of them mix: "Fats and Oils" holds lard and beef tallow, "Soups, Sauces, and Gravies"
-- holds beef broth, "Sweets" holds gelatin. The group allowlist catches what a name cannot see;
-- the name denylist catches what a group cannot. A tag requires both to agree.
--
-- A NULL group fails the allowlist rather than being skipped: `coalesce` makes an unknown food
-- disqualifying, which is the direction a safety claim has to fail in.
--
-- Still not an allergen guarantee — resolution is 67.3% accurate at the entity level, so an
-- ingredient can be mapped to the wrong food entirely. The README says so.

with lines as (

    select
        resolved.recipe_id,
        lower(coalesce(foods.display_name, ''))  as name,
        coalesce(foods.food_category, '')        as food_group
    from {{ ref('recipe_ingredients_resolved') }} as resolved
    left join {{ ref('canonical_ingredients') }} as foods
        on foods.canonical_id = resolved.canonical_id

),

complete_recipes as (

    select recipe_id
    from {{ ref('recipe_nutrition') }}
    where nutrition_coverage = 1.0

),

-- Recipes whose METHOD names a food their ingredient list does not.
--
-- `nutrition_coverage = 1.0` means "every line we have was weighed", not "we have every line",
-- and the corpus ships truncated ingredient lists. Bronze's "Pickled Bologna" lists vinegar,
-- sugar, salt and pickling spice — no bologna — so it is fully covered, fully resolved, sits
-- entirely inside allowed food groups, and is not vegan. The directions say "remove skin from 2
-- rings bologna". They are the only place in the corpus that gap is visible.
--
-- The signal is powered rather than assumed: this pattern fires on 3,083 of 15,000 recipes
-- (20.6%). A first attempt at the same measurement matched 0 of 15,000 because `directions` is
-- stored as a JSON *string* and was being joined as a list, which spaced out every character —
-- a check that cannot fire proves nothing, so the control was run before the veto was trusted.
suspect_directions as (

    select
        recipe_id,
        regexp_matches(coalesce(directions, ''),
            '\b(beef|pork|chicken|turkey|ham|bacon|sausage|meatloaf|meatball|steak|lamb|veal'
            '|burger|brisket|salami|pepperoni|venison|fish|salmon|tuna|shrimp|crab|lobster'
            '|clam|oyster|bologna|wiener|hotdog|liver|roast|gelatin|broth)\b') as names_meat,
        regexp_matches(coalesce(directions, ''),
            '\b(flour|bread|batter|dough|pastry|noodle|pasta|macaroni|spaghetti|cracker'
            '|crust|biscuit|cake mix|breadcrumb|bread crumb)\w*\b') as names_gluten
    from {{ source('silver', 'recipe_meta') }}

),

flags as (

    select
        lines.recipe_id,

        -- Groups that contain no animal flesh. The eleven excluded are Beef / Pork / Poultry /
        -- Lamb, Veal, and Game / Finfish and Shellfish / Sausages and Luncheon Meats — plus the
        -- composite groups whose contents are unknowable from the group alone: Fast Foods,
        -- Restaurant Foods, Meals, Entrees, and Side Dishes, Baby Foods, and American
        -- Indian/Alaska Native Foods.
        bool_and(lines.food_group in (
            'Vegetables and Vegetable Products', 'Fruits and Fruit Juices',
            'Legumes and Legume Products', 'Cereal Grains and Pasta',
            'Nut and Seed Products', 'Spices and Herbs', 'Dairy and Egg Products',
            'Beverages', 'Breakfast Cereals', 'Baked Products', 'Sweets',
            'Fats and Oils', 'Soups, Sauces, and Gravies', 'Snacks'
        )) as all_groups_meatless,

        -- The same list without Dairy and Egg Products.
        bool_and(lines.food_group in (
            'Vegetables and Vegetable Products', 'Fruits and Fruit Juices',
            'Legumes and Legume Products', 'Cereal Grains and Pasta',
            'Nut and Seed Products', 'Spices and Herbs',
            'Beverages', 'Breakfast Cereals', 'Baked Products', 'Sweets',
            'Fats and Oils', 'Soups, Sauces, and Gravies', 'Snacks'
        )) as all_groups_plant,

        -- Groups with no wheat, barley or rye in them. Plain meats belong here and baked goods,
        -- pasta, cereals and snacks do not. Soups, Sauces, and Gravies is excluded because
        -- thickening with flour is the norm rather than the exception.
        bool_and(lines.food_group in (
            'Vegetables and Vegetable Products', 'Fruits and Fruit Juices',
            'Legumes and Legume Products', 'Nut and Seed Products', 'Spices and Herbs',
            'Dairy and Egg Products', 'Fats and Oils', 'Beverages',
            'Beef Products', 'Pork Products', 'Poultry Products',
            'Lamb, Veal, and Game Products', 'Finfish and Shellfish Products'
        )) as all_groups_gluten_free,

        -- Flesh of any animal, for the ingredients that sit inside an allowed group anyway:
        -- lard and tallow are "Fats and Oils", beef broth is "Soups, Sauces, and Gravies".
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
            or name like '%tallow%' or name like '%suet%' or name like '%burger%'
        ) as has_meat,

        -- Anything from an animal at all. Honey and gelatin are the ones people miss.
        bool_or(
            name like '%milk%' or name like '%butter,%' or name like '%butter %'
            or name like '%cheese%' or name like '%cream%' or name like '%yogurt%'
            or name like '%egg%' or name like '%honey%' or name like '%mayonnaise%'
            or name like '%custard%' or name like '%whey%' or name like '%casein%'
            or name like '%ghee%' or name like '%ice cream%'
        ) as has_animal_product,

        -- Wheat, barley and rye reaching into an otherwise gluten-free group — soy sauce is
        -- "Soups, Sauces, and Gravies", malted milk is "Dairy and Egg Products".
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
            or name like '%stuffing%' or name like '%oat%' or name like '%malt%'
            or name like '%soy sauce%' or name like '%seitan%'
        ) as has_gluten_source

    from lines
    join complete_recipes using (recipe_id)
    group by 1

),

checked as (

    select flags.*, suspect.names_meat, suspect.names_gluten
    from flags
    left join suspect_directions as suspect using (recipe_id)

)

select recipe_id, 'vegetarian' as tag
from checked
where all_groups_meatless and not has_meat and not coalesce(names_meat, true)

union all

select recipe_id, 'vegan' as tag
from checked
where all_groups_plant and not has_meat and not has_animal_product
  and not coalesce(names_meat, true)

union all

select recipe_id, 'gluten-free' as tag
from checked
where all_groups_gluten_free and not has_gluten_source
  and not coalesce(names_gluten, true)
