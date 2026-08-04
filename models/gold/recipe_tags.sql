-- Dietary tags, one row per (recipe, tag).
--
-- **A tag requires every ingredient to be classified as qualifying, per entity.** Not a food
-- group, not a substring of a name — the property itself, decided once per USDA entity by
-- `er/dietary.py` and stored in `silver.entity_dietary_flags`.
--
-- This is the third rule in this position, and the first two failed the same way. Phase 3 asked
-- "does this food's name contain a meat word"; Phase 4 asked "is this food's USDA group allowed".
-- Both are PROXIES for a property of the food, and both leaked wherever the proxy and the
-- property came apart:
--
--   Sauce, worcestershire      group allowed, no meat word      -> anchovies      (12 veg, 6 vegan)
--   Candies, marshmallows      group allowed, no meat word      -> gelatin        (17 veg, 4 vegan)
--   Bacon, meatless            a legume, correctly              -> wheat gluten, not GF
--   Alcoholic beverage, beer   no gluten word                   -> barley malt
--
-- Patching those four fixes those four. The measured false-tag rate under the group allowlist was
-- 30 of 572 vegetarian (5.2%) and 16 of 134 vegan (11.9%) by entity inspection — six and four
-- times what the title-based measurement reported. A rule whose failures are unbounded cannot
-- support a safety claim, so the question is now asked directly and a food nobody has looked at
-- is classified the same way as one that has.
--
-- **`unknown` disqualifies, exactly like `no`.** 705 of 8,187 entities are genuinely
-- unclassifiable from their description ('Sauce, unspecified'). Silence is the safe answer.
--
-- **Three further filters are kept, because each catches something the classifier cannot see:**
--
--  1. `nutrition_coverage = 1.0` — an unidentified ingredient is exactly the one that might
--     disqualify the recipe.
--  2. The **name denylist** — the classifier judges the ENTITY, and entity resolution is 67.3%
--     accurate. Three recipes containing real chicken, bacon and sausage resolved to the
--     *meatless* analogues, which are correctly vegetarian as entities. Only the name check
--     stopped those being tagged.
--  3. The **directions veto** — `coverage = 1.0` means "every line we have was weighed", not "we
--     have every line". Bronze's "Pickled Bologna" lists vinegar, sugar, salt and pickling spice
--     and no bologna; the method says "remove skin from 2 rings bologna".
--
-- The veto patterns end `)\w*\b`, not `)\b`. The earlier version matched no inflected form at
-- all — `hamburger`, `sausages`, `steaks`, `chickens` all missed — while the gluten pattern one
-- line below had the suffix. Adding it moves the veto from 3,309 to 3,609 of 15,000 recipes.
--
-- Still not an allergen guarantee. It rests on entity resolution being right about which food a
-- line refers to, and that is measured at 67.3% at the entity level. The README says so.

with lines as (

    select
        resolved.recipe_id,
        lower(coalesce(foods.display_name, '')) as name,
        lower(coalesce(resolved.normalized_text, '')) as line,
        coalesce(foods.is_vegetarian, 'unknown') as is_vegetarian,
        coalesce(foods.is_vegan, 'unknown')      as is_vegan,
        coalesce(foods.is_gluten_free, 'unknown') as is_gluten_free
    from {{ ref('recipe_ingredients_resolved') }} as resolved
    left join {{ ref('canonical_ingredients') }} as foods
        on foods.canonical_id = resolved.canonical_id

),

complete_recipes as (

    select recipe_id
    from {{ ref('recipe_nutrition') }}
    where nutrition_coverage = 1.0

),

-- Recipes whose METHOD names a food their ingredient list does not. See note 3 above. The signal
-- is powered rather than assumed: the meat pattern fires on 22.1% of all 15,000 recipes. An
-- earlier version of this measurement matched 0 of 15,000 because `directions` is stored as a
-- JSON *string* and was being joined as a list, spacing out every character — a check that
-- cannot fire proves nothing, so the control was run before the veto was trusted.
suspect_directions as (

    select
        recipe_id,
        regexp_matches(coalesce(directions, ''),
            '\b(beef|pork|chicken|turkey|ham|bacon|sausage|meat|meatloaf|meatball|steak|lamb'
            '|veal|burger|brisket|salami|pepperoni|venison|fish|salmon|tuna|shrimp|crab|lobster'
            '|clam|oyster|bologna|wiener|frankfurter|hotdog|liver|roast|rib|anchov|gelatin'
            '|broth|bouillon|lard|suet|tallow|duck|goose|bison|seafood|prawn|worcestershire'
            ')\w*\b') as names_meat,
        regexp_matches(coalesce(directions, ''),
            '\b(milk|butter|cheese|cream|yogurt|egg|honey|mayonnaise|marshmallow|custard'
            '|whey|casein|ghee|buttermilk|sherbet'
            ')\w*\b') as names_animal,
        regexp_matches(coalesce(directions, ''),
            '\b(flour|bread|batter|dough|pastry|noodle|pasta|macaroni|spaghetti|cracker'
            '|crust|biscuit|breadcrumb|wheat|barley|rye|malt|oat|cake mix|graham|cereal'
            ')\w*\b') as names_gluten
    from {{ source('silver', 'recipe_meta') }}

),

flags as (

    select
        lines.recipe_id,

        bool_and(lines.is_vegetarian = 'yes')  as every_entity_vegetarian,
        bool_and(lines.is_vegan = 'yes')       as every_entity_vegan,
        bool_and(lines.is_gluten_free = 'yes') as every_entity_gluten_free,

        -- Note 2: the classifier judges the entity; this catches the entity being the wrong one.
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
            or name like '%surimi%' or name like '%worcestershire%'
        ) as has_meat_name,

        -- The same idea applied to the RECIPE'S OWN WORDS rather than the entity's. Both catch
        -- the entity being the wrong one, and they catch different instances of it: a line
        -- reading "chicken breast" that resolved to something harmless is invisible to the
        -- display-name check.
        bool_or(
            line like '%beef%' or line like '%pork%' or line like '%chicken%'
            or line like '%turkey%' or line like '%bacon%' or line like '%sausage%'
            or line like '%fish%' or line like '%shrimp%' or line like '%crab%'
            or line like '%lamb%' or line like '%veal%' or line like '%venison%'
            or line like '%anchov%' or line like '%gelatin%' or line like '%lard%'
            or line like '%broth%' or line like '%bouillon%' or line like '%worcestershire%'
            or line like '%meat%' or line like '%ham %' or line like '%liver%'
        ) as has_meat_line,

        -- Gluten, judged on the recipe's words. "Icebox Cookies" writes `flour` and it resolved
        -- to *Millet flour* — which is genuinely gluten-free as an entity, so the classifier was
        -- right and the tag was still wrong. In American home cooking a bare "flour" means
        -- wheat, so an unqualified gluten word disqualifies unless the line names a
        -- gluten-free grain explicitly.
        -- `regexp_matches`, not `SIMILAR TO`: DuckDB's SIMILAR TO is anchored regex in which `%`
        -- is a LITERAL percent sign, so `'%(flour|...)%'` matched nothing at all and this check
        -- was dead on arrival. Verified against the real strings before being trusted — a check
        -- that cannot fire is not a check, which is the third time that lesson has come up here.
        bool_or(
            regexp_matches(line, '\b(flour|bread|cracker|oats|oatmeal|barley|rye|wheat|pasta'
                                 '|noodle|macaroni|spaghetti|graham|biscuit|semolina'
                                 '|couscous|bulgur|crouton|stuffing|pretzel|malt)\w*\b')
            and not regexp_matches(line, '\b(rice|corn|almond|coconut|millet|chestnut|arrowroot'
                                         '|tapioca|buckwheat|potato|quinoa|soy)\w*\b')
        ) as has_gluten_line

    from lines
    join complete_recipes using (recipe_id)
    group by 1

),

checked as (

    select flags.*, suspect.names_meat, suspect.names_animal, suspect.names_gluten
    from flags
    left join suspect_directions as suspect using (recipe_id)

)

select recipe_id, 'vegetarian' as tag
from checked
where every_entity_vegetarian
  and not has_meat_name and not has_meat_line
  and not coalesce(names_meat, true)

union all

select recipe_id, 'vegan' as tag
from checked
where every_entity_vegan
  and not has_meat_name and not has_meat_line
  and not coalesce(names_meat, true)
  and not coalesce(names_animal, true)

union all

select recipe_id, 'gluten-free' as tag
from checked
where every_entity_gluten_free
  and not has_gluten_line
  and not coalesce(names_gluten, true)
