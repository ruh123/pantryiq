-- A tag must never survive an ingredient that is not classified as qualifying.
--
-- This is the project's highest-consequence assertion, and until now it lived only in pytest —
-- which means it ran on a developer machine against a pre-built export and could not block
-- anything. Three mutations to `recipe_tags.sql` (including adding 'Beef Products' to the old
-- allowlist) left all 520 Python tests green, because `make test` never invokes dbt.
--
-- As a singular dbt test it runs inside `dbt build --target staging`, so a false dietary claim
-- fails the gate and `publish()` never swaps it into `gold`.
--
-- `coalesce` is load-bearing: an entity with no classification row must disqualify, and a bare
-- `<> 'yes'` would let NULL through.

select
    tagged.recipe_id,
    tagged.tag,
    foods.display_name,
    foods.is_vegetarian,
    foods.is_vegan,
    foods.is_gluten_free
from {{ ref('recipe_tags') }} as tagged
join {{ ref('recipe_ingredients_resolved') }} as resolved
    on resolved.recipe_id = tagged.recipe_id
join {{ ref('canonical_ingredients') }} as foods
    on foods.canonical_id = resolved.canonical_id
where
    (tagged.tag = 'vegetarian' and coalesce(foods.is_vegetarian, 'unknown') <> 'yes')
    or (tagged.tag = 'vegan' and coalesce(foods.is_vegan, 'unknown') <> 'yes')
    or (tagged.tag = 'gluten-free' and coalesce(foods.is_gluten_free, 'unknown') <> 'yes')
