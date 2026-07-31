-- One row per recipe ingredient line, joined to its entity and its mass.
--
-- This is the grain everything above it aggregates from, so it keeps EVERY line — including
-- the ones that could not be resolved or could not be weighed. Filtering them here would make
-- a recipe with four missing ingredients indistinguishable from one with none, and
-- `nutrition_coverage` downstream exists precisely to tell those apart.
--
-- `contributes_nutrition` is the flag that decides whether a line can enter a recipe total: it
-- needs an entity the resolver did not decline, a gram weight, and an energy value.

with lines as (

    select
        recipe_id,
        line_index,
        line_raw,
        quantity,
        unit,
        normalized_text
    from {{ source('silver', 'recipe_ingredient_lines') }}

),

resolved as (

    select
        normalized_text,
        fdc_id,
        confidence,
        flagged,
        abstained
    from {{ source('silver', 'ingredient_entity_map') }}

),

weighed as (

    select
        recipe_id,
        line_index,
        grams,
        method as gram_method
    from {{ source('silver', 'recipe_ingredient_grams') }}

)

select
    lines.recipe_id,
    lines.line_index,
    lines.line_raw,
    lines.quantity,
    lines.unit,
    lines.normalized_text,

    -- A declined pick is NOT an entity. §12 fits a threshold precisely so the resolver can say
    -- "no confident match", and carrying its best guess through anyway would reintroduce the
    -- confident-wrong-nutrition failure that threshold exists to prevent.
    case when resolved.abstained then null else resolved.fdc_id end as canonical_id,
    resolved.confidence,
    coalesce(resolved.flagged, false)   as flagged,
    coalesce(resolved.abstained, false) as abstained,

    weighed.grams,
    weighed.gram_method,

    (
        resolved.fdc_id is not null
        and not resolved.abstained
        and weighed.grams is not null
        and foods.kcal_per_100g is not null
    ) as contributes_nutrition

from lines
left join resolved
    on resolved.normalized_text = lines.normalized_text
left join weighed
    on weighed.recipe_id = lines.recipe_id
   and weighed.line_index = lines.line_index
left join {{ ref('canonical_ingredients') }} as foods
    on foods.canonical_id = resolved.fdc_id
   and not coalesce(resolved.abstained, false)
