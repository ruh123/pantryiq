-- The recipe's OWN words, not the entity's — the check that catches entity resolution being
-- wrong rather than the classifier being wrong.
--
-- "Icebox Cookies" writes `flour`, which resolved to *Millet flour*. Millet flour is genuinely
-- gluten-free, so the classifier was right and the tag was still false. In American home cooking
-- an unqualified "flour" means wheat.

select
    tagged.recipe_id,
    resolved.normalized_text
from {{ ref('recipe_tags') }} as tagged
join {{ ref('recipe_ingredients_resolved') }} as resolved
    on resolved.recipe_id = tagged.recipe_id
where tagged.tag = 'gluten-free'
  and regexp_matches(lower(resolved.normalized_text),
      '\b(flour|bread|oats|wheat|barley|rye|pasta|noodle|macaroni|graham|biscuit)\w*\b')
  and not regexp_matches(lower(resolved.normalized_text),
      '\b(rice|corn|almond|coconut|millet|chestnut|arrowroot|tapioca|buckwheat|potato)\w*\b')
