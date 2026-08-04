-- An independent check on the same claim, using USDA's food groups rather than the classifier.
--
-- The classifier decides the tag, so asserting the tag against the classifier is a restatement.
-- `food_category` comes from USDA and is not read by `recipe_tags.sql` at all, which makes this
-- the one assertion here that could disagree with the rule it guards.

select
    tagged.recipe_id,
    tagged.tag,
    foods.display_name,
    foods.food_category
from {{ ref('recipe_tags') }} as tagged
join {{ ref('recipe_ingredients_resolved') }} as resolved
    on resolved.recipe_id = tagged.recipe_id
join {{ ref('canonical_ingredients') }} as foods
    on foods.canonical_id = resolved.canonical_id
where tagged.tag in ('vegetarian', 'vegan')
  and foods.food_category in (
      'Beef Products', 'Pork Products', 'Poultry Products',
      'Lamb, Veal, and Game Products', 'Finfish and Shellfish Products',
      'Sausages and Luncheon Meats')
