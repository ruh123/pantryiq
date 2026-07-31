{# (recipe_id, line_index) is the grain of the resolved table. A duplicate would double-count
   that ingredient's mass, calories and cost into the recipe total — an error that looks like a
   plausible number rather than a crash. #}
{% test unique_line_per_recipe(model) %}

select recipe_id, line_index, count(*) as occurrences
from {{ model }}
group by 1, 2
having count(*) > 1

{% endtest %}
