{# A numeric bounds test, defined here rather than pulled from dbt_utils: the project needs
   exactly this one check and a package dependency for it would be more moving parts than the
   twelve lines it replaces. Nulls pass — "no value" is a coverage question, which
   nutrition_coverage answers; this test is about values that are impossible. #}
{% test accepted_range(model, column_name, min_value=none, max_value=none, inclusive=true) %}

select {{ column_name }} as failing_value
from {{ model }}
where {{ column_name }} is not null
  and (
    {%- if min_value is not none %}
      {{ column_name }} {{ '<' if inclusive else '<=' }} {{ min_value }}
      {%- if max_value is not none %} or {% endif %}
    {%- endif %}
    {%- if max_value is not none %}
      {{ column_name }} {{ '>' if inclusive else '>=' }} {{ max_value }}
    {%- endif %}
  )

{% endtest %}
