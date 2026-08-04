-- Recipe titles and source URLs — what lets the serving layer name a recipe.
--
-- Every other Gold table keys on `recipe_id` and none of them carries a title, so the agent
-- could retrieve a perfect match and be unable to say what it was. A passthrough of
-- `silver.recipe_meta`, which lifts both fields out of the Bronze payload.
--
-- **`title` is untrusted third-party text.** It comes from a public web scrape and flows into a
-- Claude prompt in Phase 4, so consumers must treat it as data — delimited and explicitly
-- ignored as instructions, never interpolated into a prompt as if it were trusted.

select
    recipe_id,
    title,
    source_url
from {{ source('silver', 'recipe_meta') }}
