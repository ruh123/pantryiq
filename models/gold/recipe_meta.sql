-- Recipe titles and source URLs — what lets the serving layer name a recipe.
--
-- Every other Gold table keys on `recipe_id` and none of them carries a title, so the agent
-- could retrieve a perfect match and be unable to say what it was. A passthrough of
-- `silver.recipe_meta`, which lifts both fields out of the Bronze payload.
--
-- **`title` and `directions` are untrusted third-party text.** They come from a public web scrape.
-- `title` flows into a Claude prompt in Phase 4, so consumers must treat it as data — delimited
-- and explicitly ignored as instructions, never interpolated into a prompt as if it were trusted.
--
-- **`directions` is for DISPLAY ONLY and must never reach `AnswerContext`.** The context is the
-- guardrail's source of truth: every number in it becomes a value the model is permitted to
-- state. A method reading "bake at 425 for 45 minutes" would license 425 and 45 as quotable, so
-- "about 425 calories" would pass a check designed to stop exactly that. §16.3 already measures
-- 38% of injected fabrications colliding with some legitimate value; adding oven temperatures and
-- timings to every recipe's ledger would widen that hole for no gain, because the model has no
-- reason to discuss the method at all. The serving layer therefore reads this column on a
-- separate path from the one retrieval feeds.
--
-- It is worth showing for a second reason: `nutrition_coverage = 1.0` means every line we *have*
-- was weighed, not that the list is complete, and the corpus ships truncated recipes. The method
-- is the one field that can reveal the gap — "Pickled Bologna" lists no bologna and the
-- directions say otherwise. Putting it on screen turns a caveat into something visible.

select
    recipe_id,
    title,
    source_url,
    directions
from {{ source('silver', 'recipe_meta') }}
