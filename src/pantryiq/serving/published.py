"""Every number the web app states, in one place, with the document each one comes from.

The recurring failure in this project is not a wrong measurement — it is a right measurement that
drifts. §15's guardrail figure survived in the journal for 140 lines after being retracted; the
test count reads 582, 581, 556 and 404 in four places. Putting a fourth surface (a screen) in
front of the same numbers without a single source would guarantee a fifth version of each.

So the app imports from here, never from a literal, and `tests/test_published_constants.py`
asserts these still appear in `README.md`. Changing a figure on screen therefore means changing
the README, which is the point: it makes the change deliberate rather than accidental.

Values are strings, formatted as they are shown. They are quotations from a document, not inputs
to a calculation — rounding them here would be inventing precision the source does not have.
"""
from __future__ import annotations

# --- the corpus this was all measured on -------------------------------------------------------
# docs/er_metrics.md §7, README "Results". The scale gate in the brief's §4 anticipated 100-500K
# recipes; it was never taken, so every figure below is a subset figure and the app says so.
RECIPES = "15,000"
INGREDIENT_LINES = "112,463"
DISTINCT_STRINGS = "9,163"
USDA_ENTITIES = "8,187"

# --- entity resolution, the headline (README "Results", docs/er_metrics.md §11) -----------------
# Rows are (what was measured, per unique string, per occurrence). The per-occurrence column is
# the one that describes the product: head strings are 5.9% of the vocabulary and 83.8% of what a
# user actually hits.
ENTITY_RESOLUTION = (
    ("nutrition within 10%, or within 5 kcal/100g", "69.3%  [60.2, 78.1]", "81.0%  [69.3, 88.0]"),
    ("entity top-1", "47.7%  [37.9, 57.6]", "67.3%  [49.5, 78.5]"),
)
ENTITY_TOP1 = "67.3%"
RESOLVER_COMMITS = "68.5%"
RESOLVER_DECLINES = "31.5%"

# --- the guardrail (docs/er_metrics.md §16.3) --------------------------------------------------
GUARDRAIL_CATCH_RATE = "86.5%"
GUARDRAIL_CAUGHT = "147/170"
GUARDRAIL_INTERVAL = "80.5-90.8%"
GUARDRAIL = (
    ("injected fabrications caught", "147 of 170  (86.5%)"),
    ("misattributed figures caught", "20 of 20"),
    ("division-by-servings caught", "10 of 10"),
    ("false positives on unmodified answers", "0 genuine, of 20"),
    ("weakest class - invented whole-dollar cost", "10 of 20"),
    ("weakest class - serving count, reworded", "14 of 20"),
)

# --- coverage: what the warehouse can and cannot answer (docs/er_metrics.md §14) ----------------
LINES_WEIGHED = "63.9%"
GRAM_ACCURACY = "77.6%"
COMPLETE_NUTRITION = "4.74%"
COMPLETE_NUTRITION_N = "711"
COMPLETE_COST_N = "111"
COVERAGE = (
    ("ingredient lines converted to a weight", "63.9%"),
    ("gram conversion accuracy", "77.6%"),
    ("recipes with complete nutrition", "4.74%  (711 of 15,000)"),
    ("recipes with complete cost", "0.74%  (111 of 15,000)"),
    ("recipes publishing a per-serving calorie figure", "118"),
)

# --- dietary tags (docs/er_metrics.md §16.4) ---------------------------------------------------
TAGS = (("vegetarian", "418"), ("vegan", "61"), ("gluten-free", "179"))

# --- label quality: the caveat under everything above ------------------------------------------
LLM_LABELS = "282 of the 300"
KRIPPENDORFF = "0.721  [0.651, 0.787]"

# --- what the screen must say, whether or not the prose happens to mention it -------------------
# Each of these is a limitation that a fluent answer can hide. They are structural, so they are
# shown structurally rather than left to the model to remember.
CAVEATS = (
    (
        "Every calorie figure travels with its coverage",
        "A recipe at 0.8 coverage has a fifth of its ingredients unweighed, so the total is a "
        "sum over the part that could be weighed. It is an undercount, not the dish's energy. "
        f"Only {COMPLETE_NUTRITION} of recipes ({COMPLETE_NUTRITION_N}) are weighed completely.",
    ),
    (
        "Full coverage does not mean the ingredient list is complete",
        "The source corpus ships truncated recipes. \"Pickled Bologna\" lists vinegar, sugar, "
        "salt and pickling spice, and no bologna. Coverage of 1.0 means every line we have was "
        "weighed, not that we have every line - and nothing downstream of the list can see the "
        "difference.",
    ),
    (
        "Prices are a stand-in, not real pricing",
        "A hand-curated table of typical US prices for 98 ingredients. Live grocery pricing is "
        "an explicit non-goal. Where only some ingredients are priced, the total is a floor and "
        "is shown as \"at least\".",
    ),
    (
        "Per-recipe is not per-serving, and nothing divides",
        "Only 118 recipes carry a serving count alongside complete coverage. Dividing a partial "
        "total by a real denominator produces a confident number for an energy value known to be "
        "short, so the pipeline refuses to publish it and the guardrail rejects it.",
    ),
    (
        "Dietary tags are decided per entity, and entity resolution is imperfect",
        f"A tag requires every ingredient to qualify and \"unknown\" disqualifies, but the "
        f"classifier judges the entity an ingredient resolved to, and that resolution is "
        f"{ENTITY_TOP1} accurate. Three extra filters exist for exactly that reason. Do not rely "
        f"on these for an allergy.",
    ),
    (
        "The accuracy figures are agreement with AI-produced labels",
        f"{LLM_LABELS} gold labels were produced by an LLM annotator rather than a human. Three "
        f"independent annotators give Krippendorff's alpha = {KRIPPENDORFF}, which straddles the "
        f"usable floor - so the interval is quoted, never the point estimate.",
    ),
    (
        "There is no concept of a meal type",
        "The data carries no breakfast/dinner/dessert information, so a request for a vegetarian "
        "dinner can return cookies. The agent says so rather than pretending otherwise.",
    ),
    (
        "The guardrail catches most fabrications, not all",
        f"{GUARDRAIL_CAUGHT} injected errors ({GUARDRAIL_CATCH_RATE}, 95% CI "
        f"{GUARDRAIL_INTERVAL}) across nine classes. Invented whole-dollar costs and reworded "
        f"serving counts are the weak ones, because small integers are dense in the set of "
        f"legitimate values.",
    ),
)

# The four stages, for the "how this works" tab. Only two involve a model, and that ordering is
# the design rather than an implementation detail.
PIPELINE = (
    ("parse", "Claude", "turns the question into a fixed filter schema. It may shape the search."),
    ("retrieve", "SQL only", "ranks recipes in the warehouse. The model has no say in this."),
    ("generate", "Claude", "sees only what retrieval returned. It may phrase, not source."),
    ("guardrail", "code only", "every number in the answer must trace back to that data."),
)
