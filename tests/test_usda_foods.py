"""Silver USDA normalization: energy mapping, facet handling, deprioritization."""
import pyarrow as pa
import pytest

from pantryiq.silver.usda_foods import (
    build_usda_rows,
    canonical_name,
    category,
    is_deprioritized,
    kcal_per_100g,
    search_text,
)


def _food(*nutrients):
    return {"foodNutrients": [dict(zip(("number", "name", "amount", "unitName"), n))
                              for n in nutrients]}


def test_energy_prefers_kcal_over_kilojoules():
    """7,887 foods carry both 208 [KCAL] and 268 [kJ]; picking kJ inflates energy 4.184x."""
    food = _food(("268", "Energy", 288.0, "kJ"), ("208", "Energy", 69.0, "KCAL"))
    assert kcal_per_100g(food) == 69.0


def test_energy_maps_atwater_variants():
    """~300 Foundation foods report energy only under an Atwater name."""
    assert kcal_per_100g(_food(("957", "Energy (Atwater General Factors)", 210.0, "KCAL"))) == 210.0
    assert kcal_per_100g(_food(("958", "Energy (Atwater Specific Factors)", 205.0, "KCAL"))) == 205.0


def test_energy_prefers_specific_over_general_atwater():
    """199 Foundation foods carry both; USDA treats Specific Factors as the more accurate."""
    food = _food(("957", "Energy (Atwater General Factors)", 210.0, "KCAL"),
                 ("958", "Energy (Atwater Specific Factors)", 205.0, "KCAL"))
    assert kcal_per_100g(food) == 205.0


def test_energy_absent_is_none_not_zero():
    """73 Foundation foods have no energy nutrient — that is missing, not 0 kcal."""
    assert kcal_per_100g(_food(("203", "Protein", 5.0, "G"))) is None


def test_facet_forms():
    description = "Egg, whole, raw, frozen"
    assert canonical_name(description) == "egg whole raw frozen"
    assert search_text(description) == "frozen raw whole egg"
    assert category(description) == "egg"


def test_facet_forms_handle_uninverted_descriptions():
    assert canonical_name("Flour, 00") == "flour 00"
    assert search_text("Tofu") == "tofu"
    assert category("Tofu") == "tofu"


@pytest.mark.parametrize("description,flagged", [
    ("Babyfood, dessert, custard pudding, vanilla, junior", True),
    ("APPLEBEE'S, crunchy onion rings", True),
    ("Frozen novelties, No Sugar Added CREAMSICLE Pops", True),
    ("Egg, whole, raw, fresh", False),
    ("Butter, stick, salted", False),
    ("Cheese, cottage, lowfat, 1% milkfat", False),
    # The FDC boilerplate's "USDA'" reads as a brand token — it flagged 58 staples.
    ("Apples, raw, with skin (Includes foods for USDA's Food Distribution Program)", False),
    ("Beans, great northern, mature seeds, raw (Includes foods for USDA's Food Distribution Program)", False),  # noqa: E501
])
def test_deprioritization(description, flagged):
    assert is_deprioritized(description) is flagged


def test_boilerplate_stripped_but_synonyms_kept():
    """Parentheticals are mostly synonyms recipes use — only the FDC boilerplate is noise."""
    assert search_text("Apples, raw, with skin (Includes foods for USDA's Food Distribution Program)") == "with skin raw apples"  # noqa: E501
    assert "scallion" in search_text("Green onion, (scallion), bulb and greens, raw")
    assert "pepitas" in search_text("Seeds, pumpkin (pepitas), raw")


class _FakeBronze:
    def __init__(self, rows):
        self._rows = rows

    def load_table(self, name):
        assert name == "bronze.raw_usda_foods"
        return self

    def scan(self):
        return self

    def to_arrow(self):
        return pa.table({
            "fdc_id": [r[0] for r in self._rows],
            "data_type": [r[1] for r in self._rows],
            "raw_payload": [r[2] for r in self._rows],
        })


def test_build_usda_rows():
    import json

    payload = json.dumps({
        "description": "Cheese, cheddar",
        "foodNutrients": [
            {"number": "208", "name": "Energy", "amount": 403.0, "unitName": "KCAL"},
            {"number": "268", "name": "Energy", "amount": 1687.0, "unitName": "kJ"},
            {"number": "203", "name": "Protein", "amount": 22.9, "unitName": "G"},
        ],
    })
    rows = build_usda_rows(_FakeBronze([("1105", "Foundation", payload)]))

    assert rows.num_rows == 1
    assert rows.column("kcal_per_100g").to_pylist() == [403.0]
    assert rows.column("protein_g").to_pylist() == [22.9]
    assert rows.column("fat_g").to_pylist() == [None]  # absent, not zero
    assert rows.column("search_text").to_pylist() == ["cheddar cheese"]
    assert rows.column("category").to_pylist() == ["cheese"]
