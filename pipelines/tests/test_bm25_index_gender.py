from pipelines.index.bm25 import _derived_gender_label


def test_derived_gender_prefers_category_path():
    got = _derived_gender_label(
        ["Clothing, Shoes & Jewelry", "Men", "Shoes"],
        "Nike Women's Air Max Shoes",
    )

    assert got == "men"


def test_derived_gender_uses_title_when_category_path_is_ambiguous():
    got = _derived_gender_label([], "Nike Women's Air Max Shoes")

    assert got == "women"


def test_derived_gender_keeps_unisex_distinct():
    got = _derived_gender_label([], "Nike Unisex Running Shoes")

    assert got == "unisex"
