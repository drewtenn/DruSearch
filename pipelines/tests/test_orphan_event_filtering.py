from __future__ import annotations

import inspect

import pandas as pd

from pipelines.features import aggregates
from pipelines.label import build_training_rows


def _normalized_sql(text: str) -> str:
    return " ".join(text.lower().split())


def test_product_aggregates_filter_events_to_current_products():
    sql = _normalized_sql(aggregates.SQL)

    assert "join products" in sql
    assert "using (product_id)" in sql


def test_training_rows_loads_only_events_for_current_products():
    source = _normalized_sql(inspect.getsource(build_training_rows._load_events))

    assert "join products" in source
    assert "using (product_id)" in source


def test_training_split_uses_canonical_esci_split_before_hash_fallback():
    rows = pd.DataFrame(
        {
            "query_id": ["search-a", "search-b", "search-c", "search-d"],
            "query": ["Running   Shoes", "running shoes", "query b", "new long tail query"],
        }
    )

    got = build_training_rows.assign_splits(
        rows,
        esci_query_splits={"running shoes": "test", "query b": "train"},
    )

    assert got["split"].tolist() == [
        "test",
        "test",
        "val",
        build_training_rows.split_for("new long tail query"),
    ]


def test_brand_intent_lexical_labels_upgrade_only_unjudged_visible_matches():
    rows = pd.DataFrame(
        {
            "query": ["jordan", "jordan", "jordan", "jordan"],
            "product_id": ["brand", "title", "hidden", "judged_bad"],
            "label": [0, 0, 0, 0],
            "query_has_brand": [1.0, 1.0, 1.0, 1.0],
            "product_brand_match": [1.0, 0.0, 0.0, 1.0],
            "title_exact_query_match": [1.0, 1.0, 0.0, 1.0],
        }
    )

    got = build_training_rows.apply_lexical_relevance_labels(
        rows,
        judged_pairs={("jordan", "judged_bad")},
    )

    labels = dict(zip(got["product_id"], got["label"]))
    assert labels == {
        "brand": 3,
        "title": 2,
        "hidden": 0,
        "judged_bad": 0,
    }


def test_brand_family_labels_upgrade_subbrand_matches_without_broad_parent_match():
    rows = pd.DataFrame(
        {
            "query": ["jordan", "jordan", "jordan", "jordan"],
            "product_id": ["family", "generic_parent", "title", "judged_bad"],
            "label": [0, 0, 0, 0],
            "query_has_brand": [1.0, 1.0, 1.0, 1.0],
            "product_brand_match": [0.0, 0.0, 0.0, 0.0],
            "brand_family_match": [1.0, 0.0, 0.0, 1.0],
            "subbrand_title_match": [1.0, 0.0, 1.0, 1.0],
            "title_exact_query_match": [0.0, 0.0, 0.0, 0.0],
        }
    )

    got = build_training_rows.apply_lexical_relevance_labels(
        rows,
        judged_pairs={("jordan", "judged_bad")},
    )

    labels = dict(zip(got["product_id"], got["label"]))
    assert labels == {
        "family": 3,
        "generic_parent": 0,
        "title": 2,
        "judged_bad": 0,
    }


def test_category_intent_lexical_labels_upgrade_only_unjudged_category_matches():
    rows = pd.DataFrame(
        {
            "query": ["running shoes", "running shoes", "running shoes", "running shoes", "nike running shoes"],
            "product_id": ["full_category", "partial_category", "mismatch", "judged_bad", "branded_category"],
            "label": [0, 0, 0, 0, 0],
            "query_has_brand": [0.0, 0.0, 0.0, 0.0, 1.0],
            "product_brand_match": [0.0, 0.0, 0.0, 0.0, 0.0],
            "title_exact_query_match": [0.0, 0.0, 0.0, 0.0, 0.0],
            "query_has_category_token": [1.0, 1.0, 1.0, 1.0, 1.0],
            "category_query_token_coverage": [1.0, 0.5, 0.0, 1.0, 1.0],
        }
    )

    got = build_training_rows.apply_lexical_relevance_labels(
        rows,
        judged_pairs={("running shoes", "judged_bad")},
    )

    labels = dict(zip(got["product_id"], got["label"]))
    assert labels == {
        "full_category": 3,
        "partial_category": 2,
        "mismatch": 0,
        "judged_bad": 0,
        "branded_category": 0,
    }


def test_category_intent_labels_do_not_upgrade_accessories_for_core_product_queries():
    rows = pd.DataFrame(
        {
            "query": ["running shoes", "running shoes", "running shoes", "shoe laces"],
            "product_id": ["core_shoe", "shoe_accessory", "running_socks", "matching_accessory"],
            "title": [
                "Trail Running Shoes",
                "Running Shoe Replacement Laces",
                "Running Crew Socks",
                "No Tie Shoe Laces",
            ],
            "category_path": [
                ["Men", "Shoes", "Athletic", "Running"],
                ["Men", "Shoes", "Accessories", "Shoe Care"],
                ["Men", "Socks", "Running"],
                ["Shoes", "Accessories", "Laces"],
            ],
            "label": [0, 0, 0, 0],
            "query_has_brand": [0.0, 0.0, 0.0, 0.0],
            "query_has_category_token": [1.0, 1.0, 1.0, 1.0],
            "category_query_token_coverage": [1.0, 0.5, 0.5, 1.0],
        }
    )

    got = build_training_rows.apply_lexical_relevance_labels(rows, judged_pairs=set())

    labels = dict(zip(got["product_id"], got["label"]))
    assert labels == {
        "core_shoe": 3,
        "shoe_accessory": 0,
        "running_socks": 0,
        "matching_accessory": 3,
    }


def test_attribute_intent_labels_upgrade_unjudged_attribute_matches():
    rows = pd.DataFrame(
        {
            "query": [
                "womens jacket",
                "womens jacket",
                "red sneakers",
                "16 oz bottle",
                "kids backpack",
                "waterproof leather boots",
                "red leather boots",
                "red leather boots",
            ],
            "product_id": [
                "gender_match",
                "gender_mismatch",
                "color_match",
                "size_match",
                "age_match",
                "style_material_match",
                "multi_attribute_match",
                "judged_bad",
            ],
            "title": [
                "Women's Rain Jacket",
                "Men's Rain Jacket",
                "Red Running Sneakers",
                "Stainless Water Bottle 16 oz",
                "Kids School Backpack",
                "Waterproof Leather Hiking Boots",
                "Red Leather Chelsea Boots",
                "Red Leather Chelsea Boots",
            ],
            "category_path": [
                ["Women", "Clothing", "Jackets"],
                ["Men", "Clothing", "Jackets"],
                ["Women", "Shoes", "Sneakers"],
                ["Kitchen", "Water Bottles"],
                ["Kids", "School", "Backpacks"],
                ["Women", "Shoes", "Boots"],
                ["Women", "Shoes", "Boots"],
                ["Women", "Shoes", "Boots"],
            ],
            "label": [0, 0, 0, 0, 0, 0, 0, 0],
            "query_has_brand": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "query_gender_intent": [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "product_gender": [2.0, 1.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            "gender_intent_match": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gender_intent_mismatch": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "query_has_color": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "product_color_match": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "query_has_size_pattern": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    got = build_training_rows.apply_lexical_relevance_labels(
        rows,
        judged_pairs={("red leather boots", "judged_bad")},
    )

    labels = dict(zip(got["product_id"], got["label"]))
    assert labels == {
        "gender_match": 2,
        "gender_mismatch": 0,
        "color_match": 2,
        "size_match": 2,
        "age_match": 2,
        "style_material_match": 3,
        "multi_attribute_match": 3,
        "judged_bad": 0,
    }
