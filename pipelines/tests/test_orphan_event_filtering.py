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
