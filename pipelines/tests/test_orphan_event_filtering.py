from __future__ import annotations

from contextlib import contextmanager
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


class _WriteRowsCursor:
    description = []

    def __init__(self):
        self.statements: list[tuple[str, object | None]] = []
        self.executemany_sql = ""
        self.executemany_records = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))

    def executemany(self, sql, records):
        self.executemany_sql = " ".join(str(sql).split())
        self.executemany_records = list(records)


class _WriteRowsConnection:
    def __init__(self):
        self.cursor_obj = _WriteRowsCursor()
        self.commits = 0

    @contextmanager
    def cursor(self):
        yield self.cursor_obj

    def commit(self):
        self.commits += 1


def test_write_training_rows_persists_build_id_and_marks_generation_ready(monkeypatch):
    conn = _WriteRowsConnection()

    @contextmanager
    def fake_conn():
        yield conn

    monkeypatch.setattr(build_training_rows.db, "conn", fake_conn)
    rows = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "product_id": ["p1", "p2"],
            "query": ["running shoes", "running shoes"],
            "user_id": [None, None],
            "ts": [pd.Timestamp("2026-05-09T00:00:00Z")] * 2,
            "features": [{"bm25_score": 1.0}, {"bm25_score": 0.5}],
            "label": [4.0, 0.0],
            "split": ["train", "train"],
            "sample_weight": [1.0, 1.0],
        }
    )

    build_training_rows.write_training_rows(rows, build_id=42)

    assert conn.commits == 1
    assert "build_id" in conn.cursor_obj.executemany_sql
    assert [record[-1] for record in conn.cursor_obj.executemany_records] == [42, 42]
    sql = "\n".join(statement for statement, _params in conn.cursor_obj.statements).lower()
    assert "update training_row_builds" in sql
    assert "status = 'ready'" in sql
    ready_update = [
        params
        for statement, params in conn.cursor_obj.statements
        if "UPDATE training_row_builds" in statement
    ][0]
    metadata = getattr(ready_update[2], "obj", ready_update[2])
    assert ready_update[0:2] == (2, 1)
    assert metadata == {"split_counts": {"train": 2}}
    assert ready_update[3] == 42


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


def test_retrieval_rank_features_encode_absent_side_as_worse_than_seen_rank():
    hits = [
        {"product_id": "both", "bm25_rank": 1, "knn_rank": 2},
        {"product_id": "bm25_only", "bm25_rank": 2, "knn_rank": 0},
        {"product_id": "knn_only", "bm25_rank": 0, "knn_rank": 1},
    ]

    got = build_training_rows.normalize_retrieval_ranks(hits)

    assert got[0]["bm25_rank"] == 1
    assert got[0]["knn_rank"] == 2
    assert got[1]["knn_rank"] == 3
    assert got[2]["bm25_rank"] == 3


def test_candidate_training_rows_use_full_retrieval_pool_not_logged_impressions():
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "title": ["Running Shoe", "Trail Sneaker", "Leather Boot"],
            "brand": ["Nike", "Altra", "Timberland"],
            "color": ["red", "blue", "brown"],
            "price_cents": [1000, 2000, 3000],
            "popularity_prior": [0.1, 0.2, 0.3],
            "category_path": [["Shoes", "Running"], ["Shoes", "Trail"], ["Shoes", "Boots"]],
        }
    )
    queries = pd.DataFrame(
        {
            "query_id": [101],
            "query": ["running shoes"],
        }
    )

    class FakeRetriever:
        def search(self, query, n):
            assert query == "running shoes"
            assert n == 200
            return [
                {"product_id": "p1", "bm25": 9.0, "bm25_rank": 1, "knn": 0.8, "knn_rank": 1, "rrf": 0.03},
                {"product_id": "p2", "bm25": 6.0, "bm25_rank": 2, "knn": 0.0, "knn_rank": 0, "rrf": 0.016},
                {"product_id": "p3", "bm25": 0.0, "bm25_rank": 0, "knn": 0.7, "knn_rank": 2, "rrf": 0.015},
            ]

    got = build_training_rows.build_candidate_training_frame(
        queries=queries,
        products=products,
        retriever=FakeRetriever(),
        cand_n=200,
    )

    assert got["query_id"].tolist() == ["101", "101", "101"]
    assert got["product_id"].tolist() == ["p1", "p2", "p3"]
    assert got.loc[got["product_id"] == "p2", "knn_rank"].item() == 3
    assert got.loc[got["product_id"] == "p3", "bm25_rank"].item() == 3


def test_candidate_training_rows_dedupe_duplicate_retrieval_hits_per_query():
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "title": ["Running Shoe", "Trail Sneaker"],
            "brand": ["Nike", "Altra"],
            "color": ["red", "blue"],
            "price_cents": [1000, 2000],
            "popularity_prior": [0.1, 0.2],
            "category_path": [["Shoes", "Running"], ["Shoes", "Trail"]],
        }
    )
    queries = pd.DataFrame({"query_id": [101], "query": ["running shoes"]})

    class FakeRetriever:
        def search(self, query, n):
            return [
                {"product_id": "p1", "bm25": 9.0, "bm25_rank": 1, "knn": 0.8, "knn_rank": 1, "rrf": 0.03},
                {"product_id": "p1", "bm25": 8.0, "bm25_rank": 2, "knn": 0.7, "knn_rank": 2, "rrf": 0.02},
                {"product_id": "p2", "bm25": 6.0, "bm25_rank": 3, "knn": 0.0, "knn_rank": 0, "rrf": 0.016},
            ]

    got = build_training_rows.build_candidate_training_frame(
        queries=queries,
        products=products,
        retriever=FakeRetriever(),
        cand_n=200,
    )

    assert got["product_id"].tolist() == ["p1", "p2"]
    assert got.duplicated(["query_id", "product_id"]).sum() == 0


def test_pseudo_labels_apply_only_to_train_rows_with_lower_weight():
    rows = pd.DataFrame(
        {
            "query": ["jordan", "jordan", "jordan"],
            "product_id": ["train_match", "val_match", "test_match"],
            "split": ["train", "val", "test"],
            "label": [0, 0, 0],
            "query_has_brand": [1.0, 1.0, 1.0],
            "product_brand_match": [1.0, 1.0, 1.0],
            "title_exact_query_match": [0.0, 0.0, 0.0],
        }
    )

    got = build_training_rows.apply_pseudo_labels_for_training(
        rows,
        judged_pairs=set(),
        enabled=True,
        pseudo_weight=0.25,
    )

    assert got["label"].tolist() == [3, 0, 0]
    assert got["sample_weight"].tolist() == [0.25, 1.0, 1.0]


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
