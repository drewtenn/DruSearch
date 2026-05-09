from __future__ import annotations

import inspect
import json

from psycopg.errors import IndeterminateDatatype

from pipelines.ingest import amazon_reviews
from pipelines.ingest.amazon_reviews import _category_path, _iter_metadata_records, _judgment_row, _product_row


def test_product_row_normalizes_amazon_metadata():
    raw = {
        "parent_asin": "B000123",
        "title": "Trail Running Shoe",
        "description": ["Lightweight trail shoe.", "Second paragraph."],
        "features": ["Durable outsole", "Breathable upper"],
        "store": "Acme",
        "main_category": "Clothing, Shoes & Jewelry",
        "categories": [["Men", "Shoes", "Athletic", "Running"]],
        "price": "79.99",
        "average_rating": 4.5,
        "rating_number": 120,
    }

    row = _product_row(raw)
    assert row == (
        "B000123",
        "us",
        "Trail Running Shoe",
        "Lightweight trail shoe. Second paragraph.",
        "Durable outsole; Breathable upper",
        "Acme",
        "",
        7999,
        "Running",
        4.5,
        ["Men", "Shoes", "Athletic", "Running"],
        raw,
    )
    assert row[-1]["rating_number"] == 120


def test_product_row_uses_main_category_when_hierarchy_is_empty():
    raw = {
        "parent_asin": "B000456",
        "title": "Leather Conditioner",
        "description": [],
        "features": [],
        "store": "Howard Products",
        "main_category": "All Beauty",
        "categories": [],
        "price": "None",
        "average_rating": 4.8,
    }

    assert _product_row(raw) == (
        "B000456",
        "us",
        "Leather Conditioner",
        None,
        None,
        "Howard Products",
        "",
        None,
        "All Beauty",
        4.8,
        ["All Beauty"],
        raw,
    )


def test_iter_metadata_records_stops_at_target_and_skips_missing_titles(tmp_path):
    path = tmp_path / "meta.jsonl"
    rows = [
        {"parent_asin": "skip", "title": ""},
        {"parent_asin": "keep-1", "title": "First"},
        {"parent_asin": "keep-2", "title": "Second"},
        {"parent_asin": "keep-3", "title": "Third"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    records = list(_iter_metadata_records(path, target=2))

    assert [record["parent_asin"] for record in records] == ["keep-1", "keep-2"]


def test_judgment_row_builds_query_from_brand_and_category():
    product = (
        "B000123",
        "us",
        "Trail Running Shoe",
        None,
        None,
        "Acme",
        "",
        7999,
        "Athletic Shoes",
        4.5,
        ["Athletic Shoes"],
        {},
    )

    assert _judgment_row(17, product) == (17, "acme athletic shoes", "B000123", "E", "train")


def test_judgment_row_falls_back_to_title_terms():
    product = (
        "B000456",
        "us",
        "Leather Conditioner and Cleaner",
        None,
        None,
        "",
        "",
        None,
        "",
        4.8,
        [],
        {},
    )

    assert _judgment_row(22, product) == (22, "leather conditioner and cleaner", "B000456", "E", "test")


def test_category_path_preserves_full_amazon_hierarchy():
    raw = {"categories": [["Men", "Shoes", "Athletic", "Running"]], "main_category": "Clothing"}

    assert _category_path(raw) == ["Men", "Shoes", "Athletic", "Running"]


def test_category_path_preserves_flat_amazon_hierarchy():
    raw = {"categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Fashion Sneakers"]}

    assert _category_path(raw) == ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Fashion Sneakers"]


def test_ingest_run_metadata_sql_casts_string_values():
    assert "%s::text" in amazon_reviews.UPDATE_RUN_OK_SQL


def test_amazon_reviews_ingest_clears_derived_state_when_replacing_catalog():
    source = " ".join(inspect.getsource(amazon_reviews.main).lower().split())

    assert "truncate search_events" in source
    assert "training_rows" in source
    assert source.index("truncate search_events") < source.index("truncate products")


def test_main_rolls_back_before_marking_ingest_failed(monkeypatch, tmp_path):
    source = tmp_path / "meta.jsonl"
    source.write_text(json.dumps({"parent_asin": "B000123", "title": "Trail Shoe"}) + "\n")
    monkeypatch.setattr(amazon_reviews, "SOURCE_FILE", str(source))
    monkeypatch.setattr(amazon_reviews, "TARGET_PRODUCTS", 1)

    class Copy:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write_row(self, row):
            pass

    class Cursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=None):
            self.conn.events.append(("execute", sql))
            if "INSERT INTO ingest_runs" in sql:
                return
            if "status='ok'" in sql:
                raise IndeterminateDatatype("could not determine data type of parameter $2")

        def fetchone(self):
            return (42,)

        def copy(self, sql):
            self.conn.events.append(("copy", sql))
            return Copy()

    class Conn:
        def __init__(self):
            self.events = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return Cursor(self)

        def commit(self):
            self.events.append(("commit", None))

        def rollback(self):
            self.events.append(("rollback", None))

    conn = Conn()
    monkeypatch.setattr(amazon_reviews.db, "conn", lambda: conn)

    try:
        amazon_reviews.main()
    except IndeterminateDatatype:
        pass

    failed_idx = next(i for i, event in enumerate(conn.events) if event[0] == "execute" and "status='failed'" in event[1])
    assert ("rollback", None) in conn.events[:failed_idx]
