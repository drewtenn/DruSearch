from __future__ import annotations

import inspect

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.ingest import esci
from pipelines.ingest.esci import (
    _category_path_from_product,
    _filtered_products_from_parquet,
    _iter_rows,
    _select_products,
)


def test_filtered_products_from_parquet_filters_in_batches(tmp_path):
    path = tmp_path / "products.parquet"
    table = pa.table(
        {
            "product_id": ["keep-us", "skip-locale", "skip-id", "keep-us-2"],
            "product_locale": ["us", "es", "us", "us"],
            "product_title": ["Keep One", "Wrong Locale", "Wrong ID", "Keep Two"],
            "product_description": ["a", "b", "c", "d"],
            "product_bullet_point": ["aa", "bb", "cc", "dd"],
            "product_brand": ["BrandA", "BrandB", "BrandC", "BrandD"],
            "product_color": ["red", "blue", "green", "black"],
        }
    )
    pq.write_table(table, path)

    products = _filtered_products_from_parquet(
        path,
        selected={"keep-us", "keep-us-2", "skip-locale"},
        batch_size=2,
    )

    assert products["product_id"].tolist() == ["keep-us", "keep-us-2"]
    assert products["product_title"].tolist() == ["Keep One", "Keep Two"]


def test_category_path_from_product_derives_ecommerce_taxonomy():
    row = pd.Series(
        {
            "product_title": "Nike Men's Trail Running Shoes",
            "product_description": "Breathable athletic sneaker for long runs",
            "product_bullet_point": "Rubber outsole",
        }
    )

    assert _category_path_from_product(row) == [
        "Clothing, Shoes & Jewelry",
        "Men",
        "Shoes",
        "Athletic",
        "Running",
    ]


def test_iter_rows_populates_category_fields_for_esci_products():
    products = pd.DataFrame(
        {
            "product_id": ["p1"],
            "product_title": ["Women's Hooded Sweatshirt"],
            "product_description": ["Fleece hoodie"],
            "product_bullet_point": ["Pullover"],
            "product_brand": ["BrandA"],
            "product_color": ["black"],
        }
    )

    row = next(_iter_rows(products))

    assert row[8] == "Hoodies & Sweatshirts"
    assert row[10] == [
        "Clothing, Shoes & Jewelry",
        "Women",
        "Clothing",
        "Tops",
        "Hoodies & Sweatshirts",
    ]


def test_select_products_keeps_query_groups_complete():
    examples = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 3],
            "product_id": ["p1", "p2", "p1", "p3"],
        }
    )

    selected = _select_products(examples, target=2, seed=0)
    subset = examples[examples["query_id"].isin(selected.query_ids)]

    assert selected.product_ids == {"p1", "p2"}
    assert selected.query_ids == {1}
    assert subset["query_id"].tolist() == [1, 1]


def test_esci_ingest_clears_derived_state_when_replacing_catalog():
    source = " ".join(inspect.getsource(esci.main).lower().split())

    assert "truncate search_events" in source
    assert "training_rows" in source
    assert source.index("truncate search_events") < source.index("truncate products")
