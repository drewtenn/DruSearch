from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.ingest.esci import _filtered_products_from_parquet


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
