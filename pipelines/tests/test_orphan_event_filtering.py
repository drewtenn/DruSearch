from __future__ import annotations

import inspect

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
