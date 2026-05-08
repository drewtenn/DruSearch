"""Index Postgres products into OpenSearch (BM25 only; vectors come in Phase 2).

Run:  docker compose run --rm pipelines python -m pipelines.index.bm25
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from opensearchpy.helpers import bulk

from pipelines.common import db, opensearch_client
from pipelines.common.logging import configure

log = configure("index.bm25")

INDEX_TEMPLATE = "products_template"
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "products_v1")
TEMPLATE_PATH = Path("/app/infra/opensearch/index_template.json")
BATCH_SIZE = int(os.getenv("INDEX_BATCH", "500"))


def _ensure_index_template(client) -> None:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"missing index template: {TEMPLATE_PATH}")
    template = json.loads(TEMPLATE_PATH.read_text())
    client.indices.put_index_template(name=INDEX_TEMPLATE, body=template)
    log.info("applied index template %s", INDEX_TEMPLATE)


def _ensure_index(client) -> None:
    if client.indices.exists(index=INDEX_NAME):
        log.info("deleting existing index %s", INDEX_NAME)
        client.indices.delete(index=INDEX_NAME)
    client.indices.create(index=INDEX_NAME)
    log.info("created index %s", INDEX_NAME)


def _iter_docs() -> Iterable[dict]:
    with db.conn() as c, c.cursor(name="products_cursor") as cur:
        cur.execute(
            "SELECT product_id, title, COALESCE(description, ''), COALESCE(bullet_points, ''),"
            " COALESCE(brand, ''), COALESCE(color, ''), COALESCE(category, ''),"
            " COALESCE(price_cents, 0), COALESCE(popularity_prior, 0)"
            " FROM products"
        )
        for row in cur:
            pid, title, desc, bullets, brand, color, cat, price, pop = row
            yield {
                "_index": INDEX_NAME,
                "_id": pid,
                "_source": {
                    "product_id":       pid,
                    "title":            title,
                    "description":      desc,
                    "bullets":          bullets,
                    "brand":            brand or None,
                    "color":            color or None,
                    "category":         cat or None,
                    "price_cents":      int(price),
                    "popularity_prior": float(pop),
                    "ctr_prior":        0.0,
                },
            }


def main() -> int:
    client = opensearch_client.client()
    _ensure_index_template(client)
    _ensure_index(client)

    success, errors = bulk(
        client,
        _iter_docs(),
        chunk_size=BATCH_SIZE,
        request_timeout=120,
        raise_on_error=True,
    )
    log.info("bulk indexed success=%d errors=%s", success, errors)

    client.indices.refresh(index=INDEX_NAME)
    count = client.count(index=INDEX_NAME)["count"]
    log.info("index %s now has %d docs", INDEX_NAME, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
