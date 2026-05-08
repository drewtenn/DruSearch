"""Compute product embeddings and write them to OpenSearch as `title_vec`.

Source text:  f"{title}. {first_bullet}"  (truncated to ~256 tokens by the model).
Model:        sentence-transformers loaded by the embedder sidecar.
Strategy:     stream from Postgres in batches; POST to embedder /embed_batch;
              bulk update OpenSearch documents.

Run:  docker compose run --rm pipelines python -m pipelines.embed.build_vectors
"""

from __future__ import annotations

import os
from typing import Iterable

import httpx
from opensearchpy.helpers import bulk
from tenacity import retry, stop_after_attempt, wait_exponential

from pipelines.common import db, opensearch_client
from pipelines.common.config import load
from pipelines.common.logging import configure

log = configure("embed.build_vectors")

INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "products_v1")
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "64"))
MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "1024"))


def _embed_text(title: str, bullets: str) -> str:
    first_bullet = (bullets or "").split("\n", 1)[0].strip()
    text = f"{title}. {first_bullet}".strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
def _embed_batch(client: httpx.Client, base_url: str, texts: list[str]) -> list[list[float]]:
    r = client.post(
        f"{base_url}/embed_batch",
        json={"texts": texts},
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()["vectors"]


def _iter_products(batch: int) -> Iterable[list[tuple]]:
    with db.conn() as c, c.cursor(name="products_for_embed") as cur:
        cur.itersize = batch
        cur.execute(
            "SELECT product_id, title, COALESCE(bullet_points, '')"
            " FROM products"
            " ORDER BY product_id"
        )
        rows: list[tuple] = []
        for row in cur:
            rows.append(row)
            if len(rows) >= batch:
                yield rows
                rows = []
        if rows:
            yield rows


def main() -> int:
    cfg = load()
    os_client = opensearch_client.client()
    info = os_client.info()
    log.info("opensearch=%s embedder=%s model=%s",
             info.get("version", {}).get("number"), cfg.embedder_url, cfg.embedder_model)

    total_updated = 0
    with httpx.Client() as http:
        for batch in _iter_products(BATCH_SIZE):
            ids = [r[0] for r in batch]
            texts = [_embed_text(r[1], r[2]) for r in batch]
            vectors = _embed_batch(http, cfg.embedder_url, texts)
            if vectors and len(vectors[0]) != cfg.embedder_dim:
                raise RuntimeError(
                    f"embedding dim mismatch: got {len(vectors[0])}, "
                    f"expected {cfg.embedder_dim}; check EMBEDDER_DIM and index mapping"
                )

            actions = (
                {
                    "_op_type": "update",
                    "_index":   INDEX_NAME,
                    "_id":      pid,
                    "doc":      {"title_vec": vec},
                }
                for pid, vec in zip(ids, vectors)
            )
            ok, errors = bulk(os_client, actions, request_timeout=60, raise_on_error=True)
            total_updated += ok
            if errors:
                log.warning("bulk-update errors=%s", errors)
            log.info("embedded batch=%d total=%d", len(batch), total_updated)

    os_client.indices.refresh(index=INDEX_NAME)
    count = os_client.count(
        index=INDEX_NAME,
        body={"query": {"exists": {"field": "title_vec"}}},
    )["count"]
    log.info("docs with title_vec=%d total_updated=%d", count, total_updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
