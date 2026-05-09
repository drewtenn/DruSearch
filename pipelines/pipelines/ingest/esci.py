"""Ingest the Amazon ESCI shopping-queries dataset into Postgres.

Strategy:
  1. Cache the raw parquets to MinIO so subsequent runs are offline.
  2. Filter examples to small_version=1, US locale, with E/S/C/I labels.
  3. Walk queries in shuffled order, accumulating products until ~10k unique.
  4. Filter products.parquet to that set; bulk-COPY into Postgres products.
  5. Insert filtered examples into esci_judgments with their split.

Run:  docker compose run --rm pipelines python -m pipelines.ingest.esci
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd
import pyarrow.parquet as pq

from pipelines.common import db
from pipelines.common.config import load
from pipelines.common.logging import configure
from pipelines.common.storage import s3_client

log = configure("esci.ingest")

ESCI_BASE = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset"
EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"

TARGET_PRODUCTS = int(os.getenv("ESCI_TARGET_PRODUCTS", "50000"))
SEED = int(os.getenv("ESCI_SEED", "42"))
PRODUCT_BATCH_ROWS = int(os.getenv("ESCI_PRODUCT_BATCH_ROWS", "50000"))
PRODUCT_COLUMNS = [
    "product_id",
    "product_locale",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]


@dataclass(frozen=True)
class Selection:
    query_ids: set[int]
    product_ids: set[str]


def _missing_object(exc: Exception) -> bool:
    text = str(exc)
    return "NoSuchKey" in text or "404" in text or "Not Found" in text


def _cached_path(name: str) -> Path:
    cfg = load()
    s3 = s3_client()
    key = f"esci/{name}"
    cache_dir = Path(os.getenv("ESCI_CACHE_DIR", "/tmp/drusearch-esci"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    if path.exists() and path.stat().st_size > 0:
        log.info("local cache hit %s", path)
        return path

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        log.info("cache hit s3://%s/%s", cfg.minio_bucket_data, key)
        s3.download_file(cfg.minio_bucket_data, key, str(tmp))
        tmp.replace(path)
        return path
    except Exception as exc:  # pragma: no cover - first-run path
        tmp.unlink(missing_ok=True)
        if not _missing_object(exc):
            raise

    url = f"{ESCI_BASE}/{name}"
    log.info("downloading %s", url)
    started = time.perf_counter()
    with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as r:
        r.raise_for_status()
        total = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                total += len(chunk)
        log.info("downloaded %s bytes=%d in %.1fs", name, total, time.perf_counter() - started)
    tmp.replace(path)

    log.info("uploading to s3://%s/%s", cfg.minio_bucket_data, key)
    s3.upload_file(str(path), cfg.minio_bucket_data, key)
    return path


def _select_products(examples: pd.DataFrame, target: int, seed: int) -> Selection:
    """Walk shuffled queries and keep complete query groups until the product target."""
    rng = random.Random(seed)
    by_query = (
        examples.groupby("query_id")["product_id"].apply(lambda s: list(set(s))).to_dict()
    )
    qids = list(by_query.keys())
    rng.shuffle(qids)
    chosen_queries: set[int] = set()
    chosen_products: set[str] = set()
    for q in qids:
        chosen_queries.add(int(q))
        chosen_products.update(by_query[q])
        if len(chosen_products) >= target:
            break
    return Selection(query_ids=chosen_queries, product_ids=chosen_products)


def _iter_rows(df: pd.DataFrame) -> Iterable[tuple]:
    """Yield rows for COPY in the order matching the COPY column list."""
    for r in df.itertuples(index=False):
        yield (
            r.product_id,
            "us",
            getattr(r, "product_title", None) or "",
            getattr(r, "product_description", None) or None,
            getattr(r, "product_bullet_point", None) or None,
            getattr(r, "product_brand", None) or None,
            getattr(r, "product_color", None) or None,
        )


def _filtered_products_from_parquet(
    path: str | Path,
    selected: set[str],
    batch_size: int = PRODUCT_BATCH_ROWS,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=PRODUCT_COLUMNS):
        chunk = batch.to_pandas()
        chunk = chunk[
            (chunk["product_locale"] == "us") & chunk["product_id"].isin(selected)
        ]
        if len(chunk) > 0:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=PRODUCT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    log.info("starting ESCI ingest target=%d seed=%d", TARGET_PRODUCTS, SEED)

    examples = pq.read_table(_cached_path(EXAMPLES_FILE)).to_pandas()
    log.info("examples loaded rows=%d", len(examples))

    mask = (
        (examples["small_version"] == 1)
        & (examples["product_locale"] == "us")
        & examples["esci_label"].notna()
        & examples["split"].isin(["train", "test"])
    )
    examples = examples.loc[mask].reset_index(drop=True)
    log.info(
        "examples filtered rows=%d uniq_products=%d uniq_queries=%d",
        len(examples),
        examples["product_id"].nunique(),
        examples["query_id"].nunique(),
    )

    selected = _select_products(examples, TARGET_PRODUCTS, SEED)
    examples = examples[examples["query_id"].isin(selected.query_ids)].reset_index(drop=True)
    log.info(
        "after subset rows=%d products=%d queries=%d",
        len(examples),
        examples["product_id"].nunique(),
        examples["query_id"].nunique(),
    )

    products = _filtered_products_from_parquet(_cached_path(PRODUCTS_FILE), selected.product_ids)
    log.info("products filtered rows=%d", len(products))

    if len(products) == 0:
        log.error("no products to ingest after filtering; aborting")
        return 1

    with db.conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_runs(pipeline, dataset_hash, status) VALUES (%s, %s, 'running') RETURNING run_id",
                ("ingest.esci", f"esci-small-us-{SEED}-{TARGET_PRODUCTS}"),
            )
            run_id = cur.fetchone()[0]
        c.commit()

        try:
            with c.cursor() as cur:
                cur.execute("TRUNCATE search_events, training_rows, user_sessions RESTART IDENTITY")
                cur.execute("TRUNCATE products RESTART IDENTITY CASCADE")
                with cur.copy(
                    "COPY products (product_id, locale, title, description, bullet_points, brand, color) FROM STDIN"
                ) as copy:
                    for row in _iter_rows(products):
                        copy.write_row(row)
                log.info("copied %d products", len(products))

                with cur.copy(
                    "COPY esci_judgments (query_id, query, product_id, esci_label, split) FROM STDIN"
                ) as copy:
                    for r in examples.itertuples(index=False):
                        copy.write_row((int(r.query_id), r.query, r.product_id, r.esci_label, r.split))
                log.info("copied %d judgments", len(examples))

                cur.execute(
                    "UPDATE ingest_runs SET finished_at=now(), status='ok',"
                    " metadata=jsonb_build_object('products', %s, 'judgments', %s)"
                    " WHERE run_id=%s",
                    (len(products), len(examples), run_id),
                )
            c.commit()
        except Exception:
            with c.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_runs SET finished_at=now(), status='failed' WHERE run_id=%s",
                    (run_id,),
                )
            c.commit()
            raise

    log.info("ESCI ingest complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
