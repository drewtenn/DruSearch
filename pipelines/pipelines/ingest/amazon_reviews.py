"""Ingest Amazon Reviews 2023 item metadata into Postgres.

Run:  docker compose run --rm pipelines python -m pipelines.ingest.amazon_reviews
"""

from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx
from psycopg.types.json import Jsonb

from pipelines.common import db
from pipelines.common.logging import configure

log = configure("amazon.ingest")

DEFAULT_CATEGORY = "Clothing_Shoes_and_Jewelry"
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories"

TARGET_PRODUCTS = int(os.getenv("AMAZON_REVIEWS_TARGET_PRODUCTS", "10000"))
CATEGORY = os.getenv("AMAZON_REVIEWS_CATEGORY", DEFAULT_CATEGORY)
SOURCE_URL = os.getenv(
    "AMAZON_REVIEWS_META_URL",
    f"{HF_BASE}/meta_{CATEGORY}.jsonl",
)
SOURCE_FILE = os.getenv("AMAZON_REVIEWS_META_FILE", "")

UPDATE_RUN_OK_SQL = (
    "UPDATE ingest_runs SET finished_at=now(), status='ok',"
    " metadata=jsonb_build_object('products', %s, 'judgments', %s, 'category', %s::text, 'source', %s::text)"
    " WHERE run_id=%s"
)


def _compact_text(value: Any) -> str | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, list):
        parts = [_compact_text(item) for item in value]
        text = " ".join(part for part in parts if part)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _feature_text(value: Any) -> str | None:
    if not isinstance(value, list):
        return _compact_text(value)
    parts = [_compact_text(item) for item in value]
    text = "; ".join(part for part in parts if part)
    return text or None


def _price_cents(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
    try:
        cents = round(float(value) * 100)
    except (TypeError, ValueError):
        return None
    return cents if cents >= 0 else None


def _category(value: dict[str, Any]) -> str:
    path = _category_path(value)
    if path:
        return path[-1]
    return _compact_text(value.get("main_category")) or ""


def _category_path(value: dict[str, Any]) -> list[str]:
    categories = value.get("categories")
    if isinstance(categories, list) and categories:
        raw_path = categories[-1] if isinstance(categories[-1], list) else categories
        if not isinstance(raw_path, list):
            raw_path = [raw_path]
        path = [_compact_text(item) for item in raw_path]
        return [item for item in path if item]
    fallback = _compact_text(value.get("main_category"))
    return [fallback] if fallback else []


def _details(value: dict[str, Any]) -> dict[str, Any]:
    raw = value.get("details")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _color(value: dict[str, Any]) -> str:
    details = _details(value)
    for key in ("Color", "Colour"):
        text = _compact_text(details.get(key))
        if text:
            return text
    return ""


def _product_row(value: dict[str, Any]) -> tuple:
    product_id = _compact_text(value.get("parent_asin") or value.get("asin"))
    title = _compact_text(value.get("title"))
    if not product_id or not title:
        raise ValueError("Amazon metadata row is missing parent_asin/asin or title")
    popularity = value.get("average_rating")
    try:
        popularity_prior = float(popularity) if popularity is not None else 0.0
    except (TypeError, ValueError):
        popularity_prior = 0.0
    return (
        product_id,
        "us",
        title,
        _compact_text(value.get("description")),
        _feature_text(value.get("features")),
        _compact_text(value.get("store")),
        _color(value),
        _price_cents(value.get("price")),
        _category(value),
        popularity_prior,
        _category_path(value),
        value,
    )


def _query_text(parts: Iterable[str | None]) -> str:
    text = _compact_text(" ".join(part for part in parts if part))
    return (text or "").lower()


def _judgment_row(query_id: int, product: tuple) -> tuple[int, str, str, str, str]:
    product_id = product[0]
    title = product[2]
    brand = product[5]
    category = product[8]
    category_path = product[10]
    if not isinstance(category_path, list):
        category_path = []
    query = _query_text((brand, *category_path)) or _query_text((brand, category)) or _query_text((title,))
    split = "test" if query_id % 10 == 2 else "train"
    return (query_id, query, product_id, "E", split)


def _local_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield from f


def _remote_lines(url: str) -> Iterator[str]:
    with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        yield from resp.iter_lines()


def _iter_metadata_records(source: str | Path, target: int) -> Iterable[dict[str, Any]]:
    emitted = 0
    lines = _remote_lines(str(source)) if str(source).startswith(("http://", "https://")) else _local_lines(Path(source))
    for line in lines:
        if not line:
            continue
        record = json.loads(line)
        if not _compact_text(record.get("title")):
            continue
        yield record
        emitted += 1
        if emitted >= target:
            break


def main() -> int:
    source = SOURCE_FILE or SOURCE_URL
    log.info("starting Amazon Reviews 2023 ingest category=%s target=%d source=%s", CATEGORY, TARGET_PRODUCTS, source)

    records = _iter_metadata_records(source, TARGET_PRODUCTS)
    rows = []
    for record in records:
        try:
            rows.append(_product_row(record))
        except ValueError:
            continue

    if not rows:
        log.error("no products to ingest; aborting")
        return 1
    judgments = [_judgment_row(i + 1, row) for i, row in enumerate(rows)]

    with db.conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_runs(pipeline, dataset_hash, status) VALUES (%s, %s, 'running') RETURNING run_id",
                ("ingest.amazon_reviews", f"amazon-reviews-2023-{CATEGORY}-{len(rows)}"),
            )
            run_id = cur.fetchone()[0]
        c.commit()

        try:
            with c.cursor() as cur:
                cur.execute("TRUNCATE products RESTART IDENTITY CASCADE")
                cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS category_path TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[]")
                cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS raw_metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
                with cur.copy(
                    "COPY products (product_id, locale, title, description, bullet_points, brand, color, price_cents, category, popularity_prior, category_path, raw_metadata) FROM STDIN"
                ) as copy:
                    for row in rows:
                        copy.write_row((*row[:-1], Jsonb(row[-1])))
                log.info("copied %d products", len(rows))

                with cur.copy(
                    "COPY esci_judgments (query_id, query, product_id, esci_label, split) FROM STDIN"
                ) as copy:
                    for row in judgments:
                        copy.write_row(row)
                log.info("copied %d synthetic judgments", len(judgments))

                cur.execute(UPDATE_RUN_OK_SQL, (len(rows), len(judgments), CATEGORY, source, run_id))
            c.commit()
        except Exception:
            c.rollback()
            with c.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_runs SET finished_at=now(), status='failed' WHERE run_id=%s",
                    (run_id,),
                )
            c.commit()
            raise

    log.info("Amazon Reviews 2023 ingest complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
