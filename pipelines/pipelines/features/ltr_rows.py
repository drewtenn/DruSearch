"""Shared LTR candidate feature construction.

Training-row generation and offline evaluation both rerank retrieval
candidates. Keep the candidate rank encoding and interaction feature
construction here so those paths do not drift.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from pipelines.features import FEATURE_NAMES
from pipelines.features import transforms as tf


def normalize_retrieval_ranks(hits: Iterable[dict]) -> list[dict]:
    """Return hits with absent BM25/kNN ranks encoded worse than any seen rank.

    Retrieval uses rank 0 to mean "not present in that retrieval side". Tree
    models can learn rank 0 as better than rank 1, so LTR features encode
    absence as max_observed_rank + 1 within the candidate set.
    """
    out: list[dict] = []
    seen_product_ids: set[str] = set()
    for hit in hits:
        pid = str(hit.get("product_id") or "")
        if not pid or pid in seen_product_ids:
            continue
        seen_product_ids.add(pid)
        out.append(dict(hit))
    observed = [
        int(h.get(name, 0) or 0)
        for h in out
        for name in ("bm25_rank", "knn_rank")
        if int(h.get(name, 0) or 0) > 0
    ]
    missing_rank = (max(observed) + 1) if observed else (len(out) + 1)
    for h in out:
        if int(h.get("bm25_rank", 0) or 0) <= 0:
            h["bm25_rank"] = missing_rank
        if int(h.get("knn_rank", 0) or 0) <= 0:
            h["knn_rank"] = missing_rank
    return out


def token_sets(products: pd.DataFrame) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    brand_tokens = frozenset(
        t
        for b in products["brand"].dropna().unique()
        if isinstance(b, str) and b
        for t in tf.brand_tokens(b)
    )
    color_tokens = frozenset(
        t
        for c in products["color"].dropna().unique()
        if isinstance(c, str)
        for t in tf.tokenize(c)
    )
    category_tokens = frozenset(
        t
        for cp in products["category_path"].dropna()
        for part in (cp or [])
        for t in tf.tokenize(part)
    )
    return brand_tokens, color_tokens, category_tokens


def build_feature_frame(
    query: str,
    hits: list[dict],
    products: pd.DataFrame,
    brand_tokens: frozenset[str],
    color_tokens: frozenset[str],
    category_tokens: frozenset[str],
    user_brand_affinity: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build one row per candidate with product fields and LTR feature columns."""
    if products.index.name != "product_id":
        products = products.set_index("product_id", drop=False)

    normalized_hits = normalize_retrieval_ranks(hits)
    qlen = tf.query_length_tokens(query)
    qbrand = tf.query_has_brand(query, brand_tokens)
    qcolor = tf.query_has_color(query, color_tokens)
    qcategory = tf.query_has_category_token(query, category_tokens)
    qsize = tf.query_has_size_pattern(query)
    qgender = tf.query_gender_intent(query)
    qaffordability = tf.query_affordability_intent(query)
    user_brand_affinity = user_brand_affinity or {}

    rows: list[dict] = []
    for h in normalized_hits:
        pid = h["product_id"]
        prod = products.loc[pid] if pid in products.index else None
        title = "" if prod is None else (prod["title"] or "")
        brand = "" if prod is None else (prod["brand"] or "")
        color = "" if prod is None else (prod["color"] or "")
        category_path = [] if prod is None else (prod["category_path"] or [])
        derived_gender = ""
        if prod is not None and "derived_gender" in prod:
            derived_gender = prod["derived_gender"] or ""
        category_text = " ".join(category_path)
        price = 0.0 if prod is None else float(prod["price_cents"] or 0)
        pop = 0.0 if prod is None else float(prod["popularity_prior"] or 0)
        pgender = tf.product_gender_label(derived_gender) or tf.product_gender(category_path, title)

        row = {
            "product_id": pid,
            "title": title,
            "brand": brand,
            "color": color,
            "category_path": category_path,
            "price_cents": price,
            "popularity_prior": pop,
            "bm25_score": float(h.get("bm25", 0.0) or 0.0),
            "bm25_rank": float(h.get("bm25_rank", 0.0) or 0.0),
            "title_bm25_score": float(h.get("title_bm25", 0.0) or 0.0),
            "category_path_bm25_score": float(h.get("category_path_bm25", 0.0) or 0.0),
            "category_bm25_score": float(h.get("category_bm25", 0.0) or 0.0),
            "bullets_bm25_score": float(h.get("bullets_bm25", 0.0) or 0.0),
            "description_bm25_score": float(h.get("description_bm25", 0.0) or 0.0),
            "brand_bm25_score": float(h.get("brand_bm25", 0.0) or 0.0),
            "knn_score": float(h.get("knn", 0.0) or 0.0),
            "knn_rank": float(h.get("knn_rank", 0.0) or 0.0),
            "rrf_score": float(h.get("rrf", 0.0) or 0.0),
            "title_length_tokens": float(len(tf.tokenize(title))),
            "query_length_tokens": qlen,
            "query_has_brand": qbrand,
            "query_has_color": qcolor,
            "query_has_category_token": qcategory,
            "query_has_size_pattern": qsize,
            "query_gender_intent": qgender,
            "product_gender": pgender,
            "gender_intent_match": tf.gender_intent_match(qgender, pgender),
            "gender_intent_mismatch": tf.gender_intent_mismatch(qgender, pgender),
            "product_brand_match": tf.product_brand_match(query, brand),
            "product_brand_token_overlap": tf.product_brand_token_overlap(query, brand),
            "product_color_match": tf.product_color_match(query, color),
            "title_query_token_coverage": tf.query_token_coverage(query, title, brand_tokens),
            "category_query_token_coverage": tf.query_token_coverage(
                query, category_text, brand_tokens
            ),
            "product_category_token_overlap": tf.token_overlap_fraction(query, category_text),
            "title_exact_query_match": tf.exact_query_phrase_match(query, title),
            "user_brand_affinity": float(user_brand_affinity.get(brand, 0.0) or 0.0),
            "query_affordability_intent": qaffordability,
            "affordability_price_score": tf.affordability_price_score(qaffordability, price),
            "brand_family_match": tf.brand_family_match(query, brand, title),
            "subbrand_title_match": tf.subbrand_title_match(query, title),
            "price_log_cents": math.log1p(price),
        }
        row["features"] = {name: float(row[name]) for name in FEATURE_NAMES}
        rows.append(row)

    return pd.DataFrame(rows)
