"""Turn raw search_events into LightGBM-ready training rows.

For every impression we emit one row:
  features = ordered FEATURE_NAMES from pipelines.features
  label    = ESCI gain mapped E=4, S=3, C=2, I=0; 0 if (query, product) is unjudged
  split    = train | val | test  (deterministic hash on query_id; 80/10/10)

Run pipelines.label.bge_teacher after this job to add offline BGE teacher
scores and weak pseudo labels for unjudged rows before LTR training.

Why ESCI labels and not clicks: with the synthetic click model, P(click) is
dominated by examination(rank), so a click-trained LTR fits position rather
than relevance. We use ESCI as the supervised signal; click data's role
moves to personalization features in Phase 6 (see PRD).

Run: docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.label.build_training_rows
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict

import pandas as pd
from psycopg.types.json import Json

from pipelines.common import db
from pipelines.common.logging import configure
from pipelines.features import FEATURE_NAMES
from pipelines.features import transforms as tf

log = configure("label.build_training_rows")

# Fraction of training rows whose user_id is masked to '' so the model
# learns "no personalization signal == 0" instead of overfitting on the
# affinity feature being almost always present. PRD R7 mitigation.
ANON_MASK_FRACTION = float(os.getenv("ANON_MASK_FRACTION", "0.30"))
ANON_MASK_SEED = int(os.getenv("ANON_MASK_SEED", "1337"))

# ESCI to integer label gain index. label_gain default for LightGBM is
# [0, 1, 3, 7, 15, ...] (i.e., 2^i - 1 for label i). We use:
#   I -> 0,  C -> 2 (gain 3),  S -> 3 (gain 7),  E -> 4 (gain 15)
# That ordering preserves E > S > C > I and gives strong discrimination
# between Exact and lower grades.
ESCI_LABEL = {"I": 0, "C": 2, "S": 3, "E": 4}


def split_for(query_id: str) -> str:
    h = int(hashlib.sha1(query_id.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def _load_events() -> pd.DataFrame:
    log.info("loading impressions from search_events")
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT e.query_id, e.query, e.user_id, e.product_id, e.position, e.retrieval_scores, e.ts
            FROM search_events e
            JOIN products USING (product_id)
            WHERE e.event_type = 'impression'
            """
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    log.info("loaded %d impression rows", len(df))
    return df


def _load_products() -> pd.DataFrame:
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT product_id, title, COALESCE(brand,'') AS brand,
                   COALESCE(color,'') AS color,
                   COALESCE(price_cents, 0) AS price_cents,
                   COALESCE(popularity_prior, 0) AS popularity_prior,
                   COALESCE(category_path, ARRAY[]::TEXT[]) AS category_path
            FROM products
            """
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _load_esci_judgments() -> dict[tuple[str, str], str]:
    """Return {(query_text, product_id) -> 'E'/'S'/'C'/'I'}."""
    with db.conn() as c, c.cursor() as cur:
        cur.execute("SELECT query, product_id, esci_label FROM esci_judgments")
        out: dict[tuple[str, str], str] = {}
        for q, pid, lbl in cur.fetchall():
            out[(q, pid)] = lbl
    log.info("loaded %d ESCI judgments", len(out))
    return out


def _load_user_brand_affinity() -> dict[str, dict[str, float]]:
    """Recompute per-user brand share from clicks (must match user_aggs)."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT e.user_id, COALESCE(p.brand, '')
            FROM search_events e
            JOIN products p USING (product_id)
            WHERE e.event_type = 'click' AND e.user_id IS NOT NULL
            """
        )
        for u, b in cur.fetchall():
            if not u or not b:
                continue
            counts[u][b] += 1
    out: dict[str, dict[str, float]] = {}
    for u, c in counts.items():
        total = sum(c.values()) + 1
        out[u] = {brand: cnt / total for brand, cnt in c.items()}
    log.info("computed brand affinity for %d users", len(out))
    return out


def main() -> int:
    impressions = _load_events()
    if impressions.empty:
        log.error("no impression events; run the simulator first")
        return 1

    products = _load_products()
    judgments = _load_esci_judgments()
    user_brand_aff = _load_user_brand_affinity()

    # Score decode
    def _score(d: dict | None, key: str) -> float:
        if not d:
            return 0.0
        return float(d.get(key, 0.0) or 0.0)

    impressions["scores"] = impressions["retrieval_scores"].apply(
        lambda v: v if isinstance(v, dict) else (json.loads(v) if v else {})
    )
    impressions["bm25_score"] = impressions["scores"].apply(lambda d: _score(d, "bm25"))
    impressions["knn_score"]  = impressions["scores"].apply(lambda d: _score(d, "knn"))
    impressions["rrf_score"]  = impressions["scores"].apply(lambda d: _score(d, "rrf"))

    impressions["bm25_rank"] = (
        impressions.groupby("query_id")["bm25_score"]
        .rank(ascending=False, method="min")
    )
    impressions["knn_rank"] = (
        impressions.groupby("query_id")["knn_score"]
        .rank(ascending=False, method="min")
    )

    df = impressions.merge(products, on="product_id", how="left")
    df["price_log_cents"]      = df["price_cents"].apply(lambda x: math.log1p(float(x or 0)))
    df["popularity_prior"]     = df["popularity_prior"].fillna(0.0).astype(float)
    df["title_length_tokens"]  = df["title"].apply(lambda t: float(len(tf.tokenize(t))))

    # Brand / color tokens for interaction features
    brand_token_set = frozenset(
        t for b in products["brand"].dropna().unique()
        if isinstance(b, str) and b for t in tf.tokenize(b)
    )
    color_token_set = frozenset(
        t for c in products["color"].dropna().unique()
        if isinstance(c, str) for t in tf.tokenize(c)
    )
    category_token_set = frozenset(
        t
        for cp in products["category_path"].dropna()
        for part in (cp or [])
        for t in tf.tokenize(part)
    )

    df["query_length_tokens"]    = df["query"].apply(tf.query_length_tokens)
    df["query_has_brand"]        = df["query"].apply(lambda q: tf.query_has_brand(q, brand_token_set))
    df["query_has_color"]        = df["query"].apply(lambda q: tf.query_has_color(q, color_token_set))
    df["query_has_category_token"] = df["query"].apply(
        lambda q: tf.query_has_category_token(q, category_token_set)
    )
    df["query_has_size_pattern"] = df["query"].apply(tf.query_has_size_pattern)
    df["query_affordability_intent"] = df["query"].apply(tf.query_affordability_intent)
    df["affordability_price_score"] = [
        tf.affordability_price_score(qai, price)
        for qai, price in zip(df["query_affordability_intent"], df["price_cents"])
    ]
    df["query_gender_intent"]    = df["query"].apply(tf.query_gender_intent)
    df["product_gender"]         = df["category_path"].apply(tf.product_gender)
    df["gender_intent_match"] = [
        tf.gender_intent_match(qg, pg)
        for qg, pg in zip(df["query_gender_intent"], df["product_gender"])
    ]
    df["gender_intent_mismatch"] = [
        tf.gender_intent_mismatch(qg, pg)
        for qg, pg in zip(df["query_gender_intent"], df["product_gender"])
    ]
    df["product_brand_match"] = [
        tf.product_brand_match(q, b) for q, b in zip(df["query"], df["brand"])
    ]
    df["product_brand_token_overlap"] = [
        tf.product_brand_token_overlap(q, b) for q, b in zip(df["query"], df["brand"])
    ]
    df["product_color_match"] = [
        tf.product_color_match(q, c) for q, c in zip(df["query"], df["color"])
    ]
    df["title_query_token_coverage"] = [
        tf.query_token_coverage(q, t, brand_token_set)
        for q, t in zip(df["query"], df["title"])
    ]
    df["category_query_token_coverage"] = [
        tf.query_token_coverage(q, " ".join(cp or []), brand_token_set)
        for q, cp in zip(df["query"], df["category_path"])
    ]
    df["product_category_token_overlap"] = [
        tf.token_overlap_fraction(q, " ".join(cp or []))
        for q, cp in zip(df["query"], df["category_path"])
    ]
    df["title_exact_query_match"] = [
        tf.exact_query_phrase_match(q, t) for q, t in zip(df["query"], df["title"])
    ]

    # Mask a deterministic fraction of (query_id, user_id) pairs to '' so
    # the model has training signal for the anonymous case. We mask by
    # query_id (not row) so that a query is consistently anonymous or
    # personalised — this matches inference where every row of a request
    # has the same user.
    rng = random.Random(ANON_MASK_SEED)
    masked_qids = {
        qid for qid in df["query_id"].unique()
        if rng.random() < ANON_MASK_FRACTION
    }
    log.info(
        "anonymous mask: %d/%d queries set to user_id='' (fraction=%.2f)",
        len(masked_qids), df["query_id"].nunique(), ANON_MASK_FRACTION,
    )
    masked_mask = df["query_id"].isin(masked_qids)
    df.loc[masked_mask, "user_id"] = None

    def _user_brand_aff(u, b):
        if not u or not b:
            return 0.0
        per_brand = user_brand_aff.get(u)
        if not per_brand:
            return 0.0
        return per_brand.get(b, 0.0)

    df["user_brand_affinity"] = [
        _user_brand_aff(u, b) for u, b in zip(df["user_id"], df["brand"])
    ]

    # ESCI label: text key (query_text, product_id)
    keys = list(zip(df["query"], df["product_id"]))
    df["label"] = [ESCI_LABEL.get(judgments.get(k), 0) for k in keys]
    log.info(
        "label distribution (ESCI): %s",
        df["label"].value_counts().sort_index().to_dict(),
    )
    judged_share = (df["label"] > 0).mean()
    log.info("rows with non-zero ESCI label: %.1f%%", 100 * judged_share)

    df["split"] = df["query_id"].apply(split_for)
    log.info("split sizes: %s", df["split"].value_counts().to_dict())

    feat_cols = list(FEATURE_NAMES)
    df["features"] = df.apply(
        lambda r: {name: float(r[name]) for name in feat_cols},
        axis=1,
    )

    log.info("writing %d training_rows", len(df))
    with db.conn() as c, c.cursor() as cur:
        cur.execute("TRUNCATE training_rows")
        records = list(zip(
            df["query_id"], df["product_id"], df["query"], df["user_id"], df["ts"],
            df["features"].apply(Json), df["label"].astype(float), df["split"],
        ))
        cur.executemany(
            "INSERT INTO training_rows (query_id, product_id, query, user_id, ts, features, label, split)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            records,
        )
    c.commit()
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
