"""Turn serving-aligned retrieval candidates into LightGBM-ready training rows.

For every offline retrieval candidate we emit one row:
  features = ordered FEATURE_NAMES from pipelines.features
  label    = ESCI gain mapped E=4, S=3, C=2, I=0; 0 if (query, product) is unjudged
  split    = ESCI's canonical query split, or a deterministic normalized-query
             hash fallback for non-ESCI queries

Weak pseudo labels are off by default. Set LTR_PSEUDO_LABELS=1 to apply
train-only lexical pseudo labels with LTR_PSEUDO_LABEL_WEIGHT.

Why ESCI labels and not clicks: with the synthetic click model, P(click) is
dominated by examination(rank), so a click-trained LTR fits position rather
than relevance. We use ESCI as the supervised signal; click data's role
moves to personalization features in Phase 6 (see PRD).

Run: docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.label.build_training_rows
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict

import pandas as pd
from psycopg.types.json import Json

from pipelines.common import db
from pipelines.common import training_row_builds
from pipelines.common.logging import configure
from pipelines.features import FEATURE_NAMES
from pipelines.features._generated import SCHEMA_VERSION
from pipelines.features import ltr_rows
from pipelines.features import transforms as tf

log = configure("label.build_training_rows")

TRAINING_ROW_SOURCE = os.getenv("TRAINING_ROW_SOURCE", "offline_candidates")
LTR_CAND_N = int(os.getenv("LTR_CAND_N", os.getenv("EVAL_CAND_N", "200")))
PSEUDO_LABELS_ENABLED = os.getenv("LTR_PSEUDO_LABELS", "0").lower() in {"1", "true", "yes"}
PSEUDO_LABEL_WEIGHT = float(os.getenv("LTR_PSEUDO_LABEL_WEIGHT", "0.25"))
HARD_NEGATIVES_ENABLED = os.getenv("LTR_HARD_NEGATIVES", "1").lower() in {"1", "true", "yes"}
HARD_NEGATIVE_WEIGHT = float(os.getenv("LTR_HARD_NEGATIVE_WEIGHT", "2.0"))
HARD_NEGATIVE_RANK_GAP = int(os.getenv("LTR_HARD_NEGATIVE_RANK_GAP", "50"))

# ESCI to integer label gain index. label_gain default for LightGBM is
# [0, 1, 3, 7, 15, ...] (i.e., 2^i - 1 for label i). We use:
#   I -> 0,  C -> 2 (gain 3),  S -> 3 (gain 7),  E -> 4 (gain 15)
# That ordering preserves E > S > C > I and gives strong discrimination
# between Exact and lower grades.
ESCI_LABEL = {"I": 0, "C": 2, "S": 3, "E": 4}

BRAND_MATCH_PSEUDO_LABEL = int(os.getenv("BRAND_MATCH_PSEUDO_LABEL", "3"))
BRAND_FAMILY_MATCH_PSEUDO_LABEL = int(os.getenv("BRAND_FAMILY_MATCH_PSEUDO_LABEL", "3"))
TITLE_BRAND_PSEUDO_LABEL = int(os.getenv("TITLE_BRAND_PSEUDO_LABEL", "2"))
CATEGORY_FULL_MATCH_PSEUDO_LABEL = int(os.getenv("CATEGORY_FULL_MATCH_PSEUDO_LABEL", "3"))
CATEGORY_PARTIAL_MATCH_PSEUDO_LABEL = int(os.getenv("CATEGORY_PARTIAL_MATCH_PSEUDO_LABEL", "2"))
CATEGORY_PARTIAL_MATCH_MIN_COVERAGE = float(os.getenv("CATEGORY_PARTIAL_MATCH_MIN_COVERAGE", "0.5"))
ATTRIBUTE_MATCH_PSEUDO_LABEL = int(os.getenv("ATTRIBUTE_MATCH_PSEUDO_LABEL", "2"))
MULTI_ATTRIBUTE_MATCH_PSEUDO_LABEL = int(os.getenv("MULTI_ATTRIBUTE_MATCH_PSEUDO_LABEL", "3"))

ACCESSORY_INTENT_TOKENS = frozenset(
    {
        "accessories",
        "accessory",
        "care",
        "cleaner",
        "cleaning",
        "cover",
        "covers",
        "insole",
        "insoles",
        "lace",
        "laces",
        "polish",
        "protector",
        "protectors",
        "replacement",
        "strap",
        "straps",
    }
)
CORE_PRODUCT_INTENT_TOKENS = frozenset(
    {
        "boot",
        "boots",
        "sandal",
        "sandals",
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
    }
)
ATTRIBUTE_INTENT_TOKENS = frozenset(
    {
        "athletic",
        "basketball",
        "baby",
        "casual",
        "chelsea",
        "cotton",
        "denim",
        "dress",
        "formal",
        "hiking",
        "infant",
        "kid",
        "kids",
        "leather",
        "mesh",
        "running",
        "slim",
        "stainless",
        "steel",
        "suede",
        "teen",
        "toddler",
        "trail",
        "training",
        "walking",
        "waterproof",
        "wide",
        "wool",
        "workout",
        "youth",
    }
)
SIZE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:oz|ml|gb|tb|in|cm|mm|kg|lb|l|g)\b",
    re.IGNORECASE,
)

normalize_retrieval_ranks = ltr_rows.normalize_retrieval_ranks


def split_key_for_query(query: object) -> str:
    return " ".join(tf.tokenize(str(query or "")))


def split_for(query: object) -> str:
    split_key = split_key_for_query(query)
    h = int(hashlib.sha1(split_key.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def train_val_split_for(query: object) -> str:
    split_key = split_key_for_query(query)
    h = int(hashlib.sha1(split_key.encode()).hexdigest(), 16) % 100
    return "val" if 80 <= h < 90 else "train"


def split_from_canonical_esci(query: object, canonical_split: str) -> str:
    if canonical_split == "test":
        return "test"
    if canonical_split == "train":
        return train_val_split_for(query)
    return canonical_split


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


def _load_esci_query_splits() -> dict[str, str]:
    """Return {normalized_query_text -> canonical ESCI split}."""
    with db.conn() as c, c.cursor() as cur:
        cur.execute("SELECT query, split FROM esci_judgments GROUP BY query, split")
        rows = cur.fetchall()

    out: dict[str, str] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)
    for query, split in rows:
        key = split_key_for_query(query)
        existing = out.get(key)
        if existing is not None and existing != split:
            conflicts[key].update({existing, split})
            continue
        out[key] = split

    if conflicts:
        examples = ", ".join(
            f"{key!r}: {sorted(splits)}"
            for key, splits in list(conflicts.items())[:5]
        )
        raise RuntimeError(f"conflicting ESCI splits for normalized queries: {examples}")

    log.info("loaded %d canonical ESCI query splits", len(out))
    return out


def _load_esci_queries() -> pd.DataFrame:
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT query_id, query
            FROM esci_judgments
            GROUP BY query_id, query
            ORDER BY query_id
            """
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["query_id", "query"])


def assign_splits(
    df: pd.DataFrame,
    esci_query_splits: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Assign train/val/test by canonical ESCI query split, then stable query hash."""
    out = df.copy()
    split_map = esci_query_splits or {}
    split_keys = out["query"].apply(split_key_for_query)
    out["split"] = [
        split_from_canonical_esci(key, split_map[key]) if key in split_map else split_for(key)
        for key in split_keys
    ]
    return out


def build_candidate_training_frame(
    queries: pd.DataFrame,
    products: pd.DataFrame,
    retriever,
    cand_n: int,
) -> pd.DataFrame:
    """Build LTR rows from the same offline candidate distribution used in eval."""
    brand_tokens, color_tokens, category_tokens = ltr_rows.token_sets(products)
    frames: list[pd.DataFrame] = []
    ts = pd.Timestamp.now(tz="UTC")

    for qid, query in zip(queries["query_id"], queries["query"]):
        hits = retriever.search(str(query), cand_n)
        if not hits:
            continue
        frame = ltr_rows.build_feature_frame(
            query=str(query),
            hits=hits,
            products=products,
            brand_tokens=brand_tokens,
            color_tokens=color_tokens,
            category_tokens=category_tokens,
            user_brand_affinity=None,
        )
        if frame.empty:
            continue
        frame.insert(0, "query", str(query))
        frame.insert(0, "query_id", str(qid))
        frame["user_id"] = None
        frame["ts"] = ts
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def apply_pseudo_labels_for_training(
    df: pd.DataFrame,
    judged_pairs: set[tuple[str, str]],
    *,
    enabled: bool,
    pseudo_weight: float,
) -> pd.DataFrame:
    """Optionally apply weak labels to train rows only and downweight them."""
    out = df.copy()
    if "sample_weight" not in out.columns:
        out["sample_weight"] = 1.0
    else:
        out["sample_weight"] = out["sample_weight"].astype(float)

    if not enabled:
        return out

    before = out["label"].copy()
    train_mask = out["split"] == "train"
    labeled_train = apply_lexical_relevance_labels(out.loc[train_mask], judged_pairs)
    out.loc[train_mask, "label"] = labeled_train["label"]

    pseudo_mask = train_mask & (before <= 0) & (out["label"] > before)
    out.loc[pseudo_mask, "sample_weight"] = pseudo_weight
    return out


def apply_supervision(
    df: pd.DataFrame,
    judgments: dict[tuple[str, str], str],
    *,
    pseudo_labels_enabled: bool,
    pseudo_weight: float,
) -> pd.DataFrame:
    out = df.copy()
    keys = list(zip(out["query"], out["product_id"]))
    out["label"] = [ESCI_LABEL.get(judgments.get(k), 0) for k in keys]
    out["sample_weight"] = 1.0
    return apply_pseudo_labels_for_training(
        out,
        set(judgments.keys()),
        enabled=pseudo_labels_enabled,
        pseudo_weight=pseudo_weight,
    )


def apply_hard_negative_weights(
    df: pd.DataFrame,
    *,
    enabled: bool,
    hard_negative_weight: float,
    rank_gap: int,
) -> pd.DataFrame:
    """Upweight train negatives that are plausible retrieval mistakes.

    These rows are still labeled irrelevant; the larger weight teaches the
    ranker to resolve cases where lexical and vector retrieval disagree or
    where the candidate lands in the competitive middle of the fused list.
    """
    out = df.copy()
    if "sample_weight" not in out.columns:
        out["sample_weight"] = 1.0
    else:
        out["sample_weight"] = out["sample_weight"].astype(float)
    if not enabled or out.empty:
        return out

    train_negative = (out["split"] == "train") & (out["label"].astype(float) <= 0)
    rank_disagreement = (
        (out["bm25_rank"].astype(float) - out["knn_rank"].astype(float)).abs()
        >= float(rank_gap)
    )
    rrf = out["rrf_score"].astype(float)
    middle_rrf = pd.Series(False, index=out.index)
    for _qid, idx in out.groupby("query_id").groups.items():
        group_rrf = rrf.loc[idx]
        if len(group_rrf) < 4:
            continue
        lo = group_rrf.quantile(0.25)
        hi = group_rrf.quantile(0.75)
        middle_rrf.loc[idx] = group_rrf.between(lo, hi, inclusive="both")

    hard_negative = train_negative & (rank_disagreement | middle_rrf)
    out.loc[hard_negative, "sample_weight"] = out.loc[
        hard_negative, "sample_weight"
    ].clip(lower=hard_negative_weight)
    return out


def ensure_training_rows_schema() -> None:
    with db.conn() as c, c.cursor() as cur:
        training_row_builds.ensure_schema(cur)
        c.commit()


def write_training_rows(df: pd.DataFrame, *, build_id: int) -> None:
    split_counts = {
        str(split): int(count)
        for split, count in df["split"].value_counts().sort_index().items()
    }
    records = list(zip(
        df["query_id"], df["product_id"], df["query"], df["user_id"], df["ts"],
        df["features"].apply(Json), df["label"].astype(float), df["split"],
        df["sample_weight"].astype(float), [build_id] * len(df),
    ))
    with db.conn() as c, c.cursor() as cur:
        training_row_builds.ensure_schema(cur)
        cur.executemany(
            "INSERT INTO training_rows"
            " (query_id, product_id, query, user_id, ts, features, label, split,"
            " sample_weight, build_id)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            records,
        )
        training_row_builds.mark_training_row_build_ready_in_cursor(
            cur,
            build_id,
            row_count=int(len(df)),
            query_count=int(df["query_id"].nunique()),
            split_counts=split_counts,
        )
        c.commit()


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


def _category_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value if part)
    return str(value or "")


def _product_type_compatible(query: object, title: object, category_path: object) -> bool:
    query_tokens = set(tf.tokenize(str(query or "")))
    if not query_tokens:
        return True
    if query_tokens & ACCESSORY_INTENT_TOKENS:
        return True
    if not (query_tokens & CORE_PRODUCT_INTENT_TOKENS):
        return True

    product_tokens = set(tf.tokenize(f"{title or ''} {_category_text(category_path)}"))
    return bool(product_tokens & CORE_PRODUCT_INTENT_TOKENS) and not bool(
        product_tokens & ACCESSORY_INTENT_TOKENS
    )


def _size_matches(query: object, text: str) -> bool:
    query_sizes = {
        " ".join(tf.tokenize(match.group(0)))
        for match in SIZE_RE.finditer(str(query or ""))
    }
    if not query_sizes:
        return False
    text_sizes = {
        " ".join(tf.tokenize(match.group(0)))
        for match in SIZE_RE.finditer(text)
    }
    return bool(query_sizes & text_sizes)


def _text_attribute_match_count(query: object, title: object, category_path: object) -> int:
    text = f"{title or ''} {_category_text(category_path)}"
    query_tokens = set(tf.tokenize(str(query or "")))
    text_tokens = set(tf.tokenize(text))

    matched = len((query_tokens & ATTRIBUTE_INTENT_TOKENS) & text_tokens)
    if _size_matches(query, text):
        matched += 1
    return matched


def apply_lexical_relevance_labels(
    df: pd.DataFrame,
    judged_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    """Upgrade unjudged rows with visible lexical relevance signals.

    These are model-training labels, not serving-time ordering rules. ESCI
    judgments remain authoritative, including explicit irrelevant judgments.
    """
    out = df.copy()
    judged_mask = pd.Series(
        [(str(q), str(pid)) in judged_pairs for q, pid in zip(out["query"], out["product_id"])],
        index=out.index,
    )
    def _float_col(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(0.0, index=out.index)
        return out[name].astype(float)

    unjudged_brand_query = (_float_col("query_has_brand") > 0) & ~judged_mask

    brand_match = unjudged_brand_query & (
        (_float_col("product_brand_match") > 0)
        | (_float_col("brand_family_match") > 0)
    )
    out.loc[brand_match, "label"] = out.loc[brand_match, "label"].clip(
        lower=max(BRAND_MATCH_PSEUDO_LABEL, BRAND_FAMILY_MATCH_PSEUDO_LABEL)
    )

    title_match = (
        unjudged_brand_query
        & ~brand_match
        & (
            (_float_col("title_exact_query_match") > 0)
            | (_float_col("subbrand_title_match") > 0)
        )
    )
    out.loc[title_match, "label"] = out.loc[title_match, "label"].clip(
        lower=TITLE_BRAND_PSEUDO_LABEL
    )

    unjudged_category_query = (
        (_float_col("query_has_category_token") > 0)
        & (_float_col("query_has_brand") == 0)
        & ~judged_mask
    )
    if {"title", "category_path"}.issubset(out.columns):
        product_type_compatible = pd.Series(
            [
                _product_type_compatible(q, title, category_path)
                for q, title, category_path in zip(out["query"], out["title"], out["category_path"])
            ],
            index=out.index,
        )
        unjudged_category_query = unjudged_category_query & product_type_compatible

    category_coverage = _float_col("category_query_token_coverage")
    full_category_match = unjudged_category_query & (category_coverage >= 1.0)
    out.loc[full_category_match, "label"] = out.loc[full_category_match, "label"].clip(
        lower=CATEGORY_FULL_MATCH_PSEUDO_LABEL
    )

    partial_category_match = (
        unjudged_category_query
        & ~full_category_match
        & (category_coverage >= CATEGORY_PARTIAL_MATCH_MIN_COVERAGE)
    )
    out.loc[partial_category_match, "label"] = out.loc[partial_category_match, "label"].clip(
        lower=CATEGORY_PARTIAL_MATCH_PSEUDO_LABEL
    )

    unjudged_attribute_query = ~judged_mask
    attribute_match_count = (
        (_float_col("gender_intent_match") > 0).astype(int)
        + (_float_col("product_color_match") > 0).astype(int)
    )
    if {"title", "category_path"}.issubset(out.columns):
        text_attribute_counts = pd.Series(
            [
                _text_attribute_match_count(q, title, category_path)
                if _product_type_compatible(q, title, category_path)
                else 0
                for q, title, category_path in zip(out["query"], out["title"], out["category_path"])
            ],
            index=out.index,
        )
        attribute_match_count = attribute_match_count + text_attribute_counts

    single_attribute_match = unjudged_attribute_query & (attribute_match_count == 1)
    out.loc[single_attribute_match, "label"] = out.loc[single_attribute_match, "label"].clip(
        lower=ATTRIBUTE_MATCH_PSEUDO_LABEL
    )

    multi_attribute_match = unjudged_attribute_query & (attribute_match_count >= 2)
    out.loc[multi_attribute_match, "label"] = out.loc[multi_attribute_match, "label"].clip(
        lower=MULTI_ATTRIBUTE_MATCH_PSEUDO_LABEL
    )
    return out


def main() -> int:
    build_id = training_row_builds.begin_training_row_build(
        source=TRAINING_ROW_SOURCE,
        feature_schema_version=SCHEMA_VERSION,
        cand_n=LTR_CAND_N,
        pseudo_labels_enabled=PSEUDO_LABELS_ENABLED,
        pseudo_label_weight=PSEUDO_LABEL_WEIGHT,
        feature_names=list(FEATURE_NAMES),
    )
    log.info("started training row build_id=%d", build_id)
    try:
        products = _load_products()
        judgments = _load_esci_judgments()
        esci_query_splits = _load_esci_query_splits()

        if TRAINING_ROW_SOURCE != "offline_candidates":
            raise RuntimeError(
                "unsupported TRAINING_ROW_SOURCE="
                f"{TRAINING_ROW_SOURCE}; use offline_candidates for serving-aligned LTR rows"
            )

        from pipelines.evaluate.offline_eval import HybridRetriever

        queries = _load_esci_queries()
        retriever = HybridRetriever()
        try:
            df = build_candidate_training_frame(
                queries=queries,
                products=products,
                retriever=retriever,
                cand_n=LTR_CAND_N,
            )
        finally:
            retriever.close()

        if df.empty:
            raise RuntimeError("no offline candidate rows; index products and verify embedder/OpenSearch")

        log.info(
            "candidate rows: rows=%d queries=%d cand_n=%d avg_rows_per_query=%.1f",
            len(df),
            df["query_id"].nunique(),
            LTR_CAND_N,
            len(df) / max(df["query_id"].nunique(), 1),
        )

        df = assign_splits(df, esci_query_splits)
        df = apply_supervision(
            df,
            judgments,
            pseudo_labels_enabled=PSEUDO_LABELS_ENABLED,
            pseudo_weight=PSEUDO_LABEL_WEIGHT,
        )
        df = apply_hard_negative_weights(
            df,
            enabled=HARD_NEGATIVES_ENABLED,
            hard_negative_weight=HARD_NEGATIVE_WEIGHT,
            rank_gap=HARD_NEGATIVE_RANK_GAP,
        )
        label_source = "ESCI + train pseudo-labels" if PSEUDO_LABELS_ENABLED else "ESCI only"
        log.info("label distribution (%s): %s", label_source, df["label"].value_counts().sort_index().to_dict())
        judged_share = (df["label"] > 0).mean()
        log.info("rows with non-zero label: %.1f%%", 100 * judged_share)
        log.info("sample weight distribution: %s", df["sample_weight"].value_counts().sort_index().to_dict())

        canonical_split_rows = df["query"].apply(split_key_for_query).isin(esci_query_splits).sum()
        log.info(
            "split assignment: canonical_esci_rows=%d hash_fallback_rows=%d",
            canonical_split_rows, len(df) - canonical_split_rows,
        )
        log.info("split sizes: %s", df["split"].value_counts().to_dict())

        log.info("writing %d training_rows build_id=%d", len(df), build_id)
        write_training_rows(df, build_id=build_id)
    except Exception as exc:
        training_row_builds.mark_training_row_build_failed(build_id, exc)
        log.exception("training row build failed build_id=%d", build_id)
        return 1

    log.info("done build_id=%d", build_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
