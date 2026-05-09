"""Use BGE offline as an optional teacher for LTR training rows.

This job intentionally keeps the cross-encoder out of the search hot path.
It scores existing training rows offline, stores the raw teacher score in the
row's JSON feature snapshot for audit/debugging, and can optionally upgrade
train-only unjudged rows to low-confidence pseudo labels. Known ESCI judgments
remain authoritative.

Run: docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.label.bge_teacher
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
from psycopg.types.json import Json
from tqdm import tqdm

from pipelines.common import db
from pipelines.common.logging import configure
from pipelines.common.torch_device import move_wrapped_model_to_device, resolve_torch_device

log = configure("label.bge_teacher")

BGE_TEACHER_MODEL = os.getenv("BGE_TEACHER_MODEL", "BAAI/bge-reranker-v2-m3")
BGE_TEACHER_DEVICE = os.getenv("BGE_TEACHER_DEVICE", "auto")
BGE_TEACHER_BATCH_SIZE = int(os.getenv("BGE_TEACHER_BATCH_SIZE", "32"))
BGE_TEACHER_MAX_ROWS = int(os.getenv("BGE_TEACHER_MAX_ROWS", "0"))
BGE_TEACHER_PSEUDO_LABELS = os.getenv(
    "BGE_TEACHER_PSEUDO_LABELS",
    os.getenv("LTR_PSEUDO_LABELS", "0"),
).lower() in {"1", "true", "yes"}
BGE_TEACHER_PSEUDO_LABEL_WEIGHT = float(
    os.getenv("BGE_TEACHER_PSEUDO_LABEL_WEIGHT", os.getenv("LTR_PSEUDO_LABEL_WEIGHT", "0.25"))
)


def product_text(row: pd.Series) -> str:
    parts = [
        f"Title: {row.get('title') or ''}",
        f"Brand: {row.get('brand') or ''}",
    ]
    raw_category_path = row.get("category_path")
    category_path = raw_category_path if isinstance(raw_category_path, (list, tuple)) else []
    category = " > ".join(str(part) for part in category_path) if category_path else (row.get("category") or "")
    if category:
        parts.append(f"Category: {category}")
    price_cents = int(row.get("price_cents") or 0)
    if price_cents > 0:
        parts.append(f"Price: ${price_cents / 100:.2f}")
    else:
        parts.append("Price: unknown")
    return "\n".join(parts)


def teacher_percentiles(df: pd.DataFrame, score_col: str = "bge_teacher_score") -> pd.Series:
    """Return query-local [0, 1] score percentiles, with flat groups set to 0."""
    group_col = "query_id" if "query_id" in df.columns else "query"
    out = pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
    for _group_key, group in df.groupby(group_col, sort=False):
        scores = group[score_col].astype(float)
        lo = float(scores.min())
        hi = float(scores.max())
        if hi <= lo:
            continue
        out.loc[group.index] = (scores - lo) / (hi - lo)
    return out


def teacher_grade(percentile: float) -> float:
    """Quantize teacher confidence for LambdaRank label_gain indices.

    Grades 3 and 4 stay reserved for human/ESCI relevance labels. The teacher
    can add weak positive signal to unjudged rows, but cannot manufacture an
    Exact/Substitute-level label.
    """
    if percentile >= 0.85:
        return 2.0
    if percentile >= 0.60:
        return 1.0
    return 0.0


def _coerce_features(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        return dict(json.loads(value))
    return {}


def add_teacher_labels(
    rows: pd.DataFrame,
    judged_pairs: set[tuple[str, str]],
    *,
    pseudo_labels_enabled: bool = False,
    pseudo_weight: float = 0.25,
) -> pd.DataFrame:
    """Attach BGE teacher scores and optionally update train-only unjudged rows."""
    out = rows.copy()
    out["bge_teacher_percentile"] = teacher_percentiles(out)
    if "sample_weight" not in out.columns:
        out["sample_weight"] = 1.0
    else:
        out["sample_weight"] = out["sample_weight"].astype(float)

    features: list[dict[str, float]] = []
    labels: list[float] = []
    weights: list[float] = []
    for row in out.itertuples(index=False):
        feature_map = _coerce_features(getattr(row, "features", {}))
        score = float(getattr(row, "bge_teacher_score"))
        percentile = float(getattr(row, "bge_teacher_percentile"))
        feature_map["bge_teacher_score"] = score
        feature_map["bge_teacher_percentile"] = percentile
        features.append(feature_map)

        base_label = float(getattr(row, "label"))
        pair = (str(getattr(row, "query")), str(getattr(row, "product_id")))
        sample_weight = float(getattr(row, "sample_weight", 1.0))
        split = str(getattr(row, "split", ""))
        pseudo_grade = teacher_grade(percentile)
        if (
            pseudo_labels_enabled
            and split == "train"
            and pair not in judged_pairs
            and pseudo_grade > base_label
        ):
            labels.append(pseudo_grade)
            sample_weight = pseudo_weight
        else:
            labels.append(base_label)
        weights.append(sample_weight)

    out["features"] = features
    out["label"] = labels
    out["sample_weight"] = weights
    return out


def ensure_training_rows_schema() -> None:
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE training_rows "
            "ADD COLUMN IF NOT EXISTS sample_weight REAL NOT NULL DEFAULT 1"
        )
        c.commit()


def _load_training_rows() -> pd.DataFrame:
    ensure_training_rows_schema()
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT query_id, product_id, query, features, label, split, sample_weight
            FROM training_rows
            ORDER BY query_id, product_id
            """
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if BGE_TEACHER_MAX_ROWS > 0:
        df = df.head(BGE_TEACHER_MAX_ROWS).copy()
        log.warning("BGE_TEACHER_MAX_ROWS=%d limits scoring to a subset", BGE_TEACHER_MAX_ROWS)
    log.info("loaded %d training rows", len(df))
    return df


def _load_products() -> pd.DataFrame:
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT product_id, title, COALESCE(brand,'') AS brand,
                   COALESCE(color,'') AS color,
                   COALESCE(category,'') AS category,
                   COALESCE(category_path, ARRAY[]::TEXT[]) AS category_path,
                   COALESCE(price_cents, 0) AS price_cents
            FROM products
            """
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _load_judged_pairs() -> set[tuple[str, str]]:
    with db.conn() as c, c.cursor() as cur:
        cur.execute("SELECT query, product_id FROM esci_judgments")
        pairs = {(str(q), str(pid)) for q, pid in cur.fetchall()}
    log.info("loaded %d judged ESCI query/product pairs", len(pairs))
    return pairs


def _batched(items: list[list[str]], size: int) -> Iterable[list[list[str]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def score_with_bge(rows: pd.DataFrame) -> list[float]:
    from sentence_transformers import CrossEncoder

    device = resolve_torch_device(BGE_TEACHER_DEVICE, "BGE_TEACHER_DEVICE")
    log.info("loading BGE teacher model=%s device=%s", BGE_TEACHER_MODEL, device)
    model = move_wrapped_model_to_device(CrossEncoder(BGE_TEACHER_MODEL, device=device), device)
    pairs = [[str(row.query), str(row.document_text)] for row in rows.itertuples(index=False)]
    scores: list[float] = []
    total_batches = (len(pairs) + BGE_TEACHER_BATCH_SIZE - 1) // BGE_TEACHER_BATCH_SIZE
    for batch in tqdm(_batched(pairs, BGE_TEACHER_BATCH_SIZE), total=total_batches, desc="BGE teacher", unit="batch"):
        pred = model.predict(batch, batch_size=BGE_TEACHER_BATCH_SIZE, show_progress_bar=False)
        scores.extend(float(x) for x in pred)
    return scores


def _write_rows(rows: pd.DataFrame) -> None:
    records = list(
        zip(
            rows["features"].apply(Json),
            rows["label"].astype(float),
            rows["sample_weight"].astype(float),
            rows["query_id"],
            rows["product_id"],
        )
    )
    with db.conn() as c, c.cursor() as cur:
        cur.executemany(
            """
            UPDATE training_rows
            SET features = %s, label = %s, sample_weight = %s
            WHERE query_id = %s AND product_id = %s
            """,
            records,
        )
        c.commit()


def main() -> int:
    rows = _load_training_rows()
    if rows.empty:
        log.error("training_rows is empty; run pipelines.label.build_training_rows first")
        return 1

    products = _load_products()
    judged_pairs = _load_judged_pairs()
    df = rows.merge(products, on="product_id", how="left")
    df["document_text"] = df.apply(product_text, axis=1)
    df["bge_teacher_score"] = score_with_bge(df)

    labelled = add_teacher_labels(
        df,
        judged_pairs=judged_pairs,
        pseudo_labels_enabled=BGE_TEACHER_PSEUDO_LABELS,
        pseudo_weight=BGE_TEACHER_PSEUDO_LABEL_WEIGHT,
    )
    label_source = "BGE train pseudo-labels enabled" if BGE_TEACHER_PSEUDO_LABELS else "BGE scores only"
    log.info(
        "label distribution after BGE distillation (%s): %s",
        label_source,
        labelled["label"].value_counts().sort_index().to_dict(),
    )
    log.info(
        "sample weight distribution after BGE distillation: %s",
        labelled["sample_weight"].value_counts().sort_index().to_dict(),
    )
    _write_rows(labelled)
    log.info("updated %d training rows with offline BGE teacher features", len(labelled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
