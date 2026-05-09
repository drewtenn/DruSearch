"""Offline NDCG@10 / MRR / Recall@K evaluation against ESCI ground-truth.

Two variants, both computed on the same set of ESCI test queries:

  hybrid_rrf  — order returned by the API today (BM25 + k-NN, RRF-fused)
  ltr         — re-rank the same RRF candidates with the configured LTR model
                backend (latest MLflow version of LTR_MODEL_NAME)

We score against ESCI labels: E=4, S=3, C=2, I=0 in classic-NDCG gain space.

Run: docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.evaluate.offline_eval
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from pipelines.common import db
from pipelines.common.config import load, normalize_ltr_model_backend
from pipelines.common.logging import configure
from pipelines.features import FEATURE_NAMES
from pipelines.features import ltr_rows
from pipelines.features import transforms as tf

log = configure("evaluate.offline")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ESCI_GAIN = {"E": 4.0, "S": 3.0, "C": 2.0, "I": 0.0}
LTR_MODEL_NAME = os.getenv("LTR_MODEL_NAME", "ltr_reranker")
LTR_MODEL_BACKEND = normalize_ltr_model_backend(os.getenv("LTR_MODEL_BACKEND", "lgbm"))
EVAL_QUERIES = int(os.getenv("EVAL_QUERIES", "500"))
EVAL_K = int(os.getenv("EVAL_K", "10"))
CAND_N = int(os.getenv("EVAL_CAND_N", "200"))
EVAL_MIN_JUDGMENTS = int(os.getenv("EVAL_MIN_JUDGMENTS", "5"))


# ---------------------------------------------------------------------------
# Catalog state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Catalog:
    products: pd.DataFrame
    brand_tokens: frozenset[str]
    color_tokens: frozenset[str]
    category_tokens: frozenset[str]


def _load_catalog() -> Catalog:
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
        prod_rows = cur.fetchall()

    products = pd.DataFrame(prod_rows, columns=[
        "product_id", "title", "brand", "color", "price_cents", "popularity_prior", "category_path",
    ]).set_index("product_id")

    brand_tokens = frozenset(t for b in products["brand"].dropna().unique() for t in tf.brand_tokens(b))
    color_tokens = frozenset(t for c in products["color"].dropna().unique() for t in tf.tokenize(c))
    category_tokens = frozenset(
        t
        for cp in products["category_path"].dropna()
        for part in (cp or [])
        for t in tf.tokenize(part)
    )
    return Catalog(products=products, brand_tokens=brand_tokens, color_tokens=color_tokens, category_tokens=category_tokens)


def _load_test_queries(n: int) -> list[tuple[int, str]]:
    """Pick ESCI test-split queries with usable positive judgments in our catalog."""
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT j.query_id, j.query, COUNT(*) AS n
            FROM esci_judgments j
            JOIN products p ON p.product_id = j.product_id
            WHERE j.split = 'test'
            GROUP BY j.query_id, j.query
            HAVING COUNT(*) >= %s
               AND BOOL_OR(j.esci_label IN ('E', 'S', 'C'))
            ORDER BY j.query_id
            LIMIT %s
            """,
            (EVAL_MIN_JUDGMENTS, n),
        )
        return [(qid, q) for (qid, q, _n) in cur.fetchall()]


def _load_judgments(query_ids: list[int]) -> dict[int, dict[str, str]]:
    if not query_ids:
        return {}
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT query_id, product_id, esci_label FROM esci_judgments"
            " WHERE query_id = ANY(%s)",
            (list(query_ids),),
        )
        out: dict[int, dict[str, str]] = {}
        for qid, pid, lbl in cur.fetchall():
            out.setdefault(qid, {})[pid] = lbl
        return out


# ---------------------------------------------------------------------------
# Retrieval (talks directly to OpenSearch + the embedder, mirroring the Go path)
# ---------------------------------------------------------------------------

def _bm25_query(query: str) -> dict:
    def field_match(name: str, field: str, boost: float) -> dict:
        return {
            "match": {
                field: {
                    "query": query,
                    "boost": boost,
                    "_name": name,
                }
            }
        }

    base = {
        "bool": {
            "must": [
                {
                    "dis_max": {
                        "tie_breaker": 0.1,
                        "queries": [
                            field_match("bm25_title", "title", 2.0),
                            field_match("bm25_category_path", "category_path", 2.0),
                            field_match("bm25_category", "category", 1.5),
                            field_match("bm25_bullets", "bullets", 1.0),
                            field_match("bm25_description", "description", 1.0),
                        ],
                    }
                }
            ],
            "should": [
                {
                    "match": {
                        "brand.text": {
                            "query": query,
                            "boost": 2.5,
                            "fuzziness": "AUTO",
                            "prefix_length": 1,
                            "max_expansions": 20,
                            "_name": "bm25_brand",
                        }
                    }
                }
            ],
        }
    }
    intent = _gender_intent_label(query)
    if not intent:
        return base
    should = [_term_boost("derived_gender", intent, 8.0)]
    if intent in {"men", "women"}:
        should.append(_term_boost("derived_gender", "unisex", 4.0))
    return {
        "boosting": {
            "positive": {"bool": {"must": [base], "should": should}},
            "negative": {"terms": {"derived_gender": _opposite_gender_labels(intent)}},
            "negative_boost": 0.25,
        }
    }


def _term_boost(field: str, value: str, boost: float) -> dict:
    return {"term": {field: {"value": value, "boost": boost}}}


def _gender_intent_label(query: str) -> str:
    gender = tf.query_gender_intent(query)
    if gender == tf.GENDER_MEN:
        return "men"
    if gender == tf.GENDER_WOMEN:
        return "women"
    if gender == tf.GENDER_BOYS:
        return "boys"
    if gender == tf.GENDER_GIRLS:
        return "girls"
    if gender == tf.GENDER_UNISEX:
        return "unisex"
    return ""


def _opposite_gender_labels(intent: str) -> list[str]:
    if intent == "men":
        return ["women", "boys", "girls"]
    if intent == "women":
        return ["men", "boys", "girls"]
    if intent == "boys":
        return ["men", "women", "girls", "unisex"]
    if intent == "girls":
        return ["men", "women", "boys", "unisex"]
    if intent == "unisex":
        return ["men", "women", "boys", "girls"]
    return []


class HybridRetriever:
    def __init__(self) -> None:
        from pipelines.common import opensearch_client
        cfg = load()
        self.os = opensearch_client.client()
        self.embedder_url = cfg.embedder_url
        self.index = os.getenv("OPENSEARCH_INDEX", "products_v1")
        self._http = httpx.Client(timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def _embed(self, text: str) -> list[float]:
        r = self._http.post(f"{self.embedder_url}/embed", json={"text": text})
        r.raise_for_status()
        return r.json()["vector"]

    def search(self, query: str, n: int) -> list[dict]:
        """Return [{product_id, bm25, knn, rrf, bm25_rank, knn_rank}, ...] (RRF-sorted)."""
        body_bm25 = {
            "size": n,
            "_source": False,
            "include_named_queries_score": True,
            "query": _bm25_query(query),
        }
        bm25_resp = self.os.search(index=self.index, body=body_bm25)
        bm25_hits = bm25_resp["hits"]["hits"]

        vec = self._embed(query)
        body_knn = {
            "size": n,
            "_source": False,
            "query": {"knn": {"title_vec": {"vector": vec, "k": n}}},
        }
        knn_resp = self.os.search(index=self.index, body=body_knn)
        knn_hits = knn_resp["hits"]["hits"]

        merged: dict[str, dict] = {}
        for i, h in enumerate(bm25_hits):
            pid = h["_id"]
            d = merged.setdefault(pid, {"product_id": pid, "bm25": 0.0, "knn": 0.0, "rrf": 0.0, "bm25_rank": 0, "knn_rank": 0})
            d["bm25"] = h["_score"]
            d["bm25_rank"] = i + 1
            d["rrf"] += 1.0 / (60 + i + 1)
            matched = _matched_query_scores(h.get("matched_queries"))
            d["title_bm25"] = matched.get("bm25_title", 0.0)
            d["category_path_bm25"] = matched.get("bm25_category_path", 0.0)
            d["category_bm25"] = matched.get("bm25_category", 0.0)
            d["bullets_bm25"] = matched.get("bm25_bullets", 0.0)
            d["description_bm25"] = matched.get("bm25_description", 0.0)
            d["brand_bm25"] = matched.get("bm25_brand", 0.0)
        for i, h in enumerate(knn_hits):
            pid = h["_id"]
            d = merged.setdefault(pid, {"product_id": pid, "bm25": 0.0, "knn": 0.0, "rrf": 0.0, "bm25_rank": 0, "knn_rank": 0})
            d["knn"] = h["_score"]
            d["knn_rank"] = i + 1
            d["rrf"] += 1.0 / (60 + i + 1)

        out = ltr_rows.normalize_retrieval_ranks(merged.values())
        out.sort(key=lambda d: -d["rrf"])
        return out


def _matched_query_scores(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v or 0.0) for k, v in value.items()}
    if isinstance(value, list):
        return {str(k): 1.0 for k in value}
    return {}


# ---------------------------------------------------------------------------
# Feature builder (must match label.build_training_rows for parity)
# ---------------------------------------------------------------------------

def build_feature_matrix(query: str, hits: list[dict], cat: Catalog) -> np.ndarray:
    frame = ltr_rows.build_feature_frame(
        query=query,
        hits=hits,
        products=cat.products,
        brand_tokens=cat.brand_tokens,
        color_tokens=cat.color_tokens,
        category_tokens=cat.category_tokens,
        user_brand_affinity=None,
    )
    X = np.zeros((len(frame), len(FEATURE_NAMES)), dtype=np.float32)
    for i, features in enumerate(frame["features"]):
        for j, name in enumerate(FEATURE_NAMES):
            X[i, j] = float(features.get(name, 0.0) or 0.0)
    return X


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dcg(gains: np.ndarray, k: int) -> float:
    gains = gains[:k]
    if gains.size == 0:
        return 0.0
    return float((gains / np.log2(np.arange(2, gains.size + 2))).sum())


def ndcg(gains_in_pred_order: np.ndarray, gains_in_ideal_order: np.ndarray, k: int) -> float:
    idcg_v = dcg(gains_in_ideal_order, k)
    if idcg_v == 0.0:
        return 0.0
    return dcg(gains_in_pred_order, k) / idcg_v


def mrr(gains_in_pred_order: np.ndarray) -> float:
    for i, g in enumerate(gains_in_pred_order):
        if g >= 2.0:  # C, S, or E counts as a "relevant" hit
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(judgments_for_query: dict[str, str], hit_ids: list[str], k: int) -> float:
    relevant = {pid for pid, lbl in judgments_for_query.items() if ESCI_GAIN[lbl] >= 2.0}
    if not relevant:
        return 0.0
    seen = set(hit_ids[:k]) & relevant
    return len(seen) / len(relevant)


class MetricAccumulator:
    def __init__(self) -> None:
        self.total_queries = 0
        self.zero_results = 0
        self.rrf: dict[str, list[float]] = {
            "ndcg_at_5": [],
            "ndcg_at_10": [],
            "mrr": [],
            "recall_at_10": [],
            "recall_at_50": [],
            "recall_at_100": [],
            "candidate_recall": [],
        }
        self.ltr: dict[str, list[float]] = {
            key: [] for key in self.rrf
        }

    def add_zero_result(self) -> None:
        self.total_queries += 1
        self.zero_results += 1

    def add_rrf(
        self,
        judgments_for_query: dict[str, str],
        hit_ids: list[str],
        gains: np.ndarray | list[float],
        ideal_gains: np.ndarray | list[float],
    ) -> None:
        self.total_queries += 1
        gains_arr = np.asarray(gains, dtype=np.float64)
        ideal_arr = np.asarray(ideal_gains, dtype=np.float64)
        self._add_variant(self.rrf, judgments_for_query, hit_ids, gains_arr, ideal_arr)

    def add_ltr(
        self,
        judgments_for_query: dict[str, str],
        hit_ids: list[str],
        gains: np.ndarray | list[float],
        ideal_gains: np.ndarray | list[float],
    ) -> None:
        gains_arr = np.asarray(gains, dtype=np.float64)
        ideal_arr = np.asarray(ideal_gains, dtype=np.float64)
        self._add_variant(self.ltr, judgments_for_query, hit_ids, gains_arr, ideal_arr)

    def _add_variant(
        self,
        variant: dict[str, list[float]],
        judgments_for_query: dict[str, str],
        hit_ids: list[str],
        gains: np.ndarray,
        ideal_gains: np.ndarray,
    ) -> None:
        variant["ndcg_at_5"].append(ndcg(gains, ideal_gains, 5))
        variant["ndcg_at_10"].append(ndcg(gains, ideal_gains, 10))
        variant["mrr"].append(mrr(gains))
        variant["recall_at_10"].append(recall_at_k(judgments_for_query, hit_ids, 10))
        variant["recall_at_50"].append(recall_at_k(judgments_for_query, hit_ids, 50))
        variant["recall_at_100"].append(recall_at_k(judgments_for_query, hit_ids, 100))
        variant["candidate_recall"].append(
            recall_at_k(judgments_for_query, hit_ids, len(hit_ids))
        )

    def result(self, *, include_ltr: bool = False) -> dict:
        def mean(xs: list[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        out = {
            "queries_evaluated": self.total_queries,
            "queries_with_results": len(self.rrf["ndcg_at_10"]),
            "zero_results": self.zero_results,
            "zero_result_rate": (
                self.zero_results / self.total_queries if self.total_queries else 0.0
            ),
            "rrf": {key: mean(values) for key, values in self.rrf.items()},
        }
        if include_ltr:
            out["ltr"] = {key: mean(values) for key, values in self.ltr.items()}
        return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _import_xgboost():
    try:
        import xgboost as xgb  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "LTR_MODEL_BACKEND=xgboost requires the xgboost package. "
            "Rebuild the pipelines image or install the pipeline dependencies."
        ) from exc
    return xgb


def _artifact_path_for_backend(model_backend: str) -> str:
    if model_backend == "lgbm":
        return "model_text/model.txt"
    if model_backend == "xgboost":
        return "model_xgboost/model.json"
    raise ValueError(f"unsupported LTR model backend: {model_backend}")


def _version_backend(client: mlflow.tracking.MlflowClient, version) -> str:
    run = client.get_run(version.run_id)
    return normalize_ltr_model_backend(run.data.params.get("ltr_model_backend", "lgbm"))


def _load_latest_ltr_model():
    cfg = load()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{LTR_MODEL_NAME}'")
    versions = [v for v in versions if _version_backend(client, v) == cfg.ltr_model_backend]
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    log.info(
        "loading LTR model name=%s version=%s backend=%s",
        LTR_MODEL_NAME, latest.version, cfg.ltr_model_backend,
    )
    run_id = latest.run_id
    local = client.download_artifacts(run_id, _artifact_path_for_backend(cfg.ltr_model_backend))
    if cfg.ltr_model_backend == "lgbm":
        return lgb.Booster(model_file=local)
    xgb = _import_xgboost()
    booster = xgb.Booster()
    booster.load_model(local)
    return booster


def _predict_ltr(booster, X: np.ndarray) -> np.ndarray:
    if LTR_MODEL_BACKEND == "lgbm":
        return booster.predict(X, num_iteration=booster.best_iteration)
    if LTR_MODEL_BACKEND == "xgboost":
        xgb = _import_xgboost()
        kwargs = {}
        best_iteration = getattr(booster, "best_iteration", None)
        if best_iteration is not None:
            kwargs["iteration_range"] = (0, best_iteration + 1)
        return booster.predict(xgb.DMatrix(X), **kwargs)
    raise ValueError(f"unsupported LTR model backend: {LTR_MODEL_BACKEND}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    test_queries = _load_test_queries(EVAL_QUERIES)
    if not test_queries:
        log.error("no eligible ESCI test queries; check ingest")
        return 1
    log.info("evaluating on %d ESCI test queries (cand_n=%d, k=%d)",
             len(test_queries), CAND_N, EVAL_K)

    judgments = _load_judgments([qid for qid, _ in test_queries])
    cat = _load_catalog()

    booster = _load_latest_ltr_model()
    if booster is None:
        log.warning("no LTR model registered; only RRF baseline will be reported")

    retriever = HybridRetriever()
    try:
        metrics = MetricAccumulator()

        started = time.perf_counter()
        for i, (qid, qtext) in enumerate(test_queries, 1):
            jset = judgments.get(qid, {})
            if not jset:
                continue
            hits = retriever.search(qtext, CAND_N)
            if not hits:
                metrics.add_zero_result()
                continue

            ids_rrf = [h["product_id"] for h in hits]
            gains_all = np.array(
                [ESCI_GAIN.get(jset.get(pid, "I"), 0.0) for pid in ids_rrf],
                dtype=np.float64,
            )

            # Ideal: only judged products contribute (unjudged = 0)
            ideal_gains = np.sort(np.array(list(ESCI_GAIN.get(l, 0.0) for l in jset.values())))[::-1]

            metrics.add_rrf(jset, ids_rrf, gains_all, ideal_gains)

            if booster is not None:
                X = build_feature_matrix(qtext, hits, cat)
                preds = _predict_ltr(booster, X)
                order = np.argsort(-preds)
                ids_ltr = [hits[k]["product_id"] for k in order]
                gains_ltr = np.array(
                    [ESCI_GAIN.get(jset.get(pid, "I"), 0.0) for pid in ids_ltr],
                    dtype=np.float64,
                )
                metrics.add_ltr(jset, ids_ltr, gains_ltr, ideal_gains)

            if i % 50 == 0:
                partial = metrics.result(include_ltr=booster is not None)
                log.info(
                    "progress %d/%d  RRF NDCG@10=%.4f  LTR NDCG@10=%.4f",
                    i, len(test_queries),
                    partial["rrf"]["ndcg_at_10"],
                    partial.get("ltr", {}).get("ndcg_at_10", 0.0),
                )

        elapsed = time.perf_counter() - started
        result = metrics.result(include_ltr=booster is not None)
        log.info(
            "done in %.1fs evaluated_queries=%d zero_results=%d zero_result_rate=%.4f",
            elapsed,
            result["queries_evaluated"],
            result["zero_results"],
            result["zero_result_rate"],
        )
        if booster is not None:
            lift = result["ltr"]["ndcg_at_10"] - result["rrf"]["ndcg_at_10"]
            result["ltr_lift_ndcg_at_10"] = lift
            log.info("=" * 60)
            log.info("RRF  NDCG@10=%.4f NDCG@5=%.4f MRR=%.4f Recall@10=%.4f Recall@50=%.4f Recall@100=%.4f CandRecall=%.4f",
                     result["rrf"]["ndcg_at_10"], result["rrf"]["ndcg_at_5"], result["rrf"]["mrr"], result["rrf"]["recall_at_10"], result["rrf"]["recall_at_50"], result["rrf"]["recall_at_100"], result["rrf"]["candidate_recall"])
            log.info("LTR  NDCG@10=%.4f NDCG@5=%.4f MRR=%.4f Recall@10=%.4f Recall@50=%.4f Recall@100=%.4f CandRecall=%.4f",
                     result["ltr"]["ndcg_at_10"], result["ltr"]["ndcg_at_5"], result["ltr"]["mrr"], result["ltr"]["recall_at_10"], result["ltr"]["recall_at_50"], result["ltr"]["recall_at_100"], result["ltr"]["candidate_recall"])
            log.info("LTR vs RRF NDCG@10 lift = %+.4f", lift)
        else:
            log.info("RRF  NDCG@10=%.4f NDCG@5=%.4f MRR=%.4f Recall@10=%.4f Recall@50=%.4f Recall@100=%.4f CandRecall=%.4f",
                     result["rrf"]["ndcg_at_10"], result["rrf"]["ndcg_at_5"], result["rrf"]["mrr"], result["rrf"]["recall_at_10"], result["rrf"]["recall_at_50"], result["rrf"]["recall_at_100"], result["rrf"]["candidate_recall"])

        # Log to MLflow as a standalone evaluation run, attached to the same experiment
        try:
            cfg = load()
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            mlflow.set_experiment("drusearch-eval")
            with mlflow.start_run(run_name=f"eval-{int(time.time())}"):
                mlflow.log_params({
                    "queries_evaluated": result["queries_evaluated"],
                    "queries_with_results": result["queries_with_results"],
                    "zero_results": result["zero_results"],
                    "cand_n": CAND_N,
                    "k": EVAL_K,
                    "ltr_model_name": LTR_MODEL_NAME if booster is not None else "none",
                    "ltr_model_backend": LTR_MODEL_BACKEND if booster is not None else "none",
                })
                mlflow.log_metric("zero_result_rate", result["zero_result_rate"])
                for variant in ("rrf", "ltr"):
                    if variant in result:
                        for k, v in result[variant].items():
                            mlflow.log_metric(f"{variant}_{k}", v)
                if "ltr_lift_ndcg_at_10" in result:
                    mlflow.log_metric("ltr_lift_ndcg_at_10", result["ltr_lift_ndcg_at_10"])
        except Exception as exc:  # pragma: no cover - best-effort
            log.warning("failed to log eval to MLflow: %s", exc)
    finally:
        retriever.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
