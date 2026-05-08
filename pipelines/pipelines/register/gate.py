"""Promotion gate: only transition the latest model to Production stage if
its ESCI NDCG@10 is non-regressive vs the current Production version.

Default tolerance: 0.0 (strict non-regression). Override with
PROMOTE_TOL_NDCG=0.01 to permit a small drop.

Run: docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.register.gate
"""

from __future__ import annotations

import os
import sys

import lightgbm as lgb
import mlflow
import numpy as np

from mlflow.tracking import MlflowClient
from pipelines.common.config import load
from pipelines.common.logging import configure
from pipelines.evaluate.offline_eval import (
    HybridRetriever,
    _load_catalog,
    _load_judgments,
    _load_test_queries,
    build_feature_matrix,
    ndcg,
    ESCI_GAIN,
)

log = configure("register.gate")

LTR_MODEL_NAME = os.getenv("LTR_MODEL_NAME", "ltr_reranker")
TOL_NDCG = float(os.getenv("PROMOTE_TOL_NDCG", "0.0"))
EVAL_QUERIES = int(os.getenv("EVAL_QUERIES", "500"))
EVAL_K = int(os.getenv("EVAL_K", "10"))
CAND_N = int(os.getenv("EVAL_CAND_N", "200"))


def _load_model_for(version) -> lgb.Booster:
    client = MlflowClient()
    local = client.download_artifacts(version.run_id, "model_text/model.txt")
    return lgb.Booster(model_file=local)


def _ndcg_at_k_for_model(booster: lgb.Booster, queries, judgments, retriever, cat) -> float:
    scores: list[float] = []
    for qid, qtext in queries:
        jset = judgments.get(qid, {})
        if not jset:
            continue
        hits = retriever.search(qtext, CAND_N)
        if not hits:
            continue
        X = build_feature_matrix(qtext, hits, cat)
        preds = booster.predict(X, num_iteration=booster.best_iteration)
        order = np.argsort(-preds)
        gains = np.array([ESCI_GAIN.get(jset.get(hits[k]["product_id"], "I"), 0.0) for k in order], dtype=np.float64)
        ideal = np.sort(np.array([ESCI_GAIN.get(l, 0.0) for l in jset.values()]))[::-1]
        scores.append(ndcg(gains, ideal, EVAL_K))
    return float(np.mean(scores)) if scores else 0.0


def main() -> int:
    cfg = load()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{LTR_MODEL_NAME}'")
    if not versions:
        log.error("no versions registered for %s", LTR_MODEL_NAME)
        return 1
    versions.sort(key=lambda v: int(v.version))
    candidate = versions[-1]
    prod = next((v for v in versions if (v.current_stage or "") == "Production"), None)

    log.info(
        "candidate name=%s v=%s stage=%s; current production v=%s",
        candidate.name, candidate.version, candidate.current_stage,
        prod.version if prod else "(none)",
    )

    if prod and prod.version == candidate.version:
        log.info("candidate is already in Production; nothing to do")
        return 0

    test_queries = _load_test_queries(EVAL_QUERIES)
    judgments = _load_judgments([qid for qid, _ in test_queries])
    cat = _load_catalog()
    retriever = HybridRetriever()
    try:
        cand_booster = _load_model_for(candidate)
        cand_score = _ndcg_at_k_for_model(cand_booster, test_queries, judgments, retriever, cat)
        log.info("candidate v=%s ESCI NDCG@10=%.4f", candidate.version, cand_score)

        prod_score = 0.0
        if prod:
            prod_booster = _load_model_for(prod)
            prod_score = _ndcg_at_k_for_model(prod_booster, test_queries, judgments, retriever, cat)
            log.info("production v=%s ESCI NDCG@10=%.4f", prod.version, prod_score)
        else:
            log.info("no current production; any non-zero candidate is acceptable")
    finally:
        retriever.close()

    delta = cand_score - prod_score
    if delta < -TOL_NDCG:
        log.error(
            "GATE FAILED: NDCG@10 regression %.4f -> %.4f (delta=%+.4f, tol=%.4f); "
            "NOT transitioning v=%s to Production",
            prod_score, cand_score, delta, TOL_NDCG, candidate.version,
        )
        return 2
    log.info(
        "GATE PASSED: NDCG@10 %.4f -> %.4f (delta=%+.4f); transitioning v=%s to Production",
        prod_score, cand_score, delta, candidate.version,
    )
    client.transition_model_version_stage(
        name=LTR_MODEL_NAME,
        version=candidate.version,
        stage="Production",
        archive_existing_versions=True,
    )
    log.info("v=%s now in Production", candidate.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
