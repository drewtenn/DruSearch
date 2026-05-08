"""Train a LightGBM LambdaRank model on training_rows and log it to MLflow.

Run: docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.train.lgbm_ranker
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from pipelines.common import db
from pipelines.common.config import load
from pipelines.common.logging import configure
from pipelines.features import FEATURE_NAMES

log = configure("train.lgbm_ranker")

EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "drusearch-ltr")
MODEL_NAME = os.getenv("LTR_MODEL_NAME", "ltr_reranker")
NUM_BOOST_ROUND = int(os.getenv("LGBM_BOOST_ROUNDS", "1000"))
EARLY_STOP = int(os.getenv("LGBM_EARLY_STOP", "50"))


def _load_training_rows() -> pd.DataFrame:
    log.info("loading training_rows")
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT query_id, product_id, features, label, split FROM training_rows"
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["query_id", "product_id", "features", "label", "split"])
    log.info("loaded rows=%d", len(df))
    return df


def _build_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y, group_sizes.  Rows are ordered by query_id so groups are contiguous."""
    df = df.sort_values("query_id").reset_index(drop=True)
    feats = df["features"].apply(
        lambda d: d if isinstance(d, dict) else json.loads(d)
    )
    X = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    for i, fmap in enumerate(feats):
        for j, name in enumerate(FEATURE_NAMES):
            X[i, j] = float(fmap.get(name, 0.0) or 0.0)
    y = df["label"].astype(int).to_numpy()
    group_sizes = df.groupby("query_id", sort=False).size().to_numpy()
    return X, y, group_sizes


def main() -> int:
    cfg = load()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    df = _load_training_rows()
    if df.empty:
        log.error("training_rows is empty; run pipelines.label.build_training_rows first")
        return 1

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]
    log.info(
        "splits: train=%d val=%d test=%d   uniq_queries: train=%d val=%d test=%d",
        len(train_df), len(val_df), len(test_df),
        train_df["query_id"].nunique(), val_df["query_id"].nunique(), test_df["query_id"].nunique(),
    )

    X_tr, y_tr, g_tr = _build_matrix(train_df)
    X_va, y_va, g_va = _build_matrix(val_df)
    X_te, y_te, g_te = _build_matrix(test_df)

    train_set = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=list(FEATURE_NAMES))
    val_set   = lgb.Dataset(X_va, label=y_va, group=g_va, feature_name=list(FEATURE_NAMES), reference=train_set)

    # ESCI-derived labels: 0 (unjudged or 'I'), 2 ('C'), 3 ('S'), 4 ('E').
    # label_gain entries map label index i to its gain; values 0..4 are needed.
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "label_gain": [0, 1, 3, 7, 15],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    log.info("starting MLflow run experiment=%s", EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params({
            **params,
            "num_features":   len(FEATURE_NAMES),
            "feature_names":  ",".join(FEATURE_NAMES),
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOP,
            "train_rows": len(train_df),
            "val_rows":   len(val_df),
            "test_rows":  len(test_df),
            "train_queries": train_df["query_id"].nunique(),
            "val_queries":   val_df["query_id"].nunique(),
            "test_queries":  test_df["query_id"].nunique(),
        })

        evals_result: dict = {}
        started = time.perf_counter()
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOP),
                lgb.log_evaluation(period=25),
                lgb.record_evaluation(evals_result),
            ],
        )
        elapsed = time.perf_counter() - started
        log.info("trained best_iter=%d in %.1fs", booster.best_iteration, elapsed)

        # Log curves; sanitize metric names ('@' is not allowed in MLflow names).
        for split_name, metrics in evals_result.items():
            for metric, values in metrics.items():
                safe = metric.replace("@", "_at_")
                for step, v in enumerate(values):
                    mlflow.log_metric(f"{split_name}_{safe}", v, step=step)

        # Final test metrics
        if len(X_te) > 0:
            preds = booster.predict(X_te, num_iteration=booster.best_iteration)
            ndcg10 = _ndcg_at_k(test_df["query_id"].to_numpy(), y_te, preds, k=10)
            ndcg5  = _ndcg_at_k(test_df["query_id"].to_numpy(), y_te, preds, k=5)
            mrr   = _mrr(test_df["query_id"].to_numpy(), y_te, preds)
            log.info("test NDCG@5=%.4f NDCG@10=%.4f MRR=%.4f", ndcg5, ndcg10, mrr)
            mlflow.log_metric("test_ndcg_at_5", ndcg5)
            mlflow.log_metric("test_ndcg_at_10", ndcg10)
            mlflow.log_metric("test_mrr", mrr)

        # Save and log model
        with tempfile.TemporaryDirectory() as tmp:
            model_txt = Path(tmp) / "model.txt"
            booster.save_model(str(model_txt))
            mlflow.log_artifact(str(model_txt), artifact_path="model_text")

            mlflow.lightgbm.log_model(
                lgb_model=booster,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
        log.info("logged model to MLflow run_id=%s name=%s", run.info.run_id, MODEL_NAME)

    return 0


# ---------------------------------------------------------------------------
# Eval helpers (per-query NDCG, MRR using LambdaRank label_gain semantics)
# ---------------------------------------------------------------------------

_LABEL_GAIN = np.array([0.0, 1.0, 3.0, 7.0, 15.0], dtype=np.float64)


def _dcg(rel: np.ndarray, k: int) -> float:
    rel = rel[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float((rel * discounts).sum())


def _ndcg_at_k(query_ids: np.ndarray, labels: np.ndarray, preds: np.ndarray, k: int) -> float:
    df = pd.DataFrame({"q": query_ids, "label": labels, "pred": preds})
    scores: list[float] = []
    for _q, g in df.groupby("q", sort=False):
        gains = _LABEL_GAIN[g["label"].astype(int).to_numpy()]
        order = np.argsort(-g["pred"].to_numpy())
        ideal_order = np.argsort(-gains)
        dcg   = _dcg(gains[order], k)
        idcg  = _dcg(gains[ideal_order], k)
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def _mrr(query_ids: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> float:
    df = pd.DataFrame({"q": query_ids, "label": labels, "pred": preds})
    rrs: list[float] = []
    for _q, g in df.groupby("q", sort=False):
        order = np.argsort(-g["pred"].to_numpy())
        labs  = g["label"].to_numpy()[order]
        # Reciprocal rank of the first positive label (>=1).
        for i, lab in enumerate(labs):
            if lab >= 1:
                rrs.append(1.0 / (i + 1))
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs)) if rrs else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
