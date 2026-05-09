"""Train an LTR ranker on training_rows and log it to MLflow.

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
from pipelines.common import training_row_builds
from pipelines.common.config import load
from pipelines.common.logging import configure
from pipelines.features import FEATURE_NAMES
from pipelines.features._generated import SCHEMA_VERSION

log = configure("train.lgbm_ranker")

EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "drusearch-ltr")
MODEL_NAME = os.getenv("LTR_MODEL_NAME", "ltr_reranker")
NUM_BOOST_ROUND = int(os.getenv("LGBM_BOOST_ROUNDS", "1000"))
EARLY_STOP = int(os.getenv("LGBM_EARLY_STOP", "50"))
XGBOOST_TREE_METHOD = os.getenv("XGBOOST_TREE_METHOD", "hist")
EVAL_QUERIES = int(os.getenv("EVAL_QUERIES", "500"))
EVAL_K = int(os.getenv("EVAL_K", "10"))
EVAL_MIN_JUDGMENTS = int(os.getenv("EVAL_MIN_JUDGMENTS", "5"))
EXPECTED_TRAINING_ROW_SOURCE = os.getenv("TRAINING_ROW_SOURCE", "offline_candidates")
EXPECTED_LTR_CAND_N = int(os.getenv("LTR_CAND_N", os.getenv("EVAL_CAND_N", "200")))
EXPECTED_PSEUDO_LABELS_ENABLED = os.getenv("LTR_PSEUDO_LABELS", "0").lower() in {
    "1",
    "true",
    "yes",
}
EXPECTED_PSEUDO_LABEL_WEIGHT = float(os.getenv("LTR_PSEUDO_LABEL_WEIGHT", "0.25"))

_CLASSIC_ESCI_GAIN = {"E": 4.0, "S": 3.0, "C": 2.0, "I": 0.0}


def _artifact_path_for_backend(model_backend: str) -> str:
    if model_backend == "lgbm":
        return "model_text/model.txt"
    if model_backend == "xgboost":
        return "model_xgboost/model.json"
    raise ValueError(f"unsupported LTR model backend: {model_backend}")


def _load_training_rows() -> tuple[pd.DataFrame, training_row_builds.TrainingRowBuild | None]:
    log.info("loading training_rows")
    with db.conn() as c, c.cursor() as cur:
        training_row_builds.ensure_schema(cur)
        build = training_row_builds.latest_ready_training_row_build(cur)
        if build is None:
            rows = []
        else:
            cur.execute(
                """
                SELECT query_id, product_id, features, label, split, sample_weight
                FROM training_rows
                WHERE build_id = %s
                """,
                (build.build_id,),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(
        rows,
        columns=["query_id", "product_id", "features", "label", "split", "sample_weight"],
    )
    build_id = build.build_id if build is not None else None
    log.info("loaded rows=%d build_id=%s", len(df), build_id)
    return df, build


def _sort_by_query(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("query_id", kind="mergesort").reset_index(drop=True)


def _build_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y, group_sizes, weights. Rows are ordered by query_id."""
    df = _sort_by_query(df)
    feats = df["features"].apply(
        lambda d: d if isinstance(d, dict) else json.loads(d)
    )
    X = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    for i, fmap in enumerate(feats):
        for j, name in enumerate(FEATURE_NAMES):
            X[i, j] = float(fmap.get(name, 0.0) or 0.0)
    y = df["label"].astype(int).to_numpy()
    group_sizes = df.groupby("query_id", sort=False).size().to_numpy()
    if "sample_weight" in df.columns:
        weights = df["sample_weight"].astype(float).to_numpy(dtype=np.float32)
    else:
        weights = np.ones(len(df), dtype=np.float32)
    return X, y, group_sizes, weights


def _feature_column(X: np.ndarray, name: str) -> np.ndarray:
    return X[:, list(FEATURE_NAMES).index(name)]


def _query_group_weights(df: pd.DataFrame, row_weights: np.ndarray) -> np.ndarray:
    """Return one XGBoost ranking weight per query group.

    The dataframe and row_weights must already be aligned in training order.
    """
    ordered = df.reset_index(drop=True).copy()
    ordered["_row_weight"] = row_weights
    return (
        ordered.groupby("query_id", sort=False)["_row_weight"]
        .mean()
        .to_numpy(dtype=np.float32)
    )


def _sorted_query_ids(df: pd.DataFrame) -> np.ndarray:
    qids = _sort_by_query(df)["query_id"]
    codes, _uniques = pd.factorize(qids, sort=False)
    return codes.astype(np.uint32, copy=False)


def _load_offline_eval_test_query_ids(n: int) -> list[str]:
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT j.query_id
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
        return [str(qid) for (qid,) in cur.fetchall()]


def _load_esci_judgments_by_query(query_ids: list[str]) -> dict[str, dict[str, str]]:
    if not query_ids:
        return {}
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT query_id::TEXT, product_id, esci_label
            FROM esci_judgments
            WHERE query_id::TEXT = ANY(%s)
            """,
            (query_ids,),
        )
        out: dict[str, dict[str, str]] = {}
        for qid, pid, label in cur.fetchall():
            out.setdefault(str(qid), {})[str(pid)] = str(label)
        return out


def _offline_eval_test_rows(test_df: pd.DataFrame, eval_query_ids: list[str]) -> pd.DataFrame:
    return test_df[test_df["query_id"].astype(str).isin(eval_query_ids)].copy()


def _train_lgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    g_tr: np.ndarray,
    w_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    g_va: np.ndarray,
    w_va: np.ndarray,
) -> tuple[lgb.Booster, dict, dict]:
    train_set = lgb.Dataset(X_tr, label=y_tr, group=g_tr, weight=w_tr, feature_name=list(FEATURE_NAMES))
    val_set = lgb.Dataset(
        X_va,
        label=y_va,
        group=g_va,
        weight=w_va,
        feature_name=list(FEATURE_NAMES),
        reference=train_set,
    )

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
    evals_result: dict = {}
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
    return booster, params, evals_result


def _import_xgboost():
    try:
        import mlflow.xgboost  # type: ignore[import-untyped]
        import xgboost as xgb  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "LTR_MODEL_BACKEND=xgboost requires the xgboost package. "
            "Rebuild the pipelines image or install the pipeline dependencies."
        ) from exc
    return xgb, mlflow.xgboost


def _train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    w_va: np.ndarray,
) -> tuple[object, dict, dict]:
    xgb, _mlflow_xgboost = _import_xgboost()
    params = {
        "objective": "rank:ndcg",
        "eval_metric": ["ndcg@5", "ndcg@10"],
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 100,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "tree_method": XGBOOST_TREE_METHOD,
        "random_state": 42,
        "n_estimators": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOP,
    }
    ranker = xgb.XGBRanker(**params)
    eval_set = [(X_tr, y_tr), (X_va, y_va)]
    eval_qid = [_sorted_query_ids(train_df), _sorted_query_ids(val_df)]
    train_group_weight = _query_group_weights(train_df, w_tr)
    eval_group_weight = [
        _query_group_weights(train_df, w_tr),
        _query_group_weights(val_df, w_va),
    ]
    ranker.fit(
        X_tr,
        y_tr,
        qid=_sorted_query_ids(train_df),
        sample_weight=train_group_weight,
        eval_set=eval_set,
        eval_qid=eval_qid,
        sample_weight_eval_set=eval_group_weight,
        verbose=25,
    )
    return ranker, params, ranker.evals_result()


def _predict(model_backend: str, booster: object, X: np.ndarray) -> np.ndarray:
    if model_backend == "lgbm":
        return booster.predict(X, num_iteration=booster.best_iteration)  # type: ignore[attr-defined]
    if model_backend == "xgboost":
        xgb, _mlflow_xgboost = _import_xgboost()
        dmatrix = xgb.DMatrix(X)
        best_iteration = getattr(booster, "best_iteration", None)
        kwargs = {}
        if best_iteration is not None:
            kwargs["iteration_range"] = (0, best_iteration + 1)
        return booster.predict(dmatrix, **kwargs)
    raise ValueError(f"unsupported LTR model backend: {model_backend}")


def main() -> int:
    cfg = load()
    model_backend = cfg.ltr_model_backend
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    df, build = _load_training_rows()
    build_errors = training_row_builds.validate_ready_build(
        build,
        expected_source=EXPECTED_TRAINING_ROW_SOURCE,
        expected_feature_schema_version=SCHEMA_VERSION,
        expected_cand_n=EXPECTED_LTR_CAND_N,
        expected_pseudo_labels_enabled=EXPECTED_PSEUDO_LABELS_ENABLED,
        expected_pseudo_label_weight=EXPECTED_PSEUDO_LABEL_WEIGHT,
        actual_row_count=int(len(df)),
        actual_query_count=int(df["query_id"].nunique()) if "query_id" in df else 0,
    )
    if build_errors:
        log.error(
            "training_rows are not from the expected ready build; rerun "
            "pipelines.label.build_training_rows. errors=%s",
            "; ".join(build_errors),
        )
        return 1
    if df.empty:
        log.error("training_rows is empty; run pipelines.label.build_training_rows first")
        return 1

    train_df = _sort_by_query(df[df["split"] == "train"])
    val_df   = _sort_by_query(df[df["split"] == "val"])
    test_df  = _sort_by_query(df[df["split"] == "test"])
    log.info(
        "splits: train=%d val=%d test=%d   uniq_queries: train=%d val=%d test=%d",
        len(train_df), len(val_df), len(test_df),
        train_df["query_id"].nunique(), val_df["query_id"].nunique(), test_df["query_id"].nunique(),
    )
    eval_query_ids = _load_offline_eval_test_query_ids(EVAL_QUERIES)
    eval_test_df = _offline_eval_test_rows(test_df, eval_query_ids)
    if eval_test_df.empty:
        log.error(
            "training_rows contain no offline-eval eligible test queries; "
            "rerun pipelines.label.build_training_rows before training"
        )
        return 1
    eval_test_df = _sort_by_query(eval_test_df)
    judgments_by_query = _load_esci_judgments_by_query(eval_query_ids)

    X_tr, y_tr, g_tr, w_tr = _build_matrix(train_df)
    X_va, y_va, g_va, w_va = _build_matrix(val_df)

    log.info("starting MLflow run experiment=%s backend=%s", EXPERIMENT, model_backend)
    with mlflow.start_run() as run:
        mlflow.log_params({
            "ltr_model_backend": model_backend,
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
            "min_sample_weight": float(df["sample_weight"].min()),
            "mean_sample_weight": float(df["sample_weight"].mean()),
            "training_row_build_id": build.build_id if build is not None else "",
            "training_row_source": build.source if build is not None else "",
            "training_row_feature_schema_version": (
                build.feature_schema_version if build is not None else ""
            ),
            "training_row_cand_n": build.cand_n if build is not None else "",
            "training_row_pseudo_labels_enabled": (
                build.pseudo_labels_enabled if build is not None else ""
            ),
            "training_row_pseudo_label_weight": (
                build.pseudo_label_weight if build is not None else ""
            ),
        })

        started = time.perf_counter()
        if model_backend == "lgbm":
            booster, params, evals_result = _train_lgbm(X_tr, y_tr, g_tr, w_tr, X_va, y_va, g_va, w_va)
        elif model_backend == "xgboost":
            ranker, params, evals_result = _train_xgboost(train_df, val_df, X_tr, y_tr, w_tr, X_va, y_va, w_va)
            booster = ranker.get_booster()
        else:
            raise ValueError(f"unsupported LTR model backend: {model_backend}")
        mlflow.log_params(params)
        elapsed = time.perf_counter() - started
        best_iter = getattr(booster, "best_iteration", None)
        log.info("trained backend=%s best_iter=%s in %.1fs", model_backend, best_iter, elapsed)

        # Log curves; sanitize metric names ('@' is not allowed in MLflow names).
        for split_name, metrics in evals_result.items():
            for metric, values in metrics.items():
                safe = metric.replace("@", "_at_")
                for step, v in enumerate(values):
                    mlflow.log_metric(f"{split_name}_{safe}", v, step=step)

        # Final test metrics mirror pipelines.evaluate.offline_eval: same query
        # eligibility, classic ESCI gains, and ideal ranking from all judgments.
        if len(eval_test_df) > 0:
            X_te, y_te, _g_te, _w_te = _build_matrix(eval_test_df)
            preds = _predict(model_backend, booster, X_te)
            rrf_preds = _feature_column(X_te, "rrf_score")
            metrics = _offline_eval_metrics(
                eval_test_df,
                y_te,
                ltr_preds=preds,
                rrf_preds=rrf_preds,
                judgments_by_query=judgments_by_query,
                k=EVAL_K,
            )
            ltr_lift = metrics["ltr"]["ndcg_at_10"] - metrics["rrf"]["ndcg_at_10"]
            log.info(
                "offline-style test queries=%d rows=%d",
                eval_test_df["query_id"].nunique(),
                len(eval_test_df),
            )
            log.info(
                "test RRF NDCG@5=%.4f NDCG@10=%.4f MRR=%.4f Recall@10=%.4f",
                metrics["rrf"]["ndcg_at_5"],
                metrics["rrf"]["ndcg_at_10"],
                metrics["rrf"]["mrr"],
                metrics["rrf"]["recall_at_10"],
            )
            log.info(
                "test LTR NDCG@5=%.4f NDCG@10=%.4f MRR=%.4f Recall@10=%.4f  LTR lift NDCG@10=%+.4f",
                metrics["ltr"]["ndcg_at_5"],
                metrics["ltr"]["ndcg_at_10"],
                metrics["ltr"]["mrr"],
                metrics["ltr"]["recall_at_10"],
                ltr_lift,
            )
            mlflow.log_metric("test_ndcg_at_5", metrics["ltr"]["ndcg_at_5"])
            mlflow.log_metric("test_ndcg_at_10", metrics["ltr"]["ndcg_at_10"])
            mlflow.log_metric("test_mrr", metrics["ltr"]["mrr"])
            mlflow.log_metric("test_recall_at_10", metrics["ltr"]["recall_at_10"])
            mlflow.log_metric("test_rrf_ndcg_at_5", metrics["rrf"]["ndcg_at_5"])
            mlflow.log_metric("test_rrf_ndcg_at_10", metrics["rrf"]["ndcg_at_10"])
            mlflow.log_metric("test_rrf_mrr", metrics["rrf"]["mrr"])
            mlflow.log_metric("test_rrf_recall_at_10", metrics["rrf"]["recall_at_10"])
            mlflow.log_metric("test_ltr_lift_ndcg_at_10", ltr_lift)
        else:
            log.warning("no training_rows matched offline eval test query eligibility")

        # Save and log model
        with tempfile.TemporaryDirectory() as tmp:
            if model_backend == "lgbm":
                model_txt = Path(tmp) / "model.txt"
                booster.save_model(str(model_txt))
                mlflow.log_artifact(str(model_txt), artifact_path="model_text")
                mlflow.lightgbm.log_model(
                    lgb_model=booster,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME,
                )
            elif model_backend == "xgboost":
                _xgb, mlflow_xgboost = _import_xgboost()
                model_json = Path(tmp) / "model.json"
                booster.save_model(str(model_json))
                mlflow.log_artifact(str(model_json), artifact_path="model_xgboost")
                mlflow_xgboost.log_model(
                    xgb_model=booster,
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


def _classic_ndcg(gains_in_pred_order: np.ndarray, gains_in_ideal_order: np.ndarray, k: int) -> float:
    idcg = _dcg(gains_in_ideal_order, k)
    if idcg <= 0.0:
        return 0.0
    return _dcg(gains_in_pred_order, k) / idcg


def _classic_mrr(gains_in_pred_order: np.ndarray) -> float:
    for i, gain in enumerate(gains_in_pred_order):
        if gain >= 2.0:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(judgments_for_query: dict[str, str], hit_ids: list[str], k: int) -> float:
    relevant = {
        pid
        for pid, label in judgments_for_query.items()
        if _CLASSIC_ESCI_GAIN.get(label, 0.0) >= 2.0
    }
    if not relevant:
        return 0.0
    return len(set(hit_ids[:k]) & relevant) / len(relevant)


def _offline_eval_metrics(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    ltr_preds: np.ndarray,
    rrf_preds: np.ndarray,
    judgments_by_query: dict[str, dict[str, str]],
    k: int,
) -> dict[str, dict[str, float]]:
    """Compute train-time diagnostics with the same ideal/recall semantics as offline_eval."""
    eval_df = df[["query_id", "product_id"]].copy()
    eval_df["label"] = labels
    eval_df["ltr_pred"] = ltr_preds
    eval_df["rrf_pred"] = rrf_preds

    scores = {
        "rrf": {"ndcg_at_5": [], "ndcg_at_10": [], "mrr": [], "recall_at_10": []},
        "ltr": {"ndcg_at_5": [], "ndcg_at_10": [], "mrr": [], "recall_at_10": []},
    }
    for qid, group in eval_df.groupby("query_id", sort=False):
        jset = judgments_by_query.get(str(qid), {})
        if not jset:
            continue
        ideal = np.sort(
            np.array([_CLASSIC_ESCI_GAIN.get(label, 0.0) for label in jset.values()])
        )[::-1]
        if ideal.size == 0 or ideal[0] <= 0.0:
            continue

        for name, pred_col in (("rrf", "rrf_pred"), ("ltr", "ltr_pred")):
            ordered = group.sort_values(pred_col, ascending=False)
            hit_ids = [str(pid) for pid in ordered["product_id"]]
            gains = np.array(
                [_CLASSIC_ESCI_GAIN.get(jset.get(pid, "I"), 0.0) for pid in hit_ids],
                dtype=np.float64,
            )
            scores[name]["ndcg_at_5"].append(_classic_ndcg(gains, ideal, 5))
            scores[name]["ndcg_at_10"].append(_classic_ndcg(gains, ideal, k))
            scores[name]["mrr"].append(_classic_mrr(gains))
            scores[name]["recall_at_10"].append(_recall_at_k(jset, hit_ids, k))

    return {
        name: {
            metric: float(np.mean(values)) if values else 0.0
            for metric, values in metric_values.items()
        }
        for name, metric_values in scores.items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
