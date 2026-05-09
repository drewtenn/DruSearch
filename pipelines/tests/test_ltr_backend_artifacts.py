from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipelines.evaluate.offline_eval import _version_backend
from pipelines.register import promote
from pipelines.train import lgbm_ranker


@pytest.mark.parametrize(
    ("backend", "want_path"),
    [
        ("lgbm", "model_text/model.txt"),
        ("xgboost", "model_xgboost/model.json"),
    ],
)
def test_artifact_path_for_backend(backend, want_path):
    assert lgbm_ranker._artifact_path_for_backend(backend) == want_path


@pytest.mark.parametrize(
    ("backend", "want_name"),
    [
        ("lgbm", "ltr_reranker.txt"),
        ("xgboost", "ltr_reranker.xgb.json"),
    ],
)
def test_promote_target_file_for_backend(monkeypatch, tmp_path, backend, want_name):
    monkeypatch.setattr(promote, "MODEL_DIR", tmp_path)

    assert promote._target_file_for_backend(backend).name == want_name


def test_artifact_path_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported LTR model backend"):
        lgbm_ranker._artifact_path_for_backend("catboost")


def test_sorted_query_ids_are_numeric_for_xgboost_ranker():
    df = pd.DataFrame(
        {
            "query_id": ["b-query", "a-query", "b-query", "c-query", "a-query"],
        }
    )

    qids = lgbm_ranker._sorted_query_ids(df)

    assert qids.dtype == np.uint32
    assert qids.tolist() == [0, 0, 1, 1, 2]


def test_build_matrix_returns_sample_weights():
    df = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "features": [
                {"bm25_score": 1.0},
                {"bm25_score": 2.0},
            ],
            "label": [4, 0],
            "sample_weight": [1.0, 0.25],
        }
    )

    X, y, groups, weights = lgbm_ranker._build_matrix(df)

    assert X.shape == (2, len(lgbm_ranker.FEATURE_NAMES))
    assert y.tolist() == [4, 0]
    assert groups.tolist() == [2]
    assert weights.tolist() == [1.0, 0.25]


def test_sort_by_query_preserves_candidate_order_within_query():
    df = pd.DataFrame(
        {
            "query_id": ["q2", "q1", "q1", "q2"],
            "product_id": ["p4", "p1", "p2", "p3"],
        }
    )

    got = lgbm_ranker._sort_by_query(df)

    assert got["product_id"].tolist() == ["p1", "p2", "p4", "p3"]


def test_xgboost_ranking_weights_are_collapsed_to_query_groups():
    df = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
        }
    )
    row_weights = np.array([0.25, 0.75, 1.0, 0.5], dtype=np.float32)

    group_weights = lgbm_ranker._query_group_weights(df, row_weights)

    assert group_weights.tolist() == [0.5, 0.75]


def test_rrf_baseline_diagnostic_uses_training_row_candidates():
    df = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "features": [
                {"rrf_score": 0.3},
                {"rrf_score": 0.2},
                {"rrf_score": 0.1},
            ],
            "label": [0, 4, 0],
            "sample_weight": [1.0, 1.0, 1.0],
        }
    )

    X, y, _groups, _weights = lgbm_ranker._build_matrix(df)
    rrf = lgbm_ranker._feature_column(X, "rrf_score")

    assert lgbm_ranker._ndcg_at_k(df["query_id"].to_numpy(), y, rrf, k=10) < 1.0


def test_offline_diagnostics_use_all_judgments_for_ideal_and_recall():
    df = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "product_id": ["retrieved-bad", "retrieved-good"],
            "features": [
                {"rrf_score": 0.2},
                {"rrf_score": 0.1},
            ],
            "label": [0, 4],
            "sample_weight": [1.0, 1.0],
        }
    )
    X, y, _groups, _weights = lgbm_ranker._build_matrix(df)

    metrics = lgbm_ranker._offline_eval_metrics(
        df.sort_values("query_id").reset_index(drop=True),
        y,
        ltr_preds=np.array([0.2, 0.1], dtype=np.float32),
        rrf_preds=lgbm_ranker._feature_column(X, "rrf_score"),
        judgments_by_query={"q1": {"retrieved-good": "E", "missing-better": "E"}},
        k=10,
    )

    assert metrics["rrf"]["ndcg_at_10"] < lgbm_ranker._ndcg_at_k(
        df["query_id"].to_numpy(), y, lgbm_ranker._feature_column(X, "rrf_score"), k=10
    )
    assert metrics["rrf"]["recall_at_10"] == 0.5


def test_offline_eval_test_rows_require_current_candidate_training_rows():
    df = pd.DataFrame(
        {
            "query_id": ["event-hash", "another-event-hash"],
            "product_id": ["p1", "p2"],
        }
    )

    got = lgbm_ranker._offline_eval_test_rows(df, ["100", "267"])

    assert got.empty


class _LoadRowsCursor:
    description = []

    def __init__(self):
        self.statements: list[tuple[str, object | None]] = []
        self._last = ""

    def execute(self, sql, params=None):
        self._last = " ".join(str(sql).split())
        self.statements.append((self._last, params))

    def fetchone(self):
        if "FROM training_row_builds" in self._last:
            return (
                42,
                "offline_candidates",
                "v6",
                200,
                False,
                0.25,
                "ready",
                2,
                1,
                {},
            )
        return None

    def fetchall(self):
        if "FROM training_rows" in self._last:
            return [
                ("q1", "p1", {"bm25_score": 1.0}, 4.0, "train", 1.0),
                ("q1", "p2", {"bm25_score": 0.5}, 0.0, "train", 1.0),
            ]
        return []


class _LoadRowsConnection:
    def __init__(self):
        self.cursor_obj = _LoadRowsCursor()

    @contextmanager
    def cursor(self):
        yield self.cursor_obj


def test_load_training_rows_reads_only_latest_ready_generation(monkeypatch):
    conn = _LoadRowsConnection()

    @contextmanager
    def fake_conn():
        yield conn

    monkeypatch.setattr(lgbm_ranker.db, "conn", fake_conn)

    df, build = lgbm_ranker._load_training_rows()

    row_select = [
        (sql, params)
        for sql, params in conn.cursor_obj.statements
        if "FROM training_rows" in sql
    ][0]
    assert build is not None
    assert build.build_id == 42
    assert "WHERE build_id = %s" in row_select[0]
    assert row_select[1] == (42,)
    assert df["product_id"].tolist() == ["p1", "p2"]


def test_version_backend_reads_logged_mlflow_param():
    client = SimpleNamespace(
        get_run=lambda _run_id: SimpleNamespace(
            data=SimpleNamespace(params={"ltr_model_backend": "xgb"})
        )
    )
    version = SimpleNamespace(run_id="abc123")

    assert _version_backend(client, version) == "xgboost"


def test_version_backend_defaults_old_runs_to_lgbm():
    client = SimpleNamespace(
        get_run=lambda _run_id: SimpleNamespace(data=SimpleNamespace(params={}))
    )
    version = SimpleNamespace(run_id="abc123")

    assert _version_backend(client, version) == "lgbm"
