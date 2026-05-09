from __future__ import annotations

from types import SimpleNamespace

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
