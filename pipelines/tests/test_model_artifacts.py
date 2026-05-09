from __future__ import annotations

from types import SimpleNamespace

from pipelines.features import _generated as feature_schema
from pipelines.register.promote import _metadata_for_version


def test_promoted_model_metadata_records_feature_schema_version():
    version = SimpleNamespace(
        name="ltr_reranker",
        version="7",
        current_stage="Production",
        run_id="abc123",
        source="s3://mlflow-artifacts/abc123/artifacts/model",
    )

    assert _metadata_for_version(version)["feature_schema_version"] == feature_schema.SCHEMA_VERSION


def test_promoted_model_metadata_records_backend():
    version = SimpleNamespace(
        name="ltr_reranker",
        version="7",
        current_stage="Production",
        run_id="abc123",
        source="s3://mlflow-artifacts/abc123/artifacts/model",
    )

    assert _metadata_for_version(version, model_backend="lgbm")["model_backend"] == "lgbm"
