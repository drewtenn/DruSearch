from __future__ import annotations

import pytest

from pipelines.common.config import load, normalize_ltr_model_backend


def test_ltr_model_backend_defaults_to_lgbm(monkeypatch):
    monkeypatch.delenv("LTR_MODEL_BACKEND", raising=False)

    assert load().ltr_model_backend == "lgbm"


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("lgbm", "lgbm"),
        ("lightgbm", "lgbm"),
        ("xgb", "xgboost"),
        ("xgboost", "xgboost"),
        (" XGBoost ", "xgboost"),
    ],
)
def test_normalize_ltr_model_backend_accepts_supported_values(raw, want):
    assert normalize_ltr_model_backend(raw) == want


def test_ltr_model_backend_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("LTR_MODEL_BACKEND", "catboost")

    with pytest.raises(ValueError, match="LTR_MODEL_BACKEND"):
        load()
