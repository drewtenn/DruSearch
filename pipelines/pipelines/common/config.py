"""Process-wide settings sourced from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv(usecwd=True), override=False)


@dataclass(frozen=True)
class Settings:
    pg_host: str
    pg_port: int
    pg_user: str
    pg_password: str
    pg_db: str

    redis_host: str
    redis_port: int

    opensearch_host: str
    opensearch_port: int
    opensearch_scheme: str

    embedder_url: str
    embedder_model: str
    embedder_dim: int

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket_data: str
    minio_bucket_mlflow: str

    mlflow_tracking_uri: str
    ltr_model_backend: str

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val not in (None, "") else default


def normalize_ltr_model_backend(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "lgbm": "lgbm",
        "lightgbm": "lgbm",
        "xgb": "xgboost",
        "xgboost": "xgboost",
    }
    if normalized not in aliases:
        raise ValueError(
            f"LTR_MODEL_BACKEND must be one of lgbm, lightgbm, xgb, xgboost; got {value!r}"
        )
    return aliases[normalized]


def load() -> Settings:
    return Settings(
        pg_host=_env("POSTGRES_HOST", "postgres"),
        pg_port=int(_env("POSTGRES_PORT", "5432")),
        pg_user=_env("POSTGRES_USER", "drusearch"),
        pg_password=_env("POSTGRES_PASSWORD", "drusearch"),
        pg_db=_env("POSTGRES_DB", "drusearch"),
        redis_host=_env("REDIS_HOST", "redis"),
        redis_port=int(_env("REDIS_PORT", "6379")),
        opensearch_host=_env("OPENSEARCH_HOST", "opensearch"),
        opensearch_port=int(_env("OPENSEARCH_PORT", "9200")),
        opensearch_scheme=_env("OPENSEARCH_SCHEME", "http"),
        embedder_url=f"http://{_env('EMBEDDER_HOST', 'embedder')}:{_env('EMBEDDER_PORT', '8000')}",
        embedder_model=_env("EMBEDDER_MODEL", "BAAI/bge-base-en-v1.5"),
        embedder_dim=int(_env("EMBEDDER_DIM", "768")),
        minio_endpoint=f"http://{_env('MINIO_HOST', 'minio')}:{_env('MINIO_PORT', '9000')}",
        minio_access_key=_env("MINIO_ROOT_USER", "drusearch"),
        minio_secret_key=_env("MINIO_ROOT_PASSWORD", "drusearch1234"),
        minio_bucket_data=_env("MINIO_BUCKET_DATA", "drusearch-data"),
        minio_bucket_mlflow=_env("MINIO_BUCKET_MLFLOW", "mlflow-artifacts"),
        mlflow_tracking_uri=_env("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        ltr_model_backend=normalize_ltr_model_backend(_env("LTR_MODEL_BACKEND", "lgbm")),
    )
