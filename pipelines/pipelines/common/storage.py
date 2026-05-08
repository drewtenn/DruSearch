"""S3 / MinIO helpers."""

from __future__ import annotations

import functools

import boto3
from botocore.client import Config

from pipelines.common.config import load


@functools.lru_cache(maxsize=1)
def s3_client():
    cfg = load()
    return boto3.client(
        "s3",
        endpoint_url=cfg.minio_endpoint,
        aws_access_key_id=cfg.minio_access_key,
        aws_secret_access_key=cfg.minio_secret_key,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "adaptive"}),
        region_name="us-east-1",
    )


def object_exists(bucket: str, key: str) -> bool:
    s3 = s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
