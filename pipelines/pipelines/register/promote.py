"""Promote an LTR model artifact to the shared model volume.

Picks the latest version of LTR_MODEL_NAME from the MLflow registry
(highest version number, optionally filtered by stage and backend), downloads
the backend-specific artifact, and writes it to /models. The Go API watches
the metadata and reloads the matching scorer on /admin/reload-model.

Why the metadata rewrite: dmitryikh/leaves does not list `lambdarank`
in its parsed objectives. The trees still produce raw scores
(LambdaRank has no inverse link), so rewriting the model header's
`objective=lambdarank` line to `objective=regression` is safe and
makes the loader happy. Verified with a side-by-side score check
in tests/parity.

Run: docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.register.promote
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from pipelines.common.config import load, normalize_ltr_model_backend
from pipelines.features import _generated as feature_schema
from pipelines.common.logging import configure

log = configure("register.promote")

LTR_MODEL_NAME = os.getenv("LTR_MODEL_NAME", "ltr_reranker")
MODEL_DIR = Path(os.getenv("LTR_MODEL_DIR", "/models"))
META_FILE = MODEL_DIR / f"{LTR_MODEL_NAME}.json"

# Stage filter: blank means "any stage, take highest version".
STAGE_FILTER = os.getenv("LTR_MODEL_STAGE", "").strip()


def _artifact_path_for_backend(model_backend: str) -> str:
    if model_backend == "lgbm":
        return "model_text/model.txt"
    if model_backend == "xgboost":
        return "model_xgboost/model.json"
    raise ValueError(f"unsupported LTR model backend: {model_backend}")


def _target_file_for_backend(model_backend: str) -> Path:
    if model_backend == "lgbm":
        return MODEL_DIR / f"{LTR_MODEL_NAME}.txt"
    if model_backend == "xgboost":
        return MODEL_DIR / f"{LTR_MODEL_NAME}.xgb.json"
    raise ValueError(f"unsupported LTR model backend: {model_backend}")


def _version_backend(client: MlflowClient, version: mlflow.entities.model_registry.ModelVersion) -> str:
    run = client.get_run(version.run_id)
    return normalize_ltr_model_backend(run.data.params.get("ltr_model_backend", "lgbm"))


def _select_version(client: MlflowClient, model_backend: str) -> mlflow.entities.model_registry.ModelVersion:
    versions = client.search_model_versions(f"name='{LTR_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"no versions registered for model '{LTR_MODEL_NAME}'")
    versions = [v for v in versions if _version_backend(client, v) == model_backend]
    if not versions:
        raise RuntimeError(f"no {model_backend} versions registered for model '{LTR_MODEL_NAME}'")
    if STAGE_FILTER:
        versions = [v for v in versions if (v.current_stage or "").lower() == STAGE_FILTER.lower()]
        if not versions:
            raise RuntimeError(f"no {model_backend} versions for '{LTR_MODEL_NAME}' in stage '{STAGE_FILTER}'")
    versions.sort(key=lambda v: int(v.version))
    return versions[-1]


def _rewrite_for_leaves(text: str) -> str:
    """Rewrite the LightGBM model header so dmitryikh/leaves accepts it.

    Two surgical edits, both confined to the metadata block:

    1. objective=lambdarank -> objective=regression
       leaves doesn't recognise lambdarank but is happy with regression
       (no inverse-link transform applied to raw scores, which is what
       LambdaRank produces anyway).

    2. version=v4 -> version=v3
       leaves' loader allow-lists v2 and v3 only. LightGBM 4.x writes
       v4. The tree-block serialisation didn't change across the bump,
       only metadata fields expanded; rewriting the version line is
       the standard workaround. Parity is verified with a side-by-side
       prediction check (see tests/parity).
    """
    text = re.sub(
        r"^objective=lambdarank.*$",
        "objective=regression",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^version=v4\b.*$",
        "version=v3",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return text


def _metadata_for_version(
    version: mlflow.entities.model_registry.ModelVersion,
    model_backend: str = "lgbm",
) -> dict:
    return {
        "name":                   version.name,
        "version":                version.version,
        "stage":                  version.current_stage,
        "run_id":                 version.run_id,
        "source":                 version.source,
        "model_backend":          model_backend,
        "feature_schema_version": feature_schema.SCHEMA_VERSION,
    }


def main() -> int:
    cfg = load()
    model_backend = cfg.ltr_model_backend
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()

    version = _select_version(client, model_backend)
    log.info(
        "selected model name=%s version=%s stage=%s backend=%s run_id=%s",
        version.name, version.version, version.current_stage or "(none)", model_backend, version.run_id,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = _artifact_path_for_backend(model_backend)
    target_file = _target_file_for_backend(model_backend)
    log.info("downloading artifact %s", artifact_path)
    local = client.download_artifacts(version.run_id, artifact_path)
    src = Path(local)

    tmp = target_file.parent / f"{target_file.name}.tmp"
    if model_backend == "lgbm":
        raw = src.read_text()
        rewritten = _rewrite_for_leaves(raw)
        if rewritten != raw:
            log.info("rewrote header for leaves compatibility (objective + version)")
        tmp.write_text(rewritten)
    elif model_backend == "xgboost":
        tmp.write_bytes(src.read_bytes())
    else:
        raise ValueError(f"unsupported LTR model backend: {model_backend}")

    tmp.replace(target_file)
    log.info("wrote %s (%d bytes)", target_file, target_file.stat().st_size)

    # Companion metadata so the API can report what's loaded
    import json
    META_FILE.write_text(json.dumps(_metadata_for_version(version, model_backend), indent=2))
    log.info("wrote %s", META_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
