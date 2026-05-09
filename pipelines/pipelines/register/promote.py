"""Promote a LightGBM model.txt artifact to the shared model volume.

Picks the latest version of LTR_MODEL_NAME from the MLflow registry
(highest version number, optionally filtered by stage), downloads its
`model_text/model.txt` artifact, rewrites the objective metadata if
needed, and writes it to /models/<name>.txt. The Go API watches that
path and reloads on /admin/reload-model.

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
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from pipelines.common.config import load
from pipelines.features import _generated as feature_schema
from pipelines.common.logging import configure

log = configure("register.promote")

LTR_MODEL_NAME = os.getenv("LTR_MODEL_NAME", "ltr_reranker")
MODEL_DIR = Path(os.getenv("LTR_MODEL_DIR", "/models"))
TARGET_FILE = MODEL_DIR / f"{LTR_MODEL_NAME}.txt"
META_FILE = MODEL_DIR / f"{LTR_MODEL_NAME}.json"
ARTIFACT_PATH = "model_text/model.txt"

# Stage filter: blank means "any stage, take highest version".
STAGE_FILTER = os.getenv("LTR_MODEL_STAGE", "").strip()


def _select_version(client: MlflowClient) -> mlflow.entities.model_registry.ModelVersion:
    versions = client.search_model_versions(f"name='{LTR_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"no versions registered for model '{LTR_MODEL_NAME}'")
    if STAGE_FILTER:
        versions = [v for v in versions if (v.current_stage or "").lower() == STAGE_FILTER.lower()]
        if not versions:
            raise RuntimeError(f"no versions for '{LTR_MODEL_NAME}' in stage '{STAGE_FILTER}'")
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


def _metadata_for_version(version: mlflow.entities.model_registry.ModelVersion) -> dict:
    return {
        "name":                   version.name,
        "version":                version.version,
        "stage":                  version.current_stage,
        "run_id":                 version.run_id,
        "source":                 version.source,
        "feature_schema_version": feature_schema.SCHEMA_VERSION,
    }


def main() -> int:
    cfg = load()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()

    version = _select_version(client)
    log.info(
        "selected model name=%s version=%s stage=%s run_id=%s",
        version.name, version.version, version.current_stage or "(none)", version.run_id,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading artifact %s", ARTIFACT_PATH)
    local = client.download_artifacts(version.run_id, ARTIFACT_PATH)
    src = Path(local)

    raw = src.read_text()
    rewritten = _rewrite_for_leaves(raw)
    if rewritten != raw:
        log.info("rewrote header for leaves compatibility (objective + version)")

    tmp = TARGET_FILE.with_suffix(".txt.tmp")
    tmp.write_text(rewritten)
    tmp.replace(TARGET_FILE)
    log.info("wrote %s (%d bytes)", TARGET_FILE, TARGET_FILE.stat().st_size)

    # Companion metadata so the API can report what's loaded
    import json
    META_FILE.write_text(json.dumps(_metadata_for_version(version), indent=2))
    log.info("wrote %s", META_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
