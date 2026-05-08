"""Smoke test: verify pipelines container can talk to Postgres and OpenSearch.

Run via:
    docker compose run --rm pipelines python -m pipelines.check_health
"""

from __future__ import annotations

import sys

from pipelines.common import db, opensearch_client
from pipelines.common.logging import configure


def main() -> int:
    log = configure("check_health")
    pg = db.healthcheck()
    os_ok = opensearch_client.healthcheck()
    log.info("postgres=%s opensearch=%s", pg, os_ok)
    return 0 if pg and os_ok else 1


if __name__ == "__main__":
    sys.exit(main())
