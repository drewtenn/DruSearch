"""Thin OpenSearch client wrapper."""

from __future__ import annotations

from opensearchpy import OpenSearch

from pipelines.common.config import load


_CLIENT: OpenSearch | None = None


def client() -> OpenSearch:
    global _CLIENT
    if _CLIENT is None:
        cfg = load()
        _CLIENT = OpenSearch(
            hosts=[{"host": cfg.opensearch_host, "port": cfg.opensearch_port, "scheme": cfg.opensearch_scheme}],
            http_compress=True,
            use_ssl=cfg.opensearch_scheme == "https",
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
        )
    return _CLIENT


def healthcheck() -> bool:
    try:
        return bool(client().cluster.health())
    except Exception:
        return False
