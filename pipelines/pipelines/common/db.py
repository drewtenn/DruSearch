"""Postgres connection helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool

from pipelines.common.config import load


_POOL: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _POOL
    if _POOL is None:
        cfg = load()
        _POOL = ConnectionPool(conninfo=cfg.pg_dsn, min_size=1, max_size=8, open=True)
    return _POOL


@contextmanager
def conn() -> Iterator[psycopg.Connection]:
    with pool().connection() as c:
        yield c


def healthcheck() -> bool:
    try:
        with conn() as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False
