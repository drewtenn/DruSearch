from __future__ import annotations

from contextlib import AbstractContextManager

from pipelines.evaluate import offline_eval


class _FakeCursor(AbstractContextManager):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        assert "HAVING COUNT(*) >= %s" in sql
        assert "BOOL_OR(j.esci_label IN ('E', 'S', 'C'))" in sql
        assert params == (5, 10)

    def fetchall(self):
        return [(101, "running shoes", 1)]


class _FakeConnection(AbstractContextManager):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakeCursor()


def test_load_test_queries_requires_dense_positive_judgments_by_default(monkeypatch):
    monkeypatch.setattr(offline_eval.db, "conn", lambda: _FakeConnection())

    assert offline_eval._load_test_queries(10) == [(101, "running shoes")]
