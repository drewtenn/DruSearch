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


def test_metric_accumulator_tracks_deep_recall_and_zero_results():
    metrics = offline_eval.MetricAccumulator()
    judgments = {
        "p1": "E",
        "p2": "S",
        "p3": "C",
    }

    metrics.add_rrf(judgments, ["p1"], [4.0], [4.0, 3.0, 2.0])
    metrics.add_zero_result()

    result = metrics.result()

    assert result["queries_evaluated"] == 2
    assert result["zero_result_rate"] == 0.5
    assert result["rrf"]["recall_at_10"] == 1 / 3
    assert result["rrf"]["recall_at_50"] == 1 / 3
    assert result["rrf"]["recall_at_100"] == 1 / 3
    assert result["rrf"]["candidate_recall"] == 1 / 3
