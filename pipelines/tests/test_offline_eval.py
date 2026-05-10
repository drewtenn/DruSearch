from __future__ import annotations

import os
import subprocess
import sys
from contextlib import AbstractContextManager
from pathlib import Path

from pipelines.evaluate import offline_eval


ROOT = Path(__file__).resolve().parents[2]


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


def test_eval_queries_can_be_read_from_dotenv(tmp_path):
    (tmp_path / ".env").write_text("EVAL_QUERIES=17\n")
    env = os.environ.copy()
    env.pop("EVAL_QUERIES", None)
    env["PYTHONPATH"] = str(ROOT / "pipelines")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pipelines.evaluate import offline_eval; print(offline_eval.EVAL_QUERIES)",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "17"


def test_process_env_overrides_dotenv_eval_queries(tmp_path):
    (tmp_path / ".env").write_text("EVAL_QUERIES=17\n")
    env = os.environ.copy()
    env["EVAL_QUERIES"] = "23"
    env["PYTHONPATH"] = str(ROOT / "pipelines")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pipelines.evaluate import offline_eval; print(offline_eval.EVAL_QUERIES)",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "23"


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
