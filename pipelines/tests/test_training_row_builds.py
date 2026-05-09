from __future__ import annotations

from contextlib import contextmanager

from pipelines.common import training_row_builds


class FakeCursor:
    description = []

    def __init__(self):
        self.statements: list[tuple[str, object | None]] = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))

    def fetchone(self):
        return (42,)


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.commits = 0

    @contextmanager
    def cursor(self):
        yield self.cursor_obj

    def commit(self):
        self.commits += 1


def test_begin_training_row_build_records_generation_and_clears_old_rows(monkeypatch):
    conn = FakeConnection()

    @contextmanager
    def fake_conn():
        yield conn

    monkeypatch.setattr(training_row_builds.db, "conn", fake_conn)

    build_id = training_row_builds.begin_training_row_build(
        source="offline_candidates",
        feature_schema_version="v6",
        cand_n=200,
        pseudo_labels_enabled=False,
        pseudo_label_weight=0.25,
        feature_names=["bm25_score", "rrf_score"],
    )

    sql = "\n".join(statement for statement, _params in conn.cursor_obj.statements).lower()
    assert build_id == 42
    assert conn.commits == 2
    assert "create table if not exists training_row_builds" in sql
    assert "alter table training_rows add column if not exists build_id bigint" in sql
    assert "truncate training_rows" in sql
    assert "alter table training_rows alter column build_id set not null" in sql
    assert "insert into training_row_builds" in sql


def test_training_row_build_contract_rejects_stale_or_mismatched_rows():
    build = training_row_builds.TrainingRowBuild(
        build_id=7,
        source="search_events",
        feature_schema_version="v5",
        cand_n=100,
        pseudo_labels_enabled=True,
        pseudo_label_weight=0.5,
        status="ready",
        row_count=10,
        query_count=3,
        metadata={},
    )

    errors = training_row_builds.validate_ready_build(
        build,
        expected_source="offline_candidates",
        expected_feature_schema_version="v6",
        expected_cand_n=200,
        expected_pseudo_labels_enabled=False,
        expected_pseudo_label_weight=0.25,
        actual_row_count=9,
        actual_query_count=2,
    )

    assert "source search_events != offline_candidates" in errors
    assert "feature_schema_version v5 != v6" in errors
    assert "cand_n 100 != 200" in errors
    assert "pseudo_labels_enabled True != False" in errors
    assert "pseudo_label_weight 0.5 != 0.25" in errors
    assert "row_count 10 != loaded rows 9" in errors
    assert "query_count 3 != loaded queries 2" in errors


def test_missing_training_row_build_is_not_trainable():
    assert training_row_builds.validate_ready_build(
        None,
        expected_source="offline_candidates",
        expected_feature_schema_version="v6",
        expected_cand_n=200,
        expected_pseudo_labels_enabled=False,
        expected_pseudo_label_weight=0.25,
        actual_row_count=0,
        actual_query_count=0,
    ) == ["no ready training row build found"]
