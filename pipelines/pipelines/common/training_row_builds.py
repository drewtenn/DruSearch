"""Training-row generation metadata and freshness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Json

from pipelines.common import db


@dataclass(frozen=True)
class TrainingRowBuild:
    build_id: int
    source: str
    feature_schema_version: str
    cand_n: int
    pseudo_labels_enabled: bool
    pseudo_label_weight: float
    status: str
    row_count: int
    query_count: int
    metadata: dict[str, Any]


def ensure_schema(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS training_row_builds (
          build_id BIGSERIAL PRIMARY KEY,
          started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          finished_at TIMESTAMPTZ,
          status TEXT NOT NULL CHECK (status IN ('running', 'ready', 'failed')),
          source TEXT NOT NULL,
          feature_schema_version TEXT NOT NULL,
          cand_n INTEGER NOT NULL,
          pseudo_labels_enabled BOOLEAN NOT NULL,
          pseudo_label_weight REAL NOT NULL,
          row_count INTEGER NOT NULL DEFAULT 0,
          query_count INTEGER NOT NULL DEFAULT 0,
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """
    )
    cur.execute(
        "ALTER TABLE training_rows "
        "ADD COLUMN IF NOT EXISTS sample_weight REAL NOT NULL DEFAULT 1"
    )
    cur.execute("ALTER TABLE training_rows ADD COLUMN IF NOT EXISTS build_id BIGINT")
    cur.execute(
        """
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'training_rows_build_id_fkey'
          ) THEN
            ALTER TABLE training_rows
            ADD CONSTRAINT training_rows_build_id_fkey
            FOREIGN KEY (build_id) REFERENCES training_row_builds(build_id);
          END IF;
        END $$;
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS training_rows_build_id_idx ON training_rows(build_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS training_row_builds_status_build_id_idx "
        "ON training_row_builds(status, build_id DESC)"
    )


def begin_training_row_build(
    *,
    source: str,
    feature_schema_version: str,
    cand_n: int,
    pseudo_labels_enabled: bool,
    pseudo_label_weight: float,
    feature_names: list[str],
) -> int:
    """Start a new generation and invalidate previously trainable rows."""
    with db.conn() as c, c.cursor() as cur:
        ensure_schema(cur)
        cur.execute("TRUNCATE training_rows")
        cur.execute("ALTER TABLE training_rows ALTER COLUMN build_id SET NOT NULL")
        c.commit()

    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            INSERT INTO training_row_builds
              (status, source, feature_schema_version, cand_n,
               pseudo_labels_enabled, pseudo_label_weight, metadata)
            VALUES ('running', %s, %s, %s, %s, %s, %s)
            RETURNING build_id
            """,
            (
                source,
                feature_schema_version,
                cand_n,
                pseudo_labels_enabled,
                pseudo_label_weight,
                Json({"feature_names": feature_names}),
            ),
        )
        build_id = int(cur.fetchone()[0])
        c.commit()
        return build_id


def mark_training_row_build_ready(
    build_id: int,
    *,
    row_count: int,
    query_count: int,
    split_counts: dict[str, int],
) -> None:
    with db.conn() as c, c.cursor() as cur:
        ensure_schema(cur)
        mark_training_row_build_ready_in_cursor(
            cur,
            build_id,
            row_count=row_count,
            query_count=query_count,
            split_counts=split_counts,
        )
        c.commit()


def mark_training_row_build_ready_in_cursor(
    cur,
    build_id: int,
    *,
    row_count: int,
    query_count: int,
    split_counts: dict[str, int],
) -> None:
    cur.execute(
        """
        UPDATE training_row_builds
        SET status = 'ready',
            finished_at = now(),
            row_count = %s,
            query_count = %s,
            metadata = metadata || %s::jsonb
        WHERE build_id = %s
        """,
        (row_count, query_count, Json({"split_counts": split_counts}), build_id),
    )


def mark_training_row_build_failed(build_id: int, error: BaseException | str) -> None:
    with db.conn() as c, c.cursor() as cur:
        ensure_schema(cur)
        cur.execute(
            """
            UPDATE training_row_builds
            SET status = 'failed',
                finished_at = now(),
                metadata = metadata || %s::jsonb
            WHERE build_id = %s
            """,
            (Json({"error": str(error)}), build_id),
        )
        c.commit()


def latest_ready_training_row_build(cur) -> TrainingRowBuild | None:
    cur.execute(
        """
        SELECT build_id, source, feature_schema_version, cand_n,
               pseudo_labels_enabled, pseudo_label_weight, status,
               row_count, query_count, metadata
        FROM training_row_builds
        WHERE status = 'ready'
        ORDER BY build_id DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if row is None:
        return None
    return TrainingRowBuild(
        build_id=int(row[0]),
        source=str(row[1]),
        feature_schema_version=str(row[2]),
        cand_n=int(row[3]),
        pseudo_labels_enabled=bool(row[4]),
        pseudo_label_weight=float(row[5]),
        status=str(row[6]),
        row_count=int(row[7]),
        query_count=int(row[8]),
        metadata=dict(row[9] or {}),
    )


def validate_ready_build(
    build: TrainingRowBuild | None,
    *,
    expected_source: str,
    expected_feature_schema_version: str,
    expected_cand_n: int,
    expected_pseudo_labels_enabled: bool,
    expected_pseudo_label_weight: float,
    actual_row_count: int,
    actual_query_count: int,
) -> list[str]:
    if build is None:
        return ["no ready training row build found"]

    errors: list[str] = []
    if build.status != "ready":
        errors.append(f"status {build.status} != ready")
    if build.source != expected_source:
        errors.append(f"source {build.source} != {expected_source}")
    if build.feature_schema_version != expected_feature_schema_version:
        errors.append(
            f"feature_schema_version {build.feature_schema_version} "
            f"!= {expected_feature_schema_version}"
        )
    if build.cand_n != expected_cand_n:
        errors.append(f"cand_n {build.cand_n} != {expected_cand_n}")
    if build.pseudo_labels_enabled != expected_pseudo_labels_enabled:
        errors.append(
            f"pseudo_labels_enabled {build.pseudo_labels_enabled} "
            f"!= {expected_pseudo_labels_enabled}"
        )
    if abs(build.pseudo_label_weight - expected_pseudo_label_weight) > 1e-9:
        errors.append(
            f"pseudo_label_weight {build.pseudo_label_weight:g} "
            f"!= {expected_pseudo_label_weight:g}"
        )
    if build.row_count != actual_row_count:
        errors.append(f"row_count {build.row_count} != loaded rows {actual_row_count}")
    if build.query_count != actual_query_count:
        errors.append(f"query_count {build.query_count} != loaded queries {actual_query_count}")
    return errors
