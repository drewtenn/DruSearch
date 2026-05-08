-- DruSearch initial schema
-- Phase 0: tables exist; later phases populate them.

SET client_min_messages = WARNING;

-- ---------------------------------------------------------------------------
-- Catalog
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
  product_id        TEXT PRIMARY KEY,
  locale            TEXT        NOT NULL DEFAULT 'us',
  title             TEXT        NOT NULL,
  description       TEXT,
  bullet_points     TEXT,
  brand             TEXT,
  color             TEXT,
  price_cents       INTEGER,
  category          TEXT,
  category_path     TEXT[]      NOT NULL DEFAULT ARRAY[]::TEXT[],
  popularity_prior  REAL        NOT NULL DEFAULT 0,
  raw_metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS products_brand_idx    ON products(brand);
CREATE INDEX IF NOT EXISTS products_category_idx ON products(category);
CREATE INDEX IF NOT EXISTS products_category_path_gin_idx ON products USING GIN(category_path);

-- ---------------------------------------------------------------------------
-- ESCI judgments (offline eval only; not used at serve time)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS esci_judgments (
  query_id    INTEGER NOT NULL,
  query       TEXT    NOT NULL,
  product_id  TEXT    NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
  esci_label  TEXT    NOT NULL CHECK (esci_label IN ('E','S','C','I')),
  split       TEXT    NOT NULL CHECK (split IN ('train','test')),
  PRIMARY KEY (query_id, product_id)
);

CREATE INDEX IF NOT EXISTS esci_judgments_query_idx ON esci_judgments(query);

-- ---------------------------------------------------------------------------
-- Sessions (lightweight)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_sessions (
  session_id  TEXT PRIMARY KEY,
  user_id     TEXT,
  started_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Append-only event log
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS search_events (
  event_id          BIGSERIAL PRIMARY KEY,
  event_type        TEXT        NOT NULL CHECK (event_type IN ('impression','click','purchase')),
  ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_id           TEXT,
  session_id        TEXT        NOT NULL,
  query             TEXT        NOT NULL,
  query_id          TEXT        NOT NULL,
  product_id        TEXT        NOT NULL,
  position          INTEGER     NOT NULL,
  retrieval_scores  JSONB,
  source            TEXT        NOT NULL DEFAULT 'real'
);

CREATE INDEX IF NOT EXISTS search_events_query_id_idx ON search_events(query_id);
CREATE INDEX IF NOT EXISTS search_events_user_ts_idx  ON search_events(user_id, ts DESC);
CREATE INDEX IF NOT EXISTS search_events_product_idx  ON search_events(product_id);

-- ---------------------------------------------------------------------------
-- Offline feature snapshots (LTR training rows)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training_rows (
  query_id    TEXT        NOT NULL,
  product_id  TEXT        NOT NULL,
  query       TEXT        NOT NULL,
  user_id     TEXT,
  ts          TIMESTAMPTZ NOT NULL,
  features    JSONB       NOT NULL,
  label       REAL        NOT NULL,
  split       TEXT        NOT NULL CHECK (split IN ('train','val','test')),
  PRIMARY KEY (query_id, product_id)
);

-- ---------------------------------------------------------------------------
-- Product-level offline features (refreshed daily)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS product_features (
  product_id         TEXT PRIMARY KEY REFERENCES products(product_id) ON DELETE CASCADE,
  impressions_30d    INTEGER     NOT NULL DEFAULT 0,
  clicks_30d         INTEGER     NOT NULL DEFAULT 0,
  purchases_30d      INTEGER     NOT NULL DEFAULT 0,
  ctr_prior          REAL        NOT NULL DEFAULT 0,
  pos_corrected_ctr  REAL        NOT NULL DEFAULT 0,
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Bookkeeping for idempotent pipeline runs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_runs (
  run_id        BIGSERIAL PRIMARY KEY,
  pipeline      TEXT NOT NULL,
  dataset_hash  TEXT NOT NULL,
  started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at   TIMESTAMPTZ,
  status        TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','ok','failed')),
  metadata      JSONB
);

CREATE INDEX IF NOT EXISTS ingest_runs_pipeline_idx ON ingest_runs(pipeline, started_at DESC);
