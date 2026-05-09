# DruSearch Architecture

DruSearch is a local-first e-commerce search stack. It combines BM25 lexical
retrieval, dense vector retrieval, reciprocal-rank fusion, behavior logging,
offline label distillation, and a Go-served LightGBM Learning-to-Rank model.

Status: **local hybrid + personalized LTR stack is implemented.** The default
catalog source is Amazon Shopping Queries / ESCI, preserving real query groups
with graded `E`/`S`/`C`/`I` judgments for evaluation and training labels. The
default local target is about 10,000 products, vectors come from the
`BAAI/bge-small-en-v1.5` embedder sidecar, and the served ranker uses the
canonical v4 feature schema with 27 features. Model promotion writes served
artifacts under `models/`; commit those artifacts when another machine should
serve the same trained ranker without rerunning BGE teacher scoring or training.

The system is intentionally production-shaped, but it is still optimized for a
single-machine Docker Compose demo. Current sizing is about 10k products and
less than 50 QPS. Scale-rehearsal work for aliases, caches, configurable
retrieval, and larger synthetic catalogs is designed but not implemented.

## Table of Contents

- [System Diagram](#system-diagram)
- [Components](#components)
- [Request Path](#request-path)
- [Data Path](#data-path)
- [ML Pipeline](#ml-pipeline)
- [Data Model](#data-model)
- [APIs](#apis)
- [Feature Parity](#feature-parity)
- [Latency Budget](#latency-budget)
- [Operational Notes](#operational-notes)
- [Personalization](#personalization)
- [Open Lessons](#open-lessons)
- [Roadmap](#roadmap)

## System Diagram

```text
                          client / curl / UI
                                  |
                                  v
+-----------------------------------------------------------------------+
| Go API service (services/api-go)                                      |
| /search  /events  /products/{id}  /metrics  /healthz  /readyz  /admin |
|                                                                       |
| router -> retrieval -> feature builder -> LightGBM reranker           |
|             |              |                   |                      |
+-------------+--------------+-------------------+----------------------+
              |              |                   |
              v              v                   v
       +-------------+  +-----------+       +------------+
       | OpenSearch  |  | Redis     |       | models/    |
       | BM25 + kNN  |  | user      |       | LTR .txt + |
       | products_v1 |  | features  |       | metadata   |
       +------+------+  +-----+-----+       +------------+
              ^               ^
              |               |
       +------+---------------+--------------------------------------+
       | Python pipelines (pipelines/)                               |
       |                                                             |
       | ingest.esci -> index.bm25 -> embed.build_vectors            |
       | simulate.click_simulator -> /events                         |
       | features.aggregates -> Postgres product_features            |
       | features.user_aggs -> Redis feat:user:{id}                  |
       | label.build_training_rows -> label.bge_teacher              |
       | train.lgbm_ranker -> MLflow -> register.gate/promote        |
       +-----------+-----------------------+-------------------------+
                   |                       |
                   v                       v
          +----------------+        +--------------+
          | Postgres       |        | MLflow/MinIO |
          | catalog/events |        | runs/artifacts|
          | training rows  |        +--------------+
          +----------------+

       +------------------+
       | Embedder sidecar |
       | FastAPI /embed   |
       | /embed_batch     |
       +------------------+
```

All services run in one Docker Compose network. Runtime URLs, credentials, and
model names are driven by `.env` and the defaults in `docker-compose.yml`.

## Components

| Component | Module / image | Purpose | Notes |
|---|---|---|---|
| Go API | `services/api-go` | Hot path for search, product lookup, events, admin, metrics | Stateless apart from in-process model handle and event buffer. |
| Embedder sidecar | `services/embedder-py` | Encodes queries and product titles into 384-dim vectors | Default model is `BAAI/bge-small-en-v1.5`; same code path is used for indexing and querying. |
| OpenSearch 2.19.1 | `opensearch` | BM25 and Lucene HNSW kNN over `products_v1` | Single shard, zero replicas, local security disabled. |
| Postgres 16 | `postgres` | Product catalog, ESCI graded judgments, event log, training rows, product aggregates | Schema lives in `infra/postgres/001_init.sql`. |
| Redis 7 | `redis` | Online user feature store | `features.user_aggs` writes per-user brand affinity hashes. |
| MLflow | `infra/mlflow` image | Experiment tracking and model registry | SQLite backend, MinIO artifact store. |
| MinIO | `minio` | Local S3-compatible artifact storage | Used by MLflow and older/cacheable data flows. |
| Python pipelines | `pipelines/` | Ingest, indexing, embedding, simulation, features, labeling, training, evaluation, promotion | Runnable with `python -m pipelines.<module>` through Compose jobs or selected host targets. |
| Model artifact directory | `models/` | Served LTR model and metadata | LightGBM promotion writes `ltr_reranker.txt`; XGBoost promotion writes `ltr_reranker.xgb.json`; both write `ltr_reranker.json`. |

Go workspace:

```text
services/api-go/
├── cmd/api/main.go
└── internal/
    ├── config/       # env-driven config
    ├── embedder/     # sidecar client + circuit breaker
    ├── eventbus/     # async Postgres event writes
    ├── features/     # feature schema, transforms, matrix builder, Redis user features
    ├── httpapi/      # chi router and handlers
    ├── obs/          # Prometheus metrics
    ├── products/     # product lookup
    ├── rerank/       # LightGBM model loading and scoring through leaves
    ├── retrieval/    # OpenSearch BM25, kNN, and RRF fusion
    └── store/        # Postgres, Redis, OpenSearch clients
```

Python pipeline tree:

```text
pipelines/pipelines/
├── common/                 # config, db, OpenSearch, storage, logging, torch device
├── embed/build_vectors.py  # product titles -> embedder /embed_batch -> OpenSearch vectors
├── evaluate/               # offline eval and API quality smoke checks
├── features/               # product aggregates and Redis user aggregates
├── index/bm25.py           # Postgres products -> OpenSearch products_v1
├── ingest/esci.py           # Amazon Shopping Queries ESCI -> Postgres
├── ingest/amazon_reviews.py # Amazon Reviews 2023 metadata fallback -> Postgres
├── label/                  # training row build + offline BGE teacher labels
├── register/               # MLflow promotion gate + model artifact promotion
├── simulate/               # synthetic events
└── train/lgbm_ranker.py    # LightGBM LambdaRank training
```

## Request Path

Concrete trace for `GET /search?q=running+shoes&k=10&ranker=ltr`:

1. The API validates `q`, parses `k`, chooses the requested ranker, and creates
   or accepts a `session_id`.
2. It calls the embedder sidecar: `POST /embed {"text": "running shoes"}`.
   If the sidecar fails or the circuit breaker is open, the request degrades to
   BM25-only and starts with `mode: "bm25"`.
3. Retrieval runs against OpenSearch:
   - BM25: `multi_match` over `title^2`, `category_path^2`, `category^1.5`,
     `bullets`, and `description`. Gendered queries wrap that base query in a
     soft boost against indexed `derived_gender`; matching gender is boosted,
     unisex partially matches men's and women's queries, and known opposite
     genders are demoted rather than filtered out.
   - kNN: Lucene HNSW over `title_vec`, when embedding is available.
   - Structured query intent, including gender, stays as soft retrieval and
     ranking evidence rather than hard filters so sparse or ambiguous catalog
     metadata cannot zero results.
4. Hybrid retrieval fuses BM25 and kNN with client-side RRF:
   `score(d) = sum(1 / (rrf_k + rank(d)))`, with `rrf_k=60` by default.
5. If the ranker is `ltr` and a model is loaded, the API builds the v4 feature
   matrix, loads Redis user features when `user_id` is present, scores with the
   local LightGBM model, and sorts by predicted LTR score.
6. The response includes `query_id`, `session_id`, `mode`, optional
   `model_version`, `took_ms`, and result-level `explain` scores for BM25, kNN,
   RRF, and LTR.
7. Returned products are logged as impression events through the async event
   bus. Click and purchase events arrive through `POST /events`.

Current degradation behavior:

| Failure | Behavior |
|---|---|
| Embedder fails or circuit opens | Search degrades to BM25-only; LTR may still run, producing `bm25+ltr`. |
| BM25-only retrieval fails | HTTP 500. |
| Either leg of hybrid retrieval fails | HTTP 500; partial hybrid success is planned but not implemented. |
| Reranker model missing or scoring fails | Return retrieval order unchanged. |
| Admin token missing | `/admin/*` rejects requests with 503. |

## Data Path

The default data flow is:

```text
Amazon Shopping Queries / ESCI parquet files
        |
        v
pipelines.ingest.esci
        |
        +--> Postgres products
        +--> graded query-product judgments in esci_judgments
        |
        v
pipelines.index.bm25
        |
        v
OpenSearch products_v1 with BM25 fields
        |
        v
pipelines.embed.build_vectors -> embedder /embed_batch
        |
        v
OpenSearch products_v1 with title_vec
```

`make seed-catalog` runs `pipelines.ingest.esci`. It caches the Amazon Shopping
Queries examples and products parquet files, filters to US `small_version`
query-product examples, selects complete query groups until roughly
`ESCI_TARGET_PRODUCTS=50000` unique products are covered, and writes the
selected graded judgments to `esci_judgments`. These judgments are used for
offline evaluation and as authoritative labels when a training impression
matches the same `(query, product_id)` pair. `make seed-amazon-reviews` remains
available for catalog-only demos that do not need dense relevance judgments.

`pipelines.index.bm25` drops and recreates `products_v1` and bulk-loads product
documents. `pipelines.embed.build_vectors` streams products through the same
embedder sidecar used at query time and bulk-updates `title_vec`.

## ML Pipeline

The current ranking loop is:

```text
ESCI queries
        |
        v
features.aggregates        features.user_aggs
        |                          |
        v                          v
Postgres product_features   Redis feat:user:{id}
        |
        v
label.build_training_rows
        |
        v
training_rows with serving-aligned candidate feature snapshots,
ESCI-derived labels, sample weights, and required build_id
        |
        v
label.bge_teacher (optional)
        |
        v
training_rows with BGE teacher score/percentile; weak train-only labels only
when pseudo labels are explicitly enabled
        |
        v
train.lgbm_ranker -> MLflow registered model
        |
        v
register.gate -> register.promote -> models/ltr_reranker.{txt,json}
        |
        v
POST /admin/reload-model
```

`label.build_training_rows` emits one row per candidate from the same
BM25+kNN+RRF candidate distribution used by serving and offline eval. Labels
start from ESCI judgments: `E=4`, `S=3`, `C=2`, `I/unjudged=0`. LightGBM uses
`label_gain=[0,1,3,7,15]`; XGBoost uses `rank:ndcg` with the same integer
labels. Splits follow canonical ESCI query splits, with train queries further
split into train/validation by stable normalized-query hash. Weak lexical
pseudo labels are disabled by default; when `LTR_PSEUDO_LABELS=1`, they apply
only to train rows and receive `LTR_PSEUDO_LABEL_WEIGHT`.

Training row generations are tracked in `training_row_builds`. A row build
clears the previous generation before candidate retrieval begins, inserts rows
with a required `build_id`, and marks that generation `ready` only after the row
write commits. Training loads only the latest ready build and verifies the
source, feature schema version, candidate count, and pseudo-label settings
against the current environment before it starts a model run.

`label.bge_teacher` runs `BAAI/bge-reranker-v2-m3` offline. It stores
`bge_teacher_score` and `bge_teacher_percentile` in the row feature JSON for
audit/debugging. It only upgrades unjudged train rows to weak labels when
`BGE_TEACHER_PSEUDO_LABELS=1` or `LTR_PSEUDO_LABELS=1`, and those rows receive
a lower `sample_weight`. Known ESCI judgments remain authoritative. The BGE
teacher is never used on the live `/search` path.

`train.lgbm_ranker` trains the configured backend from `LTR_MODEL_BACKEND`
(`lgbm` by default, or `xgboost`) and registers it in MLflow. LightGBM logs
`model_text/model.txt`; XGBoost logs `model_xgboost/model.json`. It also logs
offline-style RRF baseline metrics and `test_ltr_lift_ndcg_at_10` on the
offline-eval-eligible ESCI test queries, using all judgments for ideal NDCG and
recall.
`register.promote` downloads the matching backend artifact, rewrites LightGBM
metadata when needed for the Go `leaves` loader, and writes either the served
`.txt` or `.xgb.json` plus companion `.json` metadata under `models/`. The API
uses `model_backend` from metadata to choose the LightGBM or XGBoost scorer and
reports the loaded model metadata as `model_version` when available.

## Data Model

Postgres tables:

| Table | Purpose |
|---|---|
| `products` | Catalog source of truth, including `category_path` and raw metadata. |
| `esci_judgments` | Offline evaluation/training judgments from real ESCI query-product labels. |
| `user_sessions` | Lightweight session metadata. |
| `search_events` | Append-only impression, click, and purchase events. |
| `training_row_builds` | Build manifest for LTR candidate rows: source, feature schema version, candidate count, label strategy settings, row/query counts, status, and metadata. |
| `training_rows` | One row per `(query_id, product_id)` training candidate with feature JSON, label, split, optional user, sample weight, and required `build_id`. |
| `product_features` | Product-level aggregate counters and CTR priors. |
| `ingest_runs` | Pipeline run bookkeeping. |

OpenSearch index `products_v1`:

| Field | Type | Source |
|---|---|---|
| `product_id` | keyword | Postgres |
| `title`, `description`, `bullets` | text with custom English analyzer | Postgres |
| `brand`, `color`, `category` | keyword | Postgres |
| `category_path` | text plus `.raw` keyword | Postgres |
| `derived_gender` | keyword | Ingestion-derived from category path, then title |
| `price_cents` | integer | Postgres |
| `popularity_prior`, `ctr_prior` | float | Postgres / aggregates |
| `title_vec` | `knn_vector(384)`, Lucene HNSW cosine similarity | Embedder sidecar |

## APIs

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Process liveness. |
| `GET` | `/readyz` | Dependency readiness for Postgres, Redis, OpenSearch, and embedder. |
| `GET` | `/metrics` | Prometheus metrics. |
| `GET` | `/search?q=&k=&user_id=&session_id=&ranker=` | Product search. |
| `GET` | `/products/{id}` | Product lookup. |
| `POST` | `/events` | Queue impression, click, or purchase feedback. |
| `POST` | `/admin/reload-model` | Hot-reload the promoted local LightGBM artifact. |
| `POST` | `/admin/reindex` | Present but returns 501; alias/reindex automation is planned. |

`/search` supports `ranker=hybrid`, `ranker=rrf`, or `ranker=ltr`. The default
comes from `DEFAULT_RANKER` and is `ltr` unless overridden. Admin endpoints
require `Authorization: Bearer $ADMIN_TOKEN` or `X-Admin-Token: $ADMIN_TOKEN`.

## Feature Parity

Python trains the ranker and Go serves it, so feature ordering and transforms
must stay identical.

The source of truth is `libs/schema/feature_schema.json`, currently schema
`v7` with 29 ordered features:

| Indexes | Source | Features |
|---|---|---|
| 0-4 | Retrieval | `bm25_score`, `bm25_rank`, `knn_score`, `knn_rank`, `rrf_score` |
| 5-7, 14 | Static product | `popularity_prior`, `price_log_cents`, `title_length_tokens`, `product_gender` |
| 8-13, 15-23, 25-26 | Interaction | Query/category/color/brand/gender/coverage/exact-match/affordability features |
| 24 | Online user | `user_brand_affinity` |

`libs/schema/codegen.py` generates:

- `pipelines/pipelines/features/_generated.py`
- `services/api-go/internal/features/schema_generated.go`

The hand-written mirrors in `pipelines/pipelines/features/__init__.py` and
`services/api-go/internal/features/schema.go` assert against generated output.
Interaction transforms are implemented in both Python and Go and tested against
shared fixtures in `libs/schema/fixtures/interaction_fixtures.json`.

Important commands:

```bash
make regen-feature-schema
make check-feature-parity
make test-go
make test-py
```

## Latency Budget

| Stage | p50 target | p99 target | Notes |
|---|---:|---:|---|
| Embedder sidecar | 6 ms | 15 ms | First request may be slower while model/runtime warms. |
| OpenSearch BM25 + kNN | 8 ms | 25 ms | Cold JVM/index path can spike. |
| RRF fuse | 1 ms | 3 ms | In-process. |
| Redis user features | 1 ms | 3 ms | One `HGETALL` when `user_id` is supplied. |
| LightGBM rerank | 3 ms | 8 ms | Local `leaves` scoring. |
| Serialize / HTTP | 2 ms | 5 ms | In-process JSON response. |
| End-to-end | about 25 ms | under 70 ms | Cold path can exceed this on a fresh stack. |

## Operational Notes

Cold start:

```bash
cp .env.example .env
make up
make ready
make seed-databases
make embed-vectors
```

Full local demo loop:

```bash
make bootstrap-search
```

Model loop after new events:

```bash
make refresh-product-features
make refresh-user-features
make build-training-rows
make label-bge-teacher
make train-ltr
make promote-model
make verify-promoted-model
make reload-model
```

On Apple Silicon, the BGE teacher can run on the host with MPS while the
databases and services stay in Docker:

```bash
make host-pipeline-venv
make label-bge-teacher-host
```

Observability:

- API metrics: `make metrics` or `curl http://localhost:8080/metrics`
- MLflow UI: `http://localhost:5000`
- MinIO console: `http://localhost:9001`
- OpenSearch health: `make os-health`

## Personalization

The online personalization signal is `user_brand_affinity`, stored in Redis as
`feat:user:{user_id}`. `pipelines.features.user_aggs` recomputes brand click
shares from `search_events`. At request time the API loads the user's hash and
sets feature 24 to the candidate brand's affinity. Anonymous requests get an
empty feature snapshot and therefore use `0` for this feature.

The ranker is trained with 30 percent anonymous masking so it learns both
personalized and non-personalized paths.

## Open Lessons

- Click labels alone are position-biased. The first click-label approach taught
  the model to imitate rank position. Current training uses synthetic
  ESCI-style exact judgments plus offline BGE teacher pseudo labels while click
  data feeds personalization features.
- BGE teacher scoring is useful offline, but too expensive for the live hot
  path. It intentionally distills into LightGBM rather than becoming a runtime
  reranker.
- Feature parity is the highest-risk correctness boundary. Any schema or
  transform change must regenerate code, run Go/Python parity tests, rebuild the
  API image, and retrain before serving.
- Partial retrieval degradation is still missing for hybrid mode. If BM25 or
  kNN fails after embedding succeeds, the current hybrid path returns 500.
- Scale-rehearsal work is planned: read/write aliases, versioned indexes,
  Redis embedding/candidate caches, configurable candidate counts/timeouts, and
  synthetic large-catalog load tests.

## Roadmap

| Phase | Status | Current demo |
|---|---|---|
| 0 - Scaffold + Compose | done | `make up && make ready` |
| 1 - BM25 search | done | `make seed-databases && curl '/search?q=running+shoes'` |
| 2 - Hybrid retrieval | done | `make embed-vectors`, then semantic queries use BM25+kNN+RRF. |
| 3 - Event ingest + simulator | done | `make simulate` writes impressions, clicks, and purchases. |
| 4 - LTR training | done | `make build-training-rows && make train-ltr`. |
| 5 - LTR serving in Go | done | `ranker=ltr`, `mode="hybrid+ltr"`, hot reload from `models/`. |
| 6 - Personalization | done | Redis `feat:user:{id}` supplies `user_brand_affinity`. |
| 7 - Observability + promotion safety | done | `/metrics`, embedder circuit breaker, `register.gate`, runbook. |
| 8 - Scale rehearsal | planned | Alias-based indexing, caches, partial retrieval, config knobs, load tests. |
