# DruSearch — Architecture

A personalized e-commerce search service: hybrid retrieval (BM25 + dense embeddings) with a Learning-to-Rank reranker driven by click data. Local-first, runnable via `docker compose up`. Sized for ~10k products and <50 QPS.

> Status: **Phase 7 complete.** Full hybrid + personalized LTR stack is live, instrumented, gated, and runbooked. Prometheus metrics on `/metrics`, embedder circuit breaker degrades to BM25-only when the sidecar fails, model promotion via `pipelines.register.gate` requires non-regression vs current Production, and the training pipeline now masks 30% of queries to anonymous so the model handles non-personalized requests cleanly. See [ROADMAP](#roadmap).

## Table of contents

- [System diagram](#system-diagram)
- [Components](#components)
- [Request path: a single search](#request-path-a-single-search)
- [Data path: catalog → indexed product](#data-path-catalog--indexed-product)
- [Data model](#data-model)
- [APIs](#apis)
- [Feature parity (the hardest design problem)](#feature-parity-the-hardest-design-problem)
- [Latency budget](#latency-budget)
- [Operational notes](#operational-notes)
- [Roadmap](#roadmap)

---

## System diagram

```
                           ┌──────────────────────────┐
                           │   client / curl / UI     │
                           └────────────┬─────────────┘
                                        │ HTTP
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         Go API service  (services/api-go)              │
 │   /search   /events   /products/{id}   /healthz   /readyz   /admin/*   │
 │                                                                        │
 │   ┌────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
 │   │ router │─▶│  retriever   │─▶│   features   │─▶│   reranker   │     │
 │   └────────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
 │                      │                 │                 │ (Phase 5)   │
 └──────────────────────┼─────────────────┼─────────────────┼─────────────┘
                        │                 │                 │
       ┌────────────────┴───────┐  ┌──────┴──────────┐  ┌───┴────────────┐
       ▼                        ▼  ▼                 ▼  ▼                ▼
┌──────────────┐   ┌────────────────────┐   ┌─────────────┐   ┌──────────────────┐
│  Embedder    │   │     OpenSearch     │   │    Redis    │   │     Postgres     │
│  (FastAPI)   │   │   (BM25 + k-NN)    │   │  (online    │   │  (catalog SoT,   │
│  /embed      │   │   products_v1      │   │   features  │   │   events, offline│
│              │   │                    │   │   Phase 6)  │   │   features)      │
└──────┬───────┘   └─────────▲──────────┘   └─────────────┘   └────────▲─────────┘
       │                     │                                         │
       │ same model code     │ bulk-update title_vec                   │
       │                     │                                         │
                                                                       │
 ┌─────┴─────────────────────┴─────────────────────────────────────────┴─────┐
 │                       Pipelines (services/pipelines, Python)               │
 │                                                                            │
 │   ingest.esci  ─▶  index.bm25  ─▶  embed.build_vectors                     │
 │                                                                            │
 │   simulate.click_simulator  ─▶  /events                                    │
 │                                                                            │
 │   features.aggregates ──┐                                                  │
 │                         ▼                                                  │
 │   label.build_training_rows  ─▶  train.lgbm_ranker  ─▶  MLflow             │
 │                              ─▶  evaluate.offline_eval (vs ESCI)           │
 │                                                                            │
 │   register.promote  ─▶  MinIO artifact ─▶  /admin/reload-model (Phase 5)   │
 │   features.user_aggs  ─▶  Redis                          (Phase 6)         │
 └────────────────────────────────────────────────────────────────────────────┘
                          │                       │
                          ▼                       ▼
                   ┌──────────────┐         ┌──────────────────┐
                   │    MLflow    │         │      MinIO       │
                   │ (registry +  │◀────────│ (S3-compat: ESCI │
                   │  tracking)   │ artifacts│  cache + models) │
                   └──────────────┘         └──────────────────┘
```

All services run in a single Docker Compose network. Full URLs and credentials live in `.env` (see `.env.example`).

---

## Components

| Component | Image / module | Purpose | Why this choice |
|---|---|---|---|
| **Go API** | `services/api-go` | Hot path: search, event ingest, product lookup, admin | Low-latency, single static binary, easy to tune for p99. |
| **Embedder sidecar** | `services/embedder-py` | Encode query text into a 384-dim vector | Same `sentence-transformers` model is used to index and to query → no tokenizer drift. ~6ms loopback overhead is acceptable. |
| **OpenSearch 2.19** | `opensearch` | BM25 lexical retrieval **and** Lucene HNSW k-NN, in one engine | Hybrid retrieval without a second vector store; native RRF support for native-pipeline mode (we use client-side RRF — see [Retrieval](#retrieval)). |
| **Postgres 16** | `postgres` | Catalog source of truth, append-only event log, offline feature snapshots | Reliable relational storage; `COPY` makes ESCI ingest fast. |
| **Redis 7** | `redis` | Online feature store (per-user / per-session), Phase 6 | Sub-ms reads on the hot path. HASH-per-entity. |
| **MLflow** | `mlflow` (custom image) | Experiment tracking + model registry, Phase 4–5 | SQLite-backed registry; MinIO is the artifact store. |
| **MinIO** | `minio` | Local S3 for ESCI parquet cache and MLflow artifacts | Clean cloud-port story; `boto3` works unmodified. |
| **Pipelines** | `pipelines/` (Python) | Batch ingest, indexing, embedding, simulation, training, eval | All ML/data work, runnable as `python -m pipelines.<name>` or via Prefect flows. |
| **Prefect (planned)** | (in `pyproject.toml`) | DAG orchestration for retrain loop, Phase 6 | Local UI on `:4200`; pipelines are also runnable directly. |

The full Go workspace contains:

```
services/api-go/
├── cmd/api/main.go            # entrypoint, signal handling, healthcheck mode
└── internal/
    ├── config/                # env-driven config
    ├── store/                 # pgxpool, go-redis, opensearch-go clients + Ping()
    ├── embedder/              # HTTP client for the FastAPI sidecar
    ├── retrieval/             # BM25, KNN, Hybrid (RRF fusion)
    ├── products/              # Postgres lookup
    └── httpapi/               # chi router, handlers (search, events, admin, health)
```

The Python pipelines tree:

```
pipelines/pipelines/
├── common/                    # config, db, opensearch_client, storage, logging
├── ingest/esci.py             # download ESCI → cache to MinIO → COPY to Postgres
├── index/bm25.py              # Postgres → OpenSearch BM25 docs
└── embed/build_vectors.py     # streaming Postgres → embedder /embed_batch → bulk update title_vec
```

---

## Request path: a single search

Concrete trace of `GET /search?q=running+shoes&k=10`:

1. **Mint a query_id.** Random 16-byte hex; will tie this SERP to clicks/purchases later.
2. **Embed the query.** `POST embedder:8000/embed {"text": "running shoes"}` → 384-dim normalized vector. On embedder failure the path degrades to BM25-only and `mode` in the response reflects this.
3. **Issue two retrievals in parallel.**
   - **BM25**: `multi_match` on `title^2`, `bullets`, `description` (size=200).
   - **k-NN**: Lucene HNSW over `title_vec` with cosine similarity (k=200).
4. **Fuse with RRF.** Client-side, in the Go process. For every doc that appeared in either result list:
   `rrf(d) = 1/(60 + rank_bm25(d)) + 1/(60 + rank_knn(d))`. Sort descending, take top-k.
5. **Build the response.** Each hit carries `_source` fields (title, brand, color, category, price) plus an `explain` block with both raw scores and ranks — these will be exactly the LTR retrieval features in Phase 4.
6. **(Phase 3+) Async impression logging.** A buffered channel batches impression events (one per returned product) and flushes to Postgres `search_events` every 100ms / 500 rows. Failures log but never block the response.

```text
client ──▶ /search ──▶ embed (sidecar) ──┬──▶ BM25 (OpenSearch)
                                          │
                                          └──▶ k-NN (OpenSearch)
                                          ▼
                                       RRF fuse  ──▶ rerank (Phase 5)  ──▶ JSON
                                          │
                                          └──▶ async impression buffer (Phase 3)
```

### Failure modes and degradation

| Failure | Behavior |
|---|---|
| Embedder sidecar unhealthy | Log + degrade to BM25-only; response sets `mode: "bm25"`. |
| BM25 query fails but k-NN succeeds | 500 (we treat retrieval as critical; a partial-success path can be added in Phase 7). |
| OpenSearch 5xx on either side | 500. |
| Reranker model fails to load (Phase 5) | Skip rerank; return RRF-sorted results unchanged. |

---

## Data path: catalog → indexed product

**ESCI → Postgres → OpenSearch (BM25) → OpenSearch (BM25 + k-NN).**

```
                      ┌──────────────────────────┐
                      │ Amazon ESCI parquet files│
                      │  (HuggingFace / GitHub)  │
                      └────────────┬─────────────┘
                                   │ download once
                                   ▼
                      ┌──────────────────────────┐
                      │   MinIO: drusearch-data  │
                      │       esci/*.parquet     │
                      └────────────┬─────────────┘
                                   │ pyarrow + filter (small/US/labeled)
                                   ▼
                      ┌──────────────────────────┐
                      │  pipelines.ingest.esci   │   subset 10,025 products
                      └────────────┬─────────────┘   18,265 judgments
                                   │ COPY
                                   ▼
                      ┌──────────────────────────┐
                      │ Postgres: products,      │
                      │ esci_judgments,          │
                      │ ingest_runs              │
                      └────────────┬─────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼ pipelines.index.bm25                    ▼ pipelines.embed.build_vectors
   ┌─────────────────────────┐              ┌──────────────────────────────────┐
   │ OpenSearch: products_v1 │              │ POST embedder:8000/embed_batch    │
   │  title, bullets, brand, │◀─────────────│  -> 384-d normalized vectors      │
   │  color, price, etc.     │              │ -> bulk-update doc.title_vec      │
   └─────────────────────────┘              └──────────────────────────────────┘
```

**Idempotency.** `ingest.esci` records each run in `ingest_runs` with a dataset hash; the indexer drops and recreates `products_v1` per run; the embedder writes via partial update (`_op_type: update`) so it never breaks BM25 fields.

---

## ML pipeline (Phase 4)

End-to-end flow for training the LTR reranker. Every step is one `python -m pipelines.<...>` and is idempotent.

```
┌──────────────────────┐    ┌─────────────────────────┐
│ search_events        │    │ esci_judgments          │
│ (impressions)        │    │ (E/S/C/I per query×prod)│
└──────────┬───────────┘    └────────────┬────────────┘
           │                             │
           │  pipelines.label            │
           │  .build_training_rows       │
           │  (joins by query text)      │
           ▼                             ▼
        ┌───────────────────────────────────┐
        │ training_rows (JSONB features +    │
        │ ESCI label 0/2/3/4 + train/val/test│
        │ split-by-query, 80/10/10)          │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │ pipelines.train.lgbm_ranker        │
        │   LightGBM LambdaRank              │
        │   label_gain=[0,1,3,7,15]          │
        │   group=query_id, NDCG@10 early    │
        │   stop                             │
        └────────────────┬───────────────────┘
                         │ mlflow.log_model + register
                         ▼
        ┌───────────────────────────────────┐
        │ MLflow registry: ltr_reranker      │
        │   versions tagged staging/prod      │
        │   artifact: model_text/model.txt   │
        │   (loaded by Go via leaves Phase 5)│
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │ pipelines.evaluate.offline_eval    │
        │   re-runs hybrid retrieval on 224  │
        │   ESCI test queries; reports       │
        │   NDCG@10 lift LTR vs RRF          │
        └────────────────────────────────────┘
```

### Feature schema (12 features in v1)

Defined in `pipelines/features/__init__.py`; this is the single source of truth for both training and (Phase 5) Go inference. Append-only ordering — never reorder.

| Feature | Source | Notes |
|---|---|---|
| `bm25_score`, `bm25_rank` | RETRIEVAL | from impression's recorded `retrieval_scores.bm25` |
| `knn_score`, `knn_rank` | RETRIEVAL | from `retrieval_scores.knn` |
| `rrf_score` | RETRIEVAL | from `retrieval_scores.rrf` |
| `popularity_prior`, `price_log_cents`, `title_length_tokens` | STATIC_PRODUCT | joined from `products` |
| `query_length_tokens`, `query_has_brand`, `query_has_color`, `query_has_size_pattern` | INTERACTION | deterministic functions of `(query, brand_set, color_set)` per `libs/schema/transforms.md` |
| `user_brand_affinity` | ONLINE_USER | per-request Redis HASH lookup; user's click-share for the candidate's brand; Phase 6 |

**Intentionally deferred to Phase 6:** `ctr_prior`, `purchase_rate`. With the synthetic click stream these features smuggle position bias into the training signal — see [Open lessons](#open-lessons) below.

### Training labels: ESCI, not clicks

`label.build_training_rows` joins each impression to `esci_judgments` by `(query_text, product_id)` and assigns:

| ESCI label | Training label | LightGBM gain |
|---|---|---|
| E (Exact)       | 4 | 15 |
| S (Substitute)  | 3 | 7 |
| C (Complement)  | 2 | 3 |
| I (Irrelevant) or unjudged | 0 | 0 |

13.2% of the 100k impression rows have a non-zero ESCI label; that's the supervised signal. The split is deterministic: `hash(query_id) % 100` → 80/10/10 train/val/test.

### Why ESCI labels and not clicks (lesson learned)

Initial Phase 4 attempt used the click signal directly: label = 2 if purchased, 1 if clicked, 0 otherwise. The synthetic-click LTR underperformed RRF by **−22 NDCG@10 points** on ESCI ground truth.

Root cause: in the position-based click model, P(click) is dominated by `examination(rank)` rather than relevance. So a click-trained LTR fits "what was at the top" rather than "what is relevant," and any extra lever (e.g. `ctr_prior`) just amplifies the position-bias circuit. The standard fix (IPS weights to deconfound from position) is on the Phase 7 list.

For v1, switching the label source to ESCI judgments is the cleanest experiment: now LTR learns a relevance signal directly. **Click data's role moves to personalization features in Phase 6** (per-user brand/category affinities in Redis).

### Results

224 ESCI test queries, 200-candidate RRF retrieval pool, k=10:

| Metric | RRF (baseline) | LTR (LightGBM) | Lift |
|---|---|---|---|
| NDCG@10  | 0.5506 | **0.5952** | **+0.0446** |
| NDCG@5   | 0.5420 | **0.6194** | +0.0774 |
| MRR      | 0.7099 | **0.8038** | +0.0939 |
| Recall@10 | 0.4118 | **0.4156** | +0.0038 |

Both runs are logged to MLflow under the `drusearch-eval` experiment.

---

## Data model

### Postgres — current tables

```
products            10,025 rows  catalog source of truth
esci_judgments      18,265 rows  E/S/C/I labels for offline eval
search_events       (Phase 3)    append-only impression/click/purchase log
user_sessions       (Phase 3)    lightweight session metadata
training_rows       (Phase 4)    one row per (query, product) impression with features + label
product_features    (Phase 4)    nightly aggregates (CTR, position-corrected CTR, etc.)
ingest_runs                      bookkeeping for idempotent pipeline runs
```

Schema lives in `infra/postgres/001_init.sql`; Postgres applies it on first boot via the `docker-entrypoint-initdb.d` mount.

### OpenSearch index `products_v1`

| Field | Type | Source |
|---|---|---|
| `product_id` | keyword | Postgres |
| `title`, `description`, `bullets` | text (BM25, English analyzer + stemmer) | Postgres |
| `brand`, `color`, `category` | keyword | Postgres |
| `price_cents`, `popularity_prior`, `ctr_prior` | int / float | Postgres |
| `title_vec` | knn_vector(384, HNSW, cosinesimil, Lucene) | embedder sidecar |

Mapping is templated in `infra/opensearch/index_template.json`; the indexer applies it before bulk-loading.

---

## APIs

```
GET  /healthz                            -> 200 always (process is up)
GET  /readyz                             -> 200 if PG, Redis, OpenSearch, embedder healthy

GET  /search?q=&k=&user_id=&session_id=  -> hybrid retrieval (RRF). Returns
                                            {query_id, query, mode, results[
                                              {product_id, title, brand, color, category,
                                               price_cents, score, explain{
                                                 bm25, bm25_rank, knn, knn_rank, rrf}}
                                            ], took_ms}

GET  /products/{id}                      -> full product record / 404

POST /events                             -> 202 (Phase 3)
POST /admin/reload-model                 -> Phase 5
POST /admin/reindex                      -> Phase 1+ (kicks off Prefect flow)
```

`mode` reflects whether the request used `hybrid` retrieval or degraded to `bm25` (e.g. embedder unreachable). `explain` is included for every hit so offline eval and the LTR feature pipeline can use exactly the same scores the runtime saw.

Auth on `/admin/*` requires a shared bearer token from `ADMIN_TOKEN`. Pass as `Authorization: Bearer $ADMIN_TOKEN` or `X-Admin-Token: $ADMIN_TOKEN`. If `ADMIN_TOKEN` is unset the server refuses all admin requests with `503` — no implicit-open mode.

---

## Feature parity (the hardest design problem)

Phase 4+ trains a LightGBM LambdaRank model in Python and serves it from Go. The same numerical features must be produced both at train time and at request time, or the trained model is meaningless. Strategy:

1. **`libs/schema/feature_schema.json`** is the single source of truth for the 13-feature schema. Each feature has a stable `index` (append-only), a `kind` (`FLOAT | INT | BOOL`), a `source` (`RETRIEVAL | STATIC_PRODUCT | PRODUCT_AGG | ONLINE_USER | ONLINE_SESSION | INTERACTION`), and a `description`. `libs/schema/features.proto` describes the wire types for any future RPC carriage.
2. **`libs/schema/codegen.py`** reads the JSON and writes language-specific schema files: `pipelines/pipelines/features/_generated.py` and `services/api-go/internal/features/schema_generated.go`. The hand-written `__init__.py` and `schema.go` assert equality with the generated files at import / test time so silent drift breaks the build. `make check-feature-parity` regenerates and `git diff --exit-code`s; CI runs it on every push.
3. **STATIC_PRODUCT and RETRIEVAL features** are read identically from OpenSearch `_source` / hit objects on both sides. Zero re-computation.
4. **ONLINE features** (Phase 6) live in Redis HASHes (`feat:user:{id}`, `feat:session:{id}`). Python writes them; Go reads them.
5. **INTERACTION features** are the danger zone: pure functions of `(query, product, user_features)`. Each is specced in `libs/schema/transforms.md` with Python (`pipelines/pipelines/features/transforms.py`) and Go (`services/api-go/internal/features/transforms.go`) reference implementations, plus shared fixtures in `libs/schema/fixtures/interaction_fixtures.json`. Two parity tests — `pipelines/tests/test_interaction_parity.py` and `services/api-go/internal/features/parity_test.go` — load the same JSON file and assert byte-equal output. CI runs both.

---

## Latency budget

| Stage | p50 (target) | p99 (target) | Phase 2 measured (cold) |
|---|---|---|---|
| Embedder sidecar | 6 ms | 15 ms | 5–15 ms |
| OpenSearch BM25 + k-NN (parallel) | 8 ms | 25 ms | 30–80 ms cold (warmup) |
| RRF fuse | 1 ms | 3 ms | <1 ms |
| Feature build (Phase 4+) | 2 ms | 5 ms | — |
| LightGBM rerank (Phase 5) | 3 ms | 8 ms | — |
| Serialize / HTTP | 2 ms | 5 ms | 1–3 ms |
| **End-to-end** | **~25 ms** | **<70 ms** | **100–280 ms cold; settles into target band warm** |

Cold-path numbers reflect first queries after a fresh container start (JVM warmup, embedder model compile). Steady-state hits the budget.

---

## Operational notes

- **Bring-up:** `cp .env.example .env && make up && make ready`
- **Demo Phase 1:** `make seed-catalog && make index-bm25 && curl 'localhost:8080/search?q=running+shoes'`
- **Demo Phase 2:** `make embed-vectors && curl 'localhost:8080/search?q=something+to+keep+my+coffee+hot&k=5'`
- **Inspect index:** `curl 'localhost:9200/products_v1/_count'`
- **Inspect model artifacts:** MLflow UI at `http://localhost:5000`, MinIO at `http://localhost:9001`.
- **Disk threshold disabled.** Single-node OpenSearch on a constrained disk would otherwise auto-block writes once usage crosses 95%; see `docker-compose.yml` env override.
- **API healthcheck.** The distroless API image has no shell, so the docker healthcheck is `["/app/api", "healthcheck"]`, which performs an internal HTTP call against `/healthz`.

---

## Personalization (Phase 6)

A per-user feature, `user_brand_affinity`, joins the LTR ranker as the 13th feature. It's a per-row signal: for each candidate at score-time, the value is the user's historical click-share for that candidate's brand.

```
                                         clicks                ┌────────────────────────┐
   ┌────────────────┐    user_aggs       (Postgres)            │   /search request      │
   │  search_events │  ───────────────►  count(user, brand) /  │   ?user_id=u_00069     │
   │  (clicks)      │                    (total + 1)           └─────────┬──────────────┘
   └────────────────┘                       │                            │
                                            ▼                            ▼
                          ┌─────────────────────────────────┐  features.LoadUserFeatures
                          │ Redis HASH feat:user:{user_id}  │  HGETALL feat:user:u_00069
                          │   brand_aff:Nike:    0.182      │ ────────────────────────►
                          │   brand_aff:adidas:  0.030      │
                          │   ...                           │
                          └─────────────────────────────────┘
                                                                          │
                                                                          ▼
                                                         BuildMatrix per row:
                                                           feature[12] = brandAff[hit.Brand]
                                                                          │
                                                                          ▼
                                                          LightGBM PredictDense
                                                                          │
                                                                          ▼
                                                          Sort by predicted score → top-K
```

Demonstration (live `/search`, query="running shoes", k=5):

| User | Click history | Nike rank | Nike LTR score | Boost vs anonymous |
|---|---|---|---|---|
| `u_00069` | 4 Nike clicks | **1** | **+1.702** | **+0.760** |
| `u_00030` | 3 PUMA clicks (no Nike) | 3 | +0.942 | 0 |
| anonymous | — | 3 | +0.942 | (baseline) |

Per-request cost: one Redis HGETALL (~1 ms), one map lookup per candidate. No code path changes when the user is anonymous — `LoadUserFeatures` returns an empty snapshot and the feature reads as 0 for every row.

Currently surfaced: `user_brand_affinity`. Future Phase 6 features (queued behind IPS-corrected click features in Phase 7): `user_color_affinity`, `user_avg_price_clicked`, `session_clicks_so_far`, `session_avg_position_clicked`.

---

## Open lessons

- **Click labels need IPS weighting.** The first Phase 4 attempt used clicks-as-labels and was −22 NDCG points worse than RRF; the model learned position bias, not relevance. We sidestepped this by training on ESCI judgments. A proper IPS-weighted click-LTR (Phase 7) is the long-term fix and would let us reintroduce `ctr_prior` and `purchase_rate` as features.
- **Cold p99 still spikes (~280 ms first request).** JVM warmup + first-request embedder load. Not a real concern at this scale; would matter under autoscaling.
- **Disk-based shard allocation watermark is disabled.** Single-node OpenSearch on a constrained dev disk would otherwise auto-block writes — see `docker-compose.yml`. Real production would size disk correctly and re-enable.

---

## Roadmap

| Phase | Status | Demo |
|---|---|---|
| 0 — Scaffold + compose | ✅ done | `make up && curl /readyz` |
| 1 — BM25 search | ✅ done | `curl '/search?q=running+shoes'` |
| 2 — Hybrid (BM25 + k-NN + RRF) | ✅ done | `curl '/search?q=something+to+keep+my+coffee+hot'` |
| 3 — Event ingest + click simulator | ✅ done | 100k impressions / 15.8k clicks / 1.7k purchases in `search_events` |
| 4 — LTR training + offline eval | ✅ done | LTR NDCG@10 0.5952 vs RRF 0.5506 (+0.045) on 224 ESCI test queries |
| 5 — LTR serving in Go | ✅ done | `mode="hybrid+ltr"` in `/search`, `/admin/reload-model` reloads from disk, Go↔Python prediction parity = 0 (bit-perfect across leaves vs lightgbm) |
| 6 — Personalization + retraining loop | ✅ done | `feat:user:{id}` HASH in Redis populated by user_aggs; Go reranker reads it per request; demonstrated +0.76 LTR-score boost on Nike for a Nike-loving user vs anonymous |
| 7 — Production polish | ✅ done | `/metrics` exposes per-stage histograms + circuit-breaker gauge + model-loaded gauge; embedder breaker trips OPEN after 5 fails and degrades `/search` to `mode="bm25+ltr"` (verified: 6 requests OK at 45ms); `pipelines.register.gate` blocks promotions on NDCG regression; `docs/RUNBOOK.md` covers bring-up + retrain + common failures |
