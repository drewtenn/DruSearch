# Search Performance Scale Design

## Goal

Improve DruSearch so the local demo can rehearse production-scale search behavior: millions of product-shaped records, horizontally runnable API replicas, predictable latency under load, and operational patterns that can later move to managed infrastructure without redesigning the application.

This design keeps Docker Compose as the primary developer experience. It does not try to run a true multi-node production cluster on a laptop. Instead, it introduces production-shaped boundaries, config, data flows, aliases, caches, degradation behavior, and observability while preserving local usability.

## Current Baseline

DruSearch currently serves search through a Go API backed by OpenSearch, Redis, Postgres, a Python embedder, and Python batch pipelines. The hot path embeds the query, runs BM25 and kNN retrieval in parallel against `products_v1`, fuses candidates with RRF in the API, optionally applies a LightGBM reranker, returns the top results, and logs impressions asynchronously.

The architecture documentation currently sizes this for about 10k products and less than 50 QPS on one local machine. The OpenSearch index template uses one shard and zero replicas. The API is mostly stateless already, but several scale decisions are still hardcoded: candidate count, index name, single-index lifecycle, strict retrieval failure handling, and no query-result or embedding cache.

## Recommended Approach

Use a scale-rehearsal architecture.

The system remains local-first, but each component behaves like a smaller version of its production counterpart:

- OpenSearch uses versioned indexes behind read/write aliases.
- Indexing is blue-green: build a new index, validate it, warm it, then atomically promote aliases.
- Retrieval settings are configurable rather than hardcoded.
- The API remains stateless and can run multiple replicas.
- Redis stores hot-path caches and online user features.
- The event path remains asynchronous and bounded.
- Load tests and metrics show whether the local system is staying within latency budgets.

This avoids premature service fragmentation while removing single-node assumptions from application code.

## Architecture

### Index Lifecycle

Replace the fixed runtime dependency on `products_v1` with aliases:

- `products_read`: alias used by the API for search.
- `products_write`: alias used by indexing jobs when they need a stable write target.
- `products_v<N>`: immutable versioned backing indexes, such as `products_v2`.

The local default remains one shard and zero replicas. The template exposes shard count, replica count, refresh interval, kNN HNSW parameters, and source fields through configuration so production deployments can change them without application code changes.

Index rebuild flow:

1. Create the next versioned index with local or production-like settings.
2. Temporarily set refresh interval and replicas for bulk loading.
3. Bulk index products in bounded chunks.
4. Build or attach vectors.
5. Validate document count, mapping, vector coverage, and sample queries.
6. Warm representative queries.
7. Atomically move `products_read` to the new index.
8. Keep the previous index for rollback until cleanup.

### Retrieval Hot Path

The API should use configurable retrieval settings:

- Candidate count for BM25.
- Candidate count for kNN.
- RRF rank constant.
- Search timeout.
- Source field list for retrieval.
- Whether partial retrieval is enabled.

BM25 and kNN continue to run in parallel. Failure behavior changes from all-or-nothing to graceful partial success:

- BM25 and kNN both succeed: return `hybrid` or `hybrid+ltr`.
- Embedder fails or circuit is open: return `bm25` or `bm25+ltr`.
- BM25 succeeds and kNN fails: return `bm25` or `bm25+ltr`.
- kNN succeeds and BM25 fails: return `knn` or `knn+ltr`.
- Both retrieval paths fail: return HTTP 500.

Each stage gets an explicit request deadline so a slow dependency cannot consume the entire request budget.

### Caching

Redis should support two hot-path caches:

- Query embedding cache keyed by normalized query and embedding model version.
- Anonymous candidate cache keyed by normalized query and retrieval config version.

The candidate cache stores product IDs and retrieval scores before personalization. The API can then apply user features and LTR reranking on top. This keeps popular anonymous queries fast without freezing personalized ranking.

Cache entries use short TTLs by default in local mode. Cache metrics expose hit rate, miss rate, and stale/config-version bypasses.

### API Horizontal Scalability

The Go API remains stateless across replicas. Request-critical state must live in OpenSearch, Redis, Postgres, or model artifacts mounted/read by each replica.

Local Compose should document how to run more than one API instance. The API must not assume that in-memory queues, query IDs, or model handles are globally unique beyond the current process. Impression logging remains async, but the queue must be bounded and observable:

- Configurable buffer size.
- Configurable flush interval.
- Configurable batch size.
- Drop counter when the queue is full.
- Stage metrics for event enqueue and flush.

### Data Pipeline Scalability

The indexing pipeline should be able to generate and index larger local catalogs. It does not need to store millions of hand-curated products in the repo. Instead, add synthetic catalog scaling that creates product-shaped records by varying title, brand, category, color, price, popularity, and description fields.

The pipeline should write in bulk chunks and avoid refreshing the index per document. Large local runs should be possible with a smaller vector mode or optional vector skip, so lexical/index lifecycle performance can be tested separately from embedding throughput.

### Observability and Load Testing

The system should expose enough metrics to see where search time is going:

- End-to-end search latency by mode.
- Stage latency for embedding, embedding cache, retrieval, BM25, kNN, Redis feature load, candidate cache, rerank, serialization, and event enqueue.
- Candidate count per request.
- OpenSearch error count by retrieval leg.
- Cache hit/miss counters.
- Partial-degradation counters.
- Event queue depth and dropped event counter.

Add repeatable local load-test targets that exercise:

- A small catalog smoke test.
- A larger synthetic catalog test.
- Popular query cache behavior.
- Mixed anonymous and personalized requests.

## Components

### Go API

Responsibilities:

- Read search performance config from environment.
- Search via `products_read` by default.
- Use request-scoped deadlines.
- Run BM25 and kNN in parallel.
- Support partial retrieval degradation.
- Read/write Redis caches.
- Preserve LTR reranking and user feature loading.
- Emit new metrics.

The API should stay deployable as multiple identical replicas.

### OpenSearch Infrastructure

Responsibilities:

- Provide an alias-aware index template.
- Support local and production-shaped index settings.
- Keep kNN mapping compatible with the existing embedder dimension.
- Keep source fields minimal for search.

OpenSearch remains single-node in default local Compose. The application code should not depend on that fact.

### Python Pipelines

Responsibilities:

- Create versioned indexes.
- Bulk index products in chunks.
- Build vectors in batches.
- Validate and warm new indexes.
- Promote aliases atomically.
- Optionally generate synthetic product catalogs for load tests.

### Redis

Responsibilities:

- Continue serving online user features.
- Store embedding cache entries.
- Store anonymous retrieval candidate cache entries.
- Keep all cache keys versioned so index/model/config changes do not reuse incompatible entries.

### Documentation and Runbook

Responsibilities:

- Explain local scale-rehearsal mode.
- Document production knobs and their local defaults.
- Document blue-green reindex and rollback.
- Document how to run multiple API replicas locally.
- Document load-test commands and expected interpretation.

## Non-Goals

- Do not introduce Kubernetes in this phase.
- Do not split retrieval/ranking into separate network services in this phase.
- Do not replace OpenSearch with a separate vector database.
- Do not require a laptop to run a genuine multi-node OpenSearch cluster.
- Do not guarantee that the default local setup can hold millions of full vector documents in memory.

## Testing Strategy

The implementation should include unit tests for config parsing, alias naming, cache keys, and partial retrieval behavior. Retrieval body tests should assert that source filtering, timeouts, aliases, and candidate counts are used correctly.

Pipeline tests should validate index naming, alias promotion payloads, synthetic catalog generation, and bulk chunking behavior without requiring a live OpenSearch cluster when possible.

Integration smoke tests should continue to verify that a local catalog can be indexed and searched. Load tests should be separate opt-in targets because they are slower and machine-dependent.

## Success Criteria

- The API searches through `products_read` rather than a hardcoded concrete index.
- Reindexing can build a new versioned index and promote it through aliases.
- Retrieval candidate counts, timeouts, index alias, and partial-degradation behavior are configurable.
- Query embedding cache and anonymous candidate cache are available through Redis.
- The API can be run as multiple local replicas without code changes.
- Local load-test commands report p50, p95, and p99 latency by mode.
- Documentation explains the gap between local defaults and production settings for millions of products and users.

## Future Production Path

When the demo moves to production, the same boundaries map cleanly:

- Docker Compose services become managed or orchestrated services.
- OpenSearch becomes a multi-node cluster with production shard and replica settings.
- Redis becomes a clustered or managed cache.
- The API scales horizontally behind a load balancer.
- Pipelines run as scheduled jobs or workflow tasks.
- Event ingestion can move from direct Postgres writes to a stream such as Kafka, Kinesis, or Pub/Sub.

The code should already speak in aliases, timeouts, configs, caches, and idempotent index promotions before that migration.
