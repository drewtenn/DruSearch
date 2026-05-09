# DruSearch — Product Requirements

> **Status:** v0.1 (living doc)
> **Owner:** drewtenn
> **Last updated:** 2026-05-07
> **Companion docs:** [ARCHITECTURE.md](./ARCHITECTURE.md)

## Summary

DruSearch is a personalized product search service for an e-commerce site. Given a free-text query and (optionally) a user/session identifier, it returns a ranked list of products that improves with click and purchase data. The system is built to be production-realistic on day one — every component (search engine, model registry, feature store, async event ingest) is one a real e-commerce team would actually run — but sized to operate on a single machine for a 10k-product catalog.

This is intentionally a learning-grade build of a production-grade architecture, not a toy and not a re-skin of an off-the-shelf SaaS.

## Problem statement

Naive product search (a single BM25 query against a title field, sorted by relevance) leaves significant revenue on the table:

1. **Lexical-only retrieval misses semantic matches.** A query for "wireless earbuds for running" should surface sweat-resistant athletic Bluetooth audio even when the product title doesn't contain "running."
2. **Static ranking ignores behavior.** Two shoppers with identical queries should not see identical results if one has a clear preference for a brand, color, or price band.
3. **No feedback loop.** Without an event pipeline, the system can never improve from production traffic.

DruSearch addresses all three: hybrid retrieval (BM25 + dense embeddings), Learning-to-Rank reranking with click/purchase signals, and a feature store that closes the loop in a way that's safe to retrain against.

## Goals

| # | Goal | How we'll know we hit it |
|---|---|---|
| G1 | Hybrid retrieval that beats BM25-only | NDCG@10 on ESCI test split improves vs BM25-only baseline. |
| G2 | Click-driven LTR reranker that beats hybrid retrieval | NDCG@10 of (BM25 + k-NN + LTR) > NDCG@10 of (BM25 + k-NN, RRF). |
| G3 | Personalization that beats non-personalized LTR | NDCG@10 with per-user features > NDCG@10 without, on a held-out simulated cohort. |
| G4 | Production-real serving path | p99 search latency < 100 ms at <50 QPS on one machine; degraded mode (BM25-only) when the embedder is unhealthy. |
| G5 | Reproducible & retrainable pipeline | A single command rebuilds the index, retrains the model, and promotes it. Every model in MLflow is reproducible from a recorded dataset hash. |
| G6 | No silent feature drift between training and serving | Cross-language parity tests pass; CI fails on divergence. |

## Non-goals

- **Multi-tenant SaaS.** One catalog, one instance.
- **Multi-locale.** US English only in v1. ESCI has ES/JP product subsets we deliberately filter out.
- **Multi-modal search** (image queries, voice). Text only.
- **Deep personalization beyond category/brand affinity.** No collaborative filtering, no sequential recommendation. Affinity counters drift with clicks; that's it.
- **A user-facing UI.** The API is the product. A small dev console at `/admin/*` is acceptable but not designed.
- **Authentication / authorization** beyond a static admin token. No user login flow, no per-merchant ACLs.

## Users & use cases

| Persona | What they do | What they need |
|---|---|---|
| **Shopper (primary)** | Types a query, scans the SERP, clicks, possibly buys | Relevant top-K results; consistent fast response (<100ms p99); results that respect their evident preferences. |
| **Catalog ingestion job** | Pushes the latest catalog into Postgres + OpenSearch | An idempotent `make seed-catalog && make index-bm25 && make embed-vectors` that can run repeatedly. |
| **ML engineer (the user)** | Iterates on retrieval, features, the model | Clean MLflow lineage; comparable offline metrics; one-command retrain; safe rollback via model registry stages. |
| **SRE-on-call (future)** | Diagnoses a regression in latency or relevance | `/healthz`, `/readyz`, `/metrics`, structured logs, a dashboard, and a runbook. |

## Functional requirements

### F1. Search

- `GET /search?q=<text>&user_id=<opt>&session_id=<opt>&k=<1..100>` → top-k ranked results.
- Each hit returns `product_id`, `title`, `brand`, `color`, `category`, `price_cents`, `score`, and an `explain` block exposing every retrieval component (`bm25`, `bm25_rank`, `knn`, `knn_rank`, `rrf`, and after Phase 5, `ltr`).
- Empty query → 400. Missing/unknown user_id → still serves (treated as a non-personalized request).
- Response includes a `query_id` so subsequent click events can be tied back to the SERP.

### F2. Event ingest (Phase 3+)

- `POST /events` accepts impression, click, and purchase events.
- Server-side validation: `event_type ∈ {impression, click, purchase}`, presence of `query_id`, `product_id`, `position`. Ingest is fire-and-forget (202).
- Impressions are also logged automatically by `/search`, asynchronously, never blocking the response.
- All events land in Postgres `search_events` as the durable log.

### F3. Product lookup

- `GET /products/{id}` → full product record. 404 if unknown.

### F4. Catalog & vector pipeline

- One-command ingestion of the Amazon ESCI dataset (10k product subset).
- Idempotent BM25 indexer that drops/recreates the OpenSearch index.
- Embedding pipeline that streams Postgres products through the embedder sidecar and writes vectors back via partial document update — so re-embedding never erases BM25 fields.

### F5. ML pipeline (Phase 4+)

- Training-row builder that turns raw events into `(query, product, features, label)` tuples, split by query.
- LightGBM LambdaRank training with NDCG@10 early stopping; results logged to MLflow.
- Auto-tag of new runs as `staging`; `production` requires either a manual transition or an eval-script gate that asserts non-regression.
- Model artifacts written to MinIO via MLflow's S3 backend.

### F6. Model serving (Phase 5+)

- Go API loads the `production` model on boot and on `/admin/reload-model`.
- LightGBM scoring via `dmitryikh/leaves` in-process — no subprocess, no extra hop.
- Reranker failure is non-fatal: skip rerank, return RRF order.

### F7. Personalization (Phase 6+)

- Nightly aggregation jobs build per-user category & brand affinities and push to Redis.
- The Go API joins Redis features into the candidate feature vectors at request time.
- Same user, same query, with different historical click patterns → demonstrably different SERP ordering.

## Non-functional requirements

### Performance

| Metric | Target |
|---|---|
| Search latency p50 | ≤ 25 ms |
| Search latency p99 | ≤ 70 ms (cold burst tolerance: < 300 ms for first 1–5 requests after cold start) |
| Throughput | ≥ 50 QPS sustained on a single machine |
| Embedding throughput | ≥ 200 docs/sec on CPU (10k catalog re-embed in < 2 min) |
| Index refresh | ≤ 5 s after `make index-bm25` |

### Scale

- Catalog: 10k products in v1; architecture sized to scale to 1M without changing components (only resource sizing).
- Active users: simulator drives 1k synthetic users; real-user scale not formally targeted.
- Event log: design for retaining 30 days of events in Postgres; `pg_partman` or archival to MinIO is a Phase 7 consideration.

### Reliability

- All stateful services (Postgres, OpenSearch, Redis, MinIO, MLflow) run with named volumes.
- The Go API degrades gracefully when the embedder is down (BM25-only mode advertised in the response `mode` field).
- All pipelines are idempotent and resumable; re-running with the same seed is a safe no-op.
- A model load failure does not crash the API.

### Security

- Out of scope: end-user authentication, encryption-at-rest, multi-tenant isolation.
- In scope: `/admin/*` is gated by `ADMIN_TOKEN` in production; not exposed externally in dev.
- Secrets via environment variables only; never committed.

### Observability

- Structured JSON logs from the Go API (Zap) with request id correlation.
- Prometheus `/metrics` (Phase 7): per-stage latency histograms, retrieval candidate counts, model-version-loaded gauge, event ingestion lag, embedder circuit-breaker state.
- Per-stage timing exposed on `/search` responses (`took_ms`).

### Maintainability

- One-language hot path (Go), one-language pipelines (Python).
- Single source of truth for the LTR feature schema (`libs/schema/features.proto`) with codegen for both languages and CI parity tests on interaction features.
- Changes follow the phase plan; phases are independently demoable so a partial rollout is safe.

## Success metrics

### Offline

- **NDCG@10 on ESCI test split.** Recorded for every model promotion.
  - Phase 2 baseline (BM25 + k-NN + RRF) is the floor.
  - Phase 4 LTR must beat the floor.
  - Phase 6 personalized LTR must beat Phase 4 on a held-out simulated cohort.
- **MRR**, **Recall@K** (K ∈ {10, 50, 100}) tracked alongside.
- Cross-language interaction-feature parity tests: 100% pass rate in CI.

### Online (when there is real traffic)

- Click-through rate (CTR) on the top result.
- Click-through rate at any position in the top-K.
- Purchase-rate-per-search.
- Tail-query coverage (queries with at least one click in the top-10).

### System

- p99 search latency.
- Model promotion frequency.
- Event ingestion lag (time from `/events` 202 to durable in Postgres).

## Constraints & decisions made

These are decisions already adopted; revisiting them requires explicit justification.

| Decision | Rationale | Relitigation cost |
|---|---|---|
| Local-first via Docker Compose | Learning-friendly, identical to `docker compose up` in any environment | Low; AWS/GCP port is straightforward. |
| Go API + Python ML | Best-in-class latency for serving + best-in-class library for ML | Medium; would force unifying the feature implementation in one language. |
| Hybrid retrieval (BM25 + dense + RRF) | Industry standard for modern e-commerce search | High; the entire ranking stack assumes this. |
| OpenSearch for both BM25 and k-NN | Avoids running a separate vector DB | Medium; pgvector is a viable alternative. |
| LightGBM LambdaRank | Strong tabular baseline, fast inference, mature tooling | Medium. |
| `BAAI/bge-base-en-v1.5` (768-d) | Stronger general-purpose BGE embedder for semantic retrieval | Medium; requires rebuilding the OpenSearch index vectors. |
| Synthetic click simulator that calls the real API | Production-shape dogfooding from day one | Low; replaceable with real event replay. |
| Protobuf feature schema + codegen | Compile-time errors when adding features in the LTR vector | Medium. |

## Out of scope (explicitly)

- Faceted filters and sorts (in-stock, ship-by, price range, brand chips). Plumbed via `_source` fields but not surfaced in `/search`.
- Spelling correction / query rewriting / synonyms.
- Diversification, business rules, freshness boosts, promoted listings.
- Multi-armed-bandit exploration during ranking.
- Online learning / streaming model updates.

These are real production search concerns; they are explicitly deferred so the LTR loop gets attention first.

## Risks & open questions

| ID | Risk / question | Mitigation / next step |
|---|---|---|
| R1 | `dmitryikh/leaves` may not parse the `lambdarank` objective string. | Verified at Phase 5 with a fixture. Mitigation: rewrite metadata to `regression` in `pipelines.register.promote` (trees unchanged; LambdaRank outputs raw scores). |
| R2 | Larger BGE models have not been A/B'd on ESCI specifically. | Offline eval measures NDCG@10 and recall diagnostics before further model-size changes. |
| R3 | Position-bias correction in training labels (IPS weights). | Disabled in v1; revisit at Phase 7 once we have enough simulated traffic. |
| R4 | OpenSearch native RRF (`hybrid` query + score-ranker pipeline). | Currently using client-side RRF in Go. Native pipeline can replace it without API changes; deferred until there's a measurable reason. |
| R5 | Single-node OpenSearch is the only HA-relevant component. | Acceptable for the local-first goal. Production deployment would shift to a managed cluster. |
| R6 | Synthetic click data may bias the model toward simulator quirks. | Sanity-check every model against ESCI labels (real human relevance), not only synthetic CTR. |
| R7 | Adding `user_brand_affinity` shaves ~3 NDCG@10 points off the anonymous-user ESCI eval (model spends capacity on a feature that's 0 in anonymous queries). | Mitigate by training with a fraction (~30%) of rows masked to `user_id=NULL` so the model learns to ignore the feature when missing. Plumbed in Phase 7 alongside IPS weighting. |

## Milestones

| Phase | Status | What ships |
|---|---|---|
| 0 — Scaffold + compose | ✅ done | All seven services healthy; `/readyz` 200. |
| 1 — BM25 search | ✅ done | 10k ESCI products in PG + OpenSearch; `/search` returns BM25 results. |
| 2 — Hybrid retrieval | ✅ done | k-NN over `title_vec`; client-side RRF fusion; degrades to BM25 if embedder is down. |
| 3 — Events + click simulator | ✅ done | `POST /events` durable in PG; simulator wrote 100,003 impressions / 15,763 clicks / 1,663 purchases against the real API. CTR 15.8%, position-decay matches PBM. |
| 4 — LTR training | ✅ done | LTR (BM25+kNN+RRF + product/query feats) beats RRF on ESCI: NDCG@10 0.595 vs 0.551 (+4.5pt), NDCG@5 0.619 vs 0.542, MRR 0.804 vs 0.710. Trained on ESCI labels, not synthetic clicks. |
| 5 — LTR serving | ✅ done | Go reranker via dmitryikh/leaves loads model.txt promoted to a shared volume; `/admin/reload-model` reloads atomically; `mode="hybrid+ltr"` advertised in `/search`; explain.ltr_score / explain.ltr_rank exposed; Go↔Python prediction parity bit-perfect (0.0e+00 max abs diff). |
| 6 — Personalization | ✅ done | `pipelines.features.user_aggs` writes per-user brand-share to `feat:user:{id}` Redis HASH; Go feature builder fetches per request and feeds `user_brand_affinity` into LTR; Nike-loving user u_00069 ranks Nike #1 (ltr=+1.702) vs rank #3 for anonymous (ltr=+0.942) on the same `running shoes` query. ESCI NDCG@10 with anonymous user_id=0: 0.5668 vs RRF 0.5506 (+0.016) — small regression vs v2 (0.5952) traceable to dead-weight feature in non-personalized eval; mitigation listed under R7. |
| 7 — Production polish | ✅ done | Prometheus `/metrics` (search/stage/event/model-loaded histograms+counters+gauges); embedder circuit breaker (sony/gobreaker, trips after 5 consecutive fails, half-open probe after 10s, degrades `mode` to `bm25+ltr`); promotion gate `pipelines.register.gate` blocks regressions and transitions versions to `Production` stage; PRD-R7 mitigated via 30% anonymous mask in training rows; `docs/RUNBOOK.md` shipped. v4 (anon-mask) ESCI NDCG@10 = 0.5731 vs RRF 0.5506 (+0.0225) and now in `Production` stage. |

## Approval / change log

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-07 | Initial draft. Phases 0–2 complete; 3–7 planned. |
