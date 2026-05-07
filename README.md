# DruSearch

Hybrid e-commerce search service: BM25 + dense embeddings + LightGBM Learning-to-Rank reranker, with click-driven personalization.

- **Go API** (search, events, admin)
- **Python embedder sidecar** (sentence-transformers)
- **Python ML pipelines** (ingest, index, embed, simulate, label, train, evaluate)
- **OpenSearch** (BM25 + k-NN), **Postgres**, **Redis**, **MLflow**, **MinIO**

See `/root/.claude/plans/i-want-to-create-typed-sprout.md` for the full plan.

## Quick start

```bash
cp .env.example .env
make up            # docker compose up -d
make ready         # wait for /readyz
```

## Phases

| Phase | Status | Demo |
|---|---|---|
| 0 — Scaffold + compose | in progress | `make up && curl localhost:8080/readyz` |
| 1 — BM25 search | — | `curl 'localhost:8080/search?q=running+shoes'` |
| 2 — Hybrid retrieval (BM25 + k-NN) | — | NDCG@10 lift on ESCI |
| 3 — Event ingest + click simulator | — | events in Postgres |
| 4 — LTR training + offline eval | — | NDCG@10 lift in MLflow |
| 5 — LTR serving in Go | — | `/admin/reload-model` |
| 6 — Personalization + retraining loop | — | per-user re-ranking |
| 7 — Production polish | — | metrics, circuit breaker, runbook |
