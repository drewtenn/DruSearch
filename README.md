# DruSearch

Hybrid e-commerce search service: BM25 + dense embeddings + LightGBM Learning-to-Rank reranker, with click-driven personalization.

- **Go API** (search, events, admin)
- **Python embedder sidecar** (sentence-transformers)
- **Python ML pipelines** (ingest, index, embed, simulate, label, train, evaluate)
- **OpenSearch** (BM25 + k-NN), **Postgres**, **Redis**, **MLflow**, **MinIO**

## Quick start

```bash
cp .env.example .env
make up            # docker compose up -d
make ready         # wait for /readyz

# Load demo data before calling /search
make seed-databases # ESCI -> Postgres + OpenSearch products_v1

# Optional: add dense vectors for the k-NN side of hybrid retrieval
make embed-vectors
```

`make ready` checks that the services are reachable. It does not load the catalog or create the `products_v1` search index; run `make seed-databases` on a fresh stack before using `/search`.

## API

The Go API listens on `http://localhost:8080` when started with `make up`. All JSON responses use `Content-Type: application/json` unless noted otherwise.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Process liveness. |
| `GET` | `/readyz` | Dependency readiness for Postgres, Redis, OpenSearch, and the embedder. |
| `GET` | `/metrics` | Prometheus metrics. |
| `GET` | `/search` | Hybrid product search with optional user personalization and LTR reranking. |
| `POST` | `/events` | Record impression, click, or purchase feedback. |
| `GET` | `/products/{id}` | Fetch one product by ID. |
| `POST` | `/admin/reload-model` | Reload the promoted LightGBM model from disk. |
| `POST` | `/admin/reindex` | Reserved admin endpoint; currently returns `501`. |

### Health and readiness

```bash
curl -fsS http://localhost:8080/healthz
curl -fsS http://localhost:8080/readyz
```

`/readyz` returns `200` only when all dependencies are healthy:

```json
{
  "status": "ok",
  "postgres": true,
  "redis": true,
  "opensearch": true,
  "embedder": true
}
```

### Search

Search requires the catalog to be loaded and indexed. On a fresh stack, run `make seed-databases` first; run `make embed-vectors` as well to populate `title_vec` for k-NN scoring.

```bash
curl -G http://localhost:8080/search \
  --data-urlencode "q=running shoes" \
  --data-urlencode "k=5" \
  --data-urlencode "user_id=user-123" \
  --data-urlencode "session_id=session-abc"
```

Query parameters:

| Name | Required | Default | Notes |
|---|---:|---|---|
| `q` | yes | - | Search text. Blank or missing values return `400`. |
| `k` | no | `20` | Number of results. Values above `100` are ignored. |
| `user_id` | no | empty | Enables per-user affinity features when known. |
| `session_id` | no | generated anonymous ID | Echoed in the response and used for impression events. |

Response:

```json
{
  "query_id": "7f33c93a6cb94836a97728582ff0aa63",
  "query": "running shoes",
  "session_id": "session-abc",
  "mode": "hybrid+ltr",
  "model_version": "ltr_reranker",
  "results": [
    {
      "product_id": "B000123",
      "title": "Trail Running Shoe",
      "brand": "Acme",
      "color": "black",
      "category": "shoes",
      "price_cents": 7999,
      "score": 2.41,
      "explain": {
        "bm25": 8.12,
        "bm25_rank": 3,
        "knn": 0.77,
        "knn_rank": 8,
        "rrf": 0.028,
        "ltr": 2.41,
        "ltr_rank": 1
      }
    }
  ],
  "took_ms": 37
}
```

`mode` is `hybrid`, `hybrid+ltr`, or `bm25`. The API degrades to `bm25` if the embedder is unavailable, and omits `model_version` when no LTR model is loaded.

### Events

`/search` automatically queues impression events for returned results. Use `/events` to send explicit impressions, clicks, or purchases:

```bash
curl -fsS -X POST http://localhost:8080/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "click",
    "query_id": "7f33c93a6cb94836a97728582ff0aa63",
    "query": "running shoes",
    "session_id": "session-abc",
    "user_id": "user-123",
    "product_id": "B000123",
    "position": 0,
    "retrieval_scores": {
      "bm25": 8.12,
      "knn": 0.77,
      "rrf": 0.028,
      "ltr": 2.41
    },
    "source": "real"
  }'
```

Required fields are `event_type`, `query_id`, `query`, `session_id`, and `product_id`. `event_type` must be `impression`, `click`, or `purchase`; `position` must be `0` or greater. Accepted events return:

```json
{"status":"queued"}
```

### Products

```bash
curl -fsS http://localhost:8080/products/B000123
```

Response:

```json
{
  "product_id": "B000123",
  "title": "Trail Running Shoe",
  "description": "Lightweight trail shoe.",
  "bullet_points": "Durable outsole; Breathable upper",
  "brand": "Acme",
  "color": "black",
  "category": "shoes",
  "locale": "us",
  "price_cents": 7999
}
```

Unknown products return `404`.

### Admin endpoints

Admin endpoints require a configured `ADMIN_TOKEN` and either `Authorization: Bearer $ADMIN_TOKEN` or `X-Admin-Token: $ADMIN_TOKEN`.

```bash
ADMIN_TOKEN=$(openssl rand -hex 16)  # add this value to .env, then recreate the api container
docker compose up -d --force-recreate api
make reload-model                   # reads ADMIN_TOKEN from your shell or .env

# Equivalent direct API call:
curl -fsS -X POST http://localhost:8080/admin/reload-model \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

Successful model reloads return:

```json
{
  "status": "ok",
  "path": "/var/lib/drusearch/models/ltr_reranker.txt",
  "loaded_at": "2026-05-08T12:00:00Z",
  "meta": {
    "name": "ltr_reranker",
    "version": "1"
  }
}
```

If `ADMIN_TOKEN` is unset, `/admin/*` returns `503`.

## Phases

| Phase | Status | Demo |
|---|---|---|
| 0 — Scaffold + compose | done | `make up && curl localhost:8080/readyz` |
| 1 — BM25 search | done | `curl 'localhost:8080/search?q=running+shoes'` |
| 2 — Hybrid retrieval (BM25 + k-NN) | done | NDCG@10 lift on ESCI |
| 3 — Event ingest + click simulator | done | events in Postgres |
| 4 — LTR training + offline eval | done | NDCG@10 lift in MLflow |
| 5 — LTR serving in Go | done | `/admin/reload-model` |
| 6 — Personalization + retraining loop | done | per-user re-ranking |
| 7 — Production polish | done | `/metrics`, embedder circuit breaker, runbook, gate-promote, **admin auth, CI, codegen+fixture feature parity** |

## Feature schema parity

`libs/schema/feature_schema.json` is the single source of truth for the 13 LTR features. `libs/schema/codegen.py` generates:

- `pipelines/pipelines/features/_generated.py` (Python tuple + per-feature index constants)
- `services/api-go/internal/features/schema_generated.go` (Go constants + ordered Names)

Both the Python (`pipelines/pipelines/features/__init__.py`) and Go (`services/api-go/internal/features/schema_test.go`) sides assert at import/test time that the hand-written constants match the generated ones. Cross-language transform parity is tested against shared fixtures in `libs/schema/fixtures/interaction_fixtures.json` from both `pipelines/tests/test_interaction_parity.py` and `services/api-go/internal/features/parity_test.go`.

```bash
make check-feature-parity   # regen schema files and fail CI on git diff
make test-go                # Go tests including interaction-feature parity
make test-py                # Python tests including interaction-feature parity
```

CI (`.github/workflows/ci.yml`) runs all three on every push and PR.
