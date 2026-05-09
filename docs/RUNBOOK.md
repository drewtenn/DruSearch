# DruSearch Runbook

One-page operational reference: bring-up, retrain loop, common failures.

## Prerequisites

- Docker (with `docker compose v5+`)
- ~8 GB free disk for OpenSearch + indices + model artifacts
- Internet egress to Hugging Face for Amazon Reviews 2023 metadata, unless `AMAZON_REVIEWS_META_FILE` points at a local JSONL/JSONL.GZ file.

## Bring-up (cold start)

```bash
cp .env.example .env
make up                                    # docker compose up -d
make ready                                 # waits for /readyz to return 200

make seed-catalog                          # Amazon Reviews 2023 -> Postgres
make index-bm25                            # Postgres -> OpenSearch (~30s)
make embed-vectors                         # title vectors -> OpenSearch (~2min on CPU)

# Phase 3+: events (only needed if you want a populated event log)
make simulate                              # ~8min for 200 users * 50 queries

# Phase 4+: train + serve
$(MAKE) -C . refresh-user-features         # per-user brand affinity -> Redis
docker compose --profile jobs run --rm pipelines python -m pipelines.features.aggregates
docker compose --profile jobs run --rm pipelines python -m pipelines.label.build_training_rows
make train-ltr                             # configured LTR backend, MLflow run
make promote-model                         # writes model.txt to shared volume
make reload-model                          # api hot-reload
```

After this `/search?q=running+shoes` returns mode `hybrid+ltr`.

## Retrain loop (after new events)

```bash
docker compose --profile jobs run --rm pipelines python -m pipelines.features.aggregates
make refresh-user-features
docker compose --profile jobs run --rm pipelines python -m pipelines.label.build_training_rows
make train-ltr
docker compose --profile jobs run --rm pipelines python -m pipelines.register.gate    # only promotes if NDCG@10 non-regressive
make promote-model
make reload-model
```

Schedule the same sequence nightly via cron / Prefect / GitHub Actions.

Set `LTR_MODEL_BACKEND=lgbm` or `LTR_MODEL_BACKEND=xgboost` to choose the
training and serving backend. LightGBM promotion writes
`models/ltr_reranker.txt`; XGBoost promotion writes
`models/ltr_reranker.xgb.json`. Both write `models/ltr_reranker.json`
metadata for API reload.

## Observability

- `GET /metrics` — Prometheus text format. Key series:
  - `drusearch_search_latency_seconds_bucket{mode=...}` — end-to-end histogram
  - `drusearch_stage_latency_seconds_bucket{stage=embed|retrieve|user_features|rerank}` — per-stage breakdown
  - `drusearch_retrieval_candidates_bucket` — candidate pool size
  - `drusearch_search_requests_total{mode, outcome}` — counter
  - `drusearch_eventbus_written_total`, `drusearch_eventbus_dropped_total`
  - `drusearch_embedder_circuit_state` — 0 closed / 1 half-open / 2 open
  - `drusearch_ltr_model_loaded_info{name, version}` — gauge with active model identity
- MLflow UI: http://localhost:5000
- MinIO console: http://localhost:9001 (login = `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`)

## Common failure modes

### `/readyz` returns 503

Whichever of `postgres / redis / opensearch / embedder` is `false` is unhealthy. Check `docker compose ps` and `docker compose logs <name>`.

### OpenSearch "index has read-only-allow-delete block"

OpenSearch tripped the disk flood-stage watermark. We disable this via env (`cluster.routing.allocation.disk.threshold_enabled=false`) — if it ever fires anyway, free disk on the host or run:

```bash
curl -s -X PUT http://localhost:9200/products_v1/_settings \
  -H 'Content-Type: application/json' \
  -d '{"index.blocks.read_only_allow_delete": null}'
```

### Embedder unhealthy / circuit open

`/search` automatically degrades to `mode="bm25"`. The `drusearch_embedder_circuit_state` gauge tracks this. To recover:

```bash
docker compose restart embedder
# wait for /readyz to show "embedder": true; circuit auto-half-opens after 10s, fully closes after a successful probe
```

### `LTR model not loaded at boot`

Either no model file is on the shared volume, or the configured scorer rejected it. Re-run `make promote-model` with the same `LTR_MODEL_BACKEND` you trained. If the LightGBM rewrite for `version=v4 -> v3` ever stops working (LightGBM 5.x changes the tree block), fall back to pinning `lightgbm==4.5.0` in `pipelines/pyproject.toml`.

### `incorrect number of columns` from leaves

Feature schema in Python and Go is out of sync. Rebuild the api image after any change to either `pipelines/features/__init__.py` or `services/api-go/internal/features/schema.go`. Both must list the same names in the same order.

### Promotion gate fails

`make` against `pipelines.register.gate` returns exit 2 when `cand_NDCG@10 < prod_NDCG@10 - PROMOTE_TOL_NDCG`. Either:
- accept the regression by re-running with `PROMOTE_TOL_NDCG=0.05` (and document why), or
- iterate on the model (more data, hyperparam tuning, IPS weighting).

## Deliberately scoped out

- Auth: there is no end-user login flow. `/admin/*` only checks `ADMIN_TOKEN` if it's set.
- TLS: out of scope for local-first dev.
- Real high availability: single-node OpenSearch, single-process Go API, single-replica Postgres — none of this is HA.
