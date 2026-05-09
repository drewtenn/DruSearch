# DruSearch

DruSearch is a local-first e-commerce search stack. It combines lexical search, dense retrieval, click/event feedback, and a LightGBM Learning-to-Rank reranker so product results can improve from behavior.

The project is intentionally production-shaped, but runnable on one machine with Docker Compose.

## Architecture

```mermaid
flowchart LR
    Client["Client / curl"] --> API["Go API\n/search /events /products /admin"]

    API --> Embedder["Python embedder\nsentence-transformers"]
    API --> OpenSearch["OpenSearch\nBM25 + k-NN"]
    API --> Redis["Redis\nuser features"]
    API --> Postgres["Postgres\ncatalog + events"]

    Pipelines["Python pipelines"] --> Postgres
    Pipelines --> OpenSearch
    Pipelines --> Embedder
    Pipelines --> MLflow["MLflow\nmodel registry"]
    Pipelines --> MinIO["MinIO\nartifacts + cached data"]
    Pipelines --> Redis

    MLflow --> API
```

Search flow:

1. The API embeds the query with the Python embedder.
2. OpenSearch retrieves BM25 and vector candidates.
3. The API fuses candidates with RRF.
4. The LightGBM reranker optionally reorders candidates.
5. Impression/click/purchase events feed future training data.

## Quick Start

```bash
cp .env.example .env
make up
make ready
make seed-databases
make embed-vectors
```

Then search:

```bash
curl -G http://localhost:8080/search \
  --data-urlencode "q=running shoes" \
  --data-urlencode "k=5"
```

`make ready` only checks service health. A fresh stack still needs `make seed-databases` before `/search` has products to return.

## Full Demo Loop

To run the complete local workflow with data, vectors, simulated behavior, feature aggregation, model training, promotion, and API reload:

```bash
make bootstrap-search
```

After that, `/search` should return `mode` values such as `hybrid+ltr` when the model and vectors are available.

The promoted LightGBM model is written to `models/ltr_reranker.txt` plus
`models/ltr_reranker.json`. Commit those files when you want another machine
to use the trained ranker without rerunning BGE teacher scoring, training, or
evaluation.

## Common Commands

| Command | Purpose |
|---|---|
| `make up` | Start the local Docker Compose stack. |
| `make down` | Stop the stack. |
| `make ready` | Wait for API dependency readiness. |
| `make seed-databases` | Load product data into Postgres and OpenSearch. |
| `make embed-vectors` | Add dense vectors for k-NN retrieval. |
| `make simulate` | Generate example searches, clicks, and purchases. |
| `make label-bge-teacher` | Distill offline BGE teacher scores into LTR training rows. |
| `make host-pipeline-venv` | Create/update the local Python env used by host-run pipelines. |
| `make label-bge-teacher-host` | Run BGE teacher scoring on the macOS host with Apple MPS/GPU. |
| `make retrain-model` | Rebuild features, train, promote, and reload the ranker. |
| `make use-checked-in-model` | Restart the API and load the model checked into `models/`. |
| `make metrics` | Show DruSearch Prometheus metrics. |
| `make test-go` | Run Go tests. |
| `make test-py` | Run Python tests. |

## Offline BGE Teacher

BGE cross-encoder scoring is an offline distillation step for the LightGBM
ranker. The default Docker target is portable but uses the CPU-only PyTorch
wheel in the pipelines image:

```bash
make label-bge-teacher
```

On Apple Silicon, run the BGE teacher on the macOS host so PyTorch can use
Apple MPS/GPU while the databases and services still run in Docker:

```bash
make up
make ready
make host-pipeline-venv
make label-bge-teacher-host
```

`make label-bge-teacher-host` sources `.env`, overrides Docker service names
such as `postgres` and `opensearch` to `localhost`, uses
`.cache/huggingface` for model downloads, and defaults to
`BGE_TEACHER_DEVICE=mps`. The target runs `make check-host-mps` first and
fails early if host PyTorch cannot see Apple MPS.

After BGE labeling finishes, train and promote the reranker:

```bash
make train-ltr
make promote-model
make verify-promoted-model
make reload-model
```

## API

The API runs at `http://localhost:8080`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Process liveness. |
| `GET` | `/readyz` | Dependency readiness. |
| `GET` | `/metrics` | Prometheus metrics. |
| `GET` | `/search?q=...&k=...` | Product search. |
| `POST` | `/events` | Impression, click, or purchase feedback. |
| `GET` | `/products/{id}` | Product lookup. |
| `POST` | `/admin/reload-model` | Reload the promoted LTR model. |

Admin endpoints require `ADMIN_TOKEN` in `.env`.

`/search` accepts an optional `ranker` query parameter:

```bash
curl -G http://localhost:8080/search \
  --data-urlencode "q=nike shoes that are low cost" \
  --data-urlencode "k=10" \
  --data-urlencode "ranker=ltr"
```

Supported values are `hybrid`/`rrf` for BM25+kNN retrieval order and `ltr` for the local LightGBM model. BGE cross-encoder scoring runs only offline through `make label-bge-teacher` or `make label-bge-teacher-host`, where it distills weak labels for LTR training. Set `DEFAULT_RANKER=hybrid|ltr` to choose the default mode.

## Project Layout

| Path | Purpose |
|---|---|
| `services/api-go` | Go search API, retrieval, features, reranking, events. |
| `services/embedder-py` | FastAPI embedding sidecar. |
| `pipelines` | Ingest, indexing, embedding, simulation, training, evaluation. |
| `libs/schema` | Shared LTR feature schema and codegen. |
| `infra` | Local Postgres and OpenSearch setup. |
| `docs` | Architecture notes, PRD, runbook, implementation plans. |

## More Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Runbook](docs/RUNBOOK.md)
- [Product requirements](docs/PRD.md)
