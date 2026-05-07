SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE := docker compose

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Lifecycle
.PHONY: up down logs ps restart
up: ## Start all services (-d)
	$(COMPOSE) up -d --build
down: ## Stop all services
	$(COMPOSE) down
logs: ## Tail logs
	$(COMPOSE) logs -f --tail=100
ps: ## Show running services
	$(COMPOSE) ps
restart: down up ## Restart everything

.PHONY: ready
ready: ## Wait for /readyz to return 200
	@echo "Waiting for API /readyz ..."
	@for i in $$(seq 1 60); do \
		if curl -fsS http://localhost:8080/readyz >/dev/null 2>&1; then echo OK; exit 0; fi; \
		sleep 2; \
	done; echo "TIMEOUT" && exit 1

##@ Phase 1 — Catalog & BM25
.PHONY: seed-catalog index-bm25
seed-catalog: ## Load ESCI subset into Postgres (Phase 1)
	$(COMPOSE) run --rm pipelines python -m pipelines.ingest.esci
index-bm25: ## Index Postgres products into OpenSearch (Phase 1)
	$(COMPOSE) run --rm pipelines python -m pipelines.index.bm25

##@ Phase 2 — Embeddings
.PHONY: embed-vectors
embed-vectors: ## Compute & write product embeddings (Phase 2)
	$(COMPOSE) run --rm pipelines python -m pipelines.embed.build_vectors

##@ Phase 3 — Events
.PHONY: simulate
simulate: ## Run the synthetic click simulator (Phase 3)
	$(COMPOSE) run --rm pipelines python -m pipelines.simulate.click_simulator

##@ Phase 4 — Training
.PHONY: build-training-rows train-ltr eval
build-training-rows: ## Build LTR training rows from events (Phase 4)
	$(COMPOSE) run --rm pipelines python -m pipelines.label.build_training_rows
train-ltr: ## Train LightGBM ranker (Phase 4)
	$(COMPOSE) run --rm pipelines python -m pipelines.train.lgbm_ranker
eval: ## Run offline evaluation (Phase 4)
	$(COMPOSE) run --rm pipelines python -m pipelines.evaluate.offline_eval

##@ Dev helpers
.PHONY: psql redis-cli os-health
psql: ## Open psql shell
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-drusearch} $${POSTGRES_DB:-drusearch}
redis-cli: ## Open redis-cli
	$(COMPOSE) exec redis redis-cli
os-health: ## Check OpenSearch cluster health
	@curl -s http://localhost:9200/_cluster/health | jq .
