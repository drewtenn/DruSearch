SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE := docker compose

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Lifecycle — start, stop, and inspect the local app
.PHONY: up down logs ps restart
up: ## Start every service in Docker
	$(COMPOSE) up -d --build
down: ## Stop every service
	$(COMPOSE) down
logs: ## Watch recent service logs
	$(COMPOSE) logs -f --tail=100
ps: ## Show running services
	$(COMPOSE) ps
restart: down up ## Restart everything

.PHONY: ready
ready: ## Wait until the API says its dependencies are reachable
	@echo "Waiting for API /readyz ..."
	@for i in $$(seq 1 60); do \
		if curl -fsS http://localhost:8080/readyz >/dev/null 2>&1; then echo OK; exit 0; fi; \
		sleep 2; \
	done; echo "TIMEOUT" && exit 1

##@ Phase 1 — load products and build basic text search
.PHONY: seed-databases seed-catalog index-bm25
seed-databases: seed-catalog index-bm25 ## Load sample products, then make them searchable
seed-catalog: ## Put sample product and query-label data into Postgres
	$(COMPOSE) run --rm pipelines python -m pipelines.ingest.esci
index-bm25: ## Build the OpenSearch keyword index used by /search
	$(COMPOSE) run --rm pipelines python -m pipelines.index.bm25

##@ Phase 2 — add meaning-based search signals
.PHONY: embed-vectors
embed-vectors: ## Turn product titles into vectors so similar meanings can match
	$(COMPOSE) run --rm pipelines python -m pipelines.embed.build_vectors

##@ Phase 3 — create example user behavior
.PHONY: simulate
simulate: ## Generate fake searches, clicks, and purchases for training data
	$(COMPOSE) run --rm pipelines python -m pipelines.simulate.click_simulator

##@ Phase 4 — teach and check a ranking model
.PHONY: build-training-rows train-ltr eval
build-training-rows: ## Convert logged behavior into examples the model can learn from
	$(COMPOSE) run --rm pipelines python -m pipelines.label.build_training_rows
train-ltr: ## Train a model to reorder search results toward better matches
	$(COMPOSE) run --rm pipelines python -m pipelines.train.lgbm_ranker
eval: ## Measure ranking quality without changing the live API
	$(COMPOSE) run --rm pipelines python -m pipelines.evaluate.offline_eval

##@ Phase 5 — put the trained ranking model behind the API
.PHONY: promote-model reload-model
promote-model: ## Copy the chosen trained model where the API can read it
	$(COMPOSE) --profile jobs run --rm -e LTR_MODEL_STAGE= pipelines python -m pipelines.register.promote
reload-model: ## Ask the running API to load the latest promoted model
	@set -euo pipefail; \
	admin_token="$${ADMIN_TOKEN:-}"; \
	if [ -z "$$admin_token" ] && [ -f .env ]; then \
		admin_token="$$(awk -F= '/^ADMIN_TOKEN=/{print substr($$0, index($$0, "=")+1)}' .env | tail -n 1)"; \
	fi; \
	if [ -z "$$admin_token" ]; then \
		echo "ADMIN_TOKEN is not set. Add ADMIN_TOKEN=<random string> to .env, run 'docker compose up -d --force-recreate api', then retry."; \
		exit 1; \
	fi; \
	curl --fail-with-body -sS -X POST http://localhost:8080/admin/reload-model \
		-H "Authorization: Bearer $$admin_token" | python3 -m json.tool

##@ Phase 6 — add simple per-user preferences
.PHONY: refresh-user-features
refresh-user-features: ## Summarize each user's clicked brands into Redis
	$(COMPOSE) --profile jobs run --rm pipelines python -m pipelines.features.user_aggs

##@ Phase 7 — observe the system and promote models safely
.PHONY: gate-promote metrics
gate-promote: ## Promote a model only if evaluation says it did not get worse
	$(COMPOSE) --profile jobs run --rm -e LTR_MODEL_STAGE= pipelines python -m pipelines.register.gate
metrics: ## Show API counters, timings, and model health signals
	@curl -s http://localhost:8080/metrics | grep -E '^drusearch_'

##@ Schema parity — keep Python and Go model inputs identical
.PHONY: regen-feature-schema check-feature-parity test-go test-py
regen-feature-schema: ## Regenerate language schema files from feature_schema.json
	python3 libs/schema/codegen.py
check-feature-parity: ## Verify generated schema files are in sync with the JSON; fail on drift
	@python3 libs/schema/codegen.py --check
test-go: ## Run Go unit tests (includes interaction-feature parity)
	cd services/api-go && go test ./...
test-py: ## Run Python unit tests (includes interaction-feature parity)
	cd pipelines && python3 -m pytest -v

##@ Dev helpers — open shells and inspect service health
.PHONY: psql redis-cli os-health
psql: ## Open psql shell
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-drusearch} $${POSTGRES_DB:-drusearch}
redis-cli: ## Open redis-cli
	$(COMPOSE) exec redis redis-cli
os-health: ## Check OpenSearch cluster health
	@curl -s http://localhost:9200/_cluster/health | jq .
