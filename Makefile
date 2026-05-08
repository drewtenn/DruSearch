SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE := docker compose

define ENSURE_ADMIN_TOKEN_PY
from pathlib import Path
import secrets

env = Path(".env")
if not env.exists():
    example = Path(".env.example")
    if not example.exists():
        raise SystemExit(".env is missing and .env.example was not found")
    env.write_text(example.read_text())
    print("Created .env from .env.example")

lines = env.read_text().splitlines()
changed = False
found = False
for i, line in enumerate(lines):
    if line.startswith("ADMIN_TOKEN="):
        found = True
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not value:
            lines[i] = f"ADMIN_TOKEN={secrets.token_hex(32)}"
            changed = True
        break

if not found:
    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"ADMIN_TOKEN={secrets.token_hex(32)}")
    changed = True

if changed:
    env.write_text("\n".join(lines) + "\n")
    print("Wrote ADMIN_TOKEN to .env")
else:
    print("ADMIN_TOKEN already present in .env")
endef
export ENSURE_ADMIN_TOKEN_PY

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Lifecycle — start, stop, and inspect the local app
.PHONY: up down logs ps restart ensure-admin-token bootstrap-search
up: ## Start every service in Docker
	$(COMPOSE) up -d --build
down: ## Stop every service
	$(COMPOSE) down
logs: ## Watch recent service logs
	$(COMPOSE) logs -f --tail=100
ps: ## Show running services
	$(COMPOSE) ps
restart: down up ## Restart everything
ensure-admin-token: ## Create .env if needed and fill ADMIN_TOKEN when it is blank
	@python3 -c "$$ENSURE_ADMIN_TOKEN_PY"
bootstrap-search: ## Cold-start search: data, vectors, clicks, features, model, and API reload
	$(MAKE) ensure-admin-token
	$(MAKE) up
	$(MAKE) ready
	$(MAKE) seed-databases
	$(MAKE) embed-vectors
	$(MAKE) simulate
	$(MAKE) refresh-product-features
	$(MAKE) refresh-user-features
	$(MAKE) build-training-rows
	$(MAKE) train-ltr
	$(MAKE) promote-model
	$(MAKE) verify-promoted-model
	$(COMPOSE) up -d --no-deps --force-recreate api
	$(MAKE) reload-model

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
seed-catalog: ## Put Amazon Reviews 2023 product metadata into Postgres
	$(COMPOSE) run --rm pipelines python -m pipelines.ingest.amazon_reviews
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
.PHONY: build-training-rows train-ltr retrain-model retrain-model-with-sim eval
build-training-rows: ## Convert logged behavior into examples the model can learn from
	$(COMPOSE) run --rm pipelines python -m pipelines.label.build_training_rows
train-ltr: ## Train a model to reorder search results toward better matches
	$(COMPOSE) run --rm pipelines python -m pipelines.train.lgbm_ranker
retrain-model: refresh-product-features refresh-user-features build-training-rows train-ltr promote-model verify-promoted-model reload-model ## Retrain and reload LTR from existing events
retrain-model-with-sim: simulate retrain-model ## Generate fresh simulated events, then retrain and reload LTR
eval: ## Measure ranking quality without changing the live API
	$(COMPOSE) run --rm pipelines python -m pipelines.evaluate.offline_eval

##@ Phase 5 — put the trained ranking model behind the API
.PHONY: promote-model verify-promoted-model reload-model
promote-model: ## Copy the chosen trained model where the API can read it
	$(COMPOSE) --profile jobs run --rm -e LTR_MODEL_STAGE= pipelines python -m pipelines.register.promote
verify-promoted-model: ## Fail early if the API model file has not been promoted
	$(COMPOSE) --profile jobs run --rm pipelines sh -c 'test -s "$${LTR_MODEL_DIR:-/lgbm_models}/$${LTR_MODEL_NAME:-ltr_reranker}.txt"'
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
.PHONY: refresh-product-features refresh-user-features
refresh-product-features: ## Summarize product impressions, clicks, and purchases into Postgres
	$(COMPOSE) --profile jobs run --rm pipelines python -m pipelines.features.aggregates
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
