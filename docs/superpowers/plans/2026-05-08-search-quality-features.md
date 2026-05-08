# Search Quality Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve ecommerce search relevance by replacing the LTR feature schema with a cleaner high-signal feature set for retrieval, brand, category, term coverage, attributes, and personalization, then rerun the full local system.

**Architecture:** Because the full system can be rerun, the feature schema does not need append-only compatibility. Replace the schema with a coherent v3 ordering, regenerate all generated files, rebuild API and pipelines, reindex/re-simulate/retrain from scratch, and keep Python training transforms byte-aligned with Go serving transforms through shared fixtures. Add only lightweight deterministic features first; avoid new external models until the model has strong lexical and catalog-attribute signals.

**Tech Stack:** Go API feature serving, Python pipelines for training rows and LightGBM LambdaRank, OpenSearch first-stage retrieval, shared JSON schema/codegen, Docker Compose jobs.

---

## File Map

- `libs/schema/feature_schema.json`: canonical feature list; this plan permits reordering/replacing because the full system will be rebuilt.
- `libs/schema/fixtures/interaction_fixtures.json`: shared parity cases for interaction transforms.
- `pipelines/pipelines/features/transforms.py`: Python reference transforms used during training.
- `services/api-go/internal/features/transforms.go`: Go reference transforms used during serving.
- `pipelines/tests/test_interaction_parity.py`: Python fixture assertions.
- `services/api-go/internal/features/parity_test.go`: Go fixture assertions.
- `pipelines/pipelines/features/__init__.py`: hand-written Python feature definitions.
- `services/api-go/internal/features/schema.go`: hand-written Go feature indices and names.
- `services/api-go/internal/features/schema_test.go`: schema drift guard.
- `pipelines/pipelines/label/build_training_rows.py`: training-row feature construction.
- `services/api-go/internal/features/builder.go`: serving-time feature matrix construction.
- `services/api-go/internal/retrieval/opensearch.go`: optional retrieval boosts/filters for very strong structured intent.
- `services/api-go/internal/retrieval/opensearch_test.go`: retrieval query-body tests.

---

### Task 1: Replace Feature Schema With a Clean v3 Layout

**Files:**
- Modify: `libs/schema/feature_schema.json`
- Modify: `pipelines/pipelines/features/__init__.py`
- Modify: `services/api-go/internal/features/schema.go`
- Modify: `services/api-go/internal/features/schema_test.go`

- [ ] **Step 1: Replace the schema**

Replace `libs/schema/feature_schema.json` with this complete schema:

```json
{
  "version": "v3",
  "_comment": "Canonical LTR feature schema. Full-system rebuild required when reordered. Run `make regen-feature-schema` and `make check-feature-parity` after editing.",
  "features": [
    {"index": 0, "name": "bm25_score", "kind": "FLOAT", "source": "RETRIEVAL", "description": "BM25 raw score"},
    {"index": 1, "name": "bm25_rank", "kind": "INT", "source": "RETRIEVAL", "description": "BM25 rank within query; 0 if absent"},
    {"index": 2, "name": "knn_score", "kind": "FLOAT", "source": "RETRIEVAL", "description": "k-NN cosine similarity"},
    {"index": 3, "name": "knn_rank", "kind": "INT", "source": "RETRIEVAL", "description": "k-NN rank within query; 0 if absent"},
    {"index": 4, "name": "rrf_score", "kind": "FLOAT", "source": "RETRIEVAL", "description": "RRF fused retrieval score"},
    {"index": 5, "name": "popularity_prior", "kind": "FLOAT", "source": "STATIC_PRODUCT", "description": "Catalog popularity prior in [0,1]"},
    {"index": 6, "name": "price_log_cents", "kind": "FLOAT", "source": "STATIC_PRODUCT", "description": "log1p(price_cents)"},
    {"index": 7, "name": "title_length_tokens", "kind": "FLOAT", "source": "STATIC_PRODUCT", "description": "Token count of title"},
    {"index": 8, "name": "query_length_tokens", "kind": "FLOAT", "source": "INTERACTION", "description": "Token count of query"},
    {"index": 9, "name": "query_has_brand", "kind": "BOOL", "source": "INTERACTION", "description": "Query contains a known brand token"},
    {"index": 10, "name": "query_has_color", "kind": "BOOL", "source": "INTERACTION", "description": "Query contains a known color token"},
    {"index": 11, "name": "query_has_category_token", "kind": "BOOL", "source": "INTERACTION", "description": "Query contains a known category token"},
    {"index": 12, "name": "query_has_size_pattern", "kind": "BOOL", "source": "INTERACTION", "description": "Query contains size/unit pattern"},
    {"index": 13, "name": "query_gender_intent", "kind": "INT", "source": "INTERACTION", "description": "Requested gender: 0=none, 1=men, 2=women, 3=boys, 4=girls"},
    {"index": 14, "name": "product_gender", "kind": "INT", "source": "STATIC_PRODUCT", "description": "Product gender from category path"},
    {"index": 15, "name": "gender_intent_match", "kind": "BOOL", "source": "INTERACTION", "description": "Query gender matches product gender"},
    {"index": 16, "name": "gender_intent_mismatch", "kind": "BOOL", "source": "INTERACTION", "description": "Known query gender differs from product gender"},
    {"index": 17, "name": "product_brand_match", "kind": "BOOL", "source": "INTERACTION", "description": "Query brand token matches product brand"},
    {"index": 18, "name": "product_brand_token_overlap", "kind": "FLOAT", "source": "INTERACTION", "description": "Fraction of product brand tokens present in query"},
    {"index": 19, "name": "product_color_match", "kind": "BOOL", "source": "INTERACTION", "description": "Query color token matches product color"},
    {"index": 20, "name": "title_query_token_coverage", "kind": "FLOAT", "source": "INTERACTION", "description": "Fraction of non-brand query tokens present in title"},
    {"index": 21, "name": "category_query_token_coverage", "kind": "FLOAT", "source": "INTERACTION", "description": "Fraction of non-brand query tokens present in category path"},
    {"index": 22, "name": "product_category_token_overlap", "kind": "FLOAT", "source": "INTERACTION", "description": "Fraction of product category tokens present in query"},
    {"index": 23, "name": "title_exact_query_match", "kind": "BOOL", "source": "INTERACTION", "description": "Normalized full query appears in normalized title"},
    {"index": 24, "name": "user_brand_affinity", "kind": "FLOAT", "source": "ONLINE_USER", "description": "User brand click-share in [0,1]"}
  ]
}
```

- [ ] **Step 2: Replace manual schema mirrors**

Update `pipelines/pipelines/features/__init__.py` and `services/api-go/internal/features/schema.go` to match the schema exactly. Keep `user_brand_affinity` last so all query-product interaction features are contiguous and easier to inspect.

- [ ] **Step 3: Regenerate generated schema**

Run:

```bash
make regen-feature-schema
make check-feature-parity
```

Expected: generated Python and Go schema files match v3.

---

### Task 2: Add Brand Match Features

**Files:**
- Modify: `libs/schema/fixtures/interaction_fixtures.json`
- Modify: `pipelines/tests/test_interaction_parity.py`
- Modify: `services/api-go/internal/features/parity_test.go`
- Modify: `pipelines/pipelines/features/transforms.py`
- Modify: `services/api-go/internal/features/transforms.go`
- Modify: `pipelines/pipelines/label/build_training_rows.py`
- Modify: `services/api-go/internal/features/builder.go`

- [ ] **Step 1: Add failing parity fixture cases**

Add expected fields to every case in `libs/schema/fixtures/interaction_fixtures.json`:

```json
"product_brand": "",
"product_brand_match": 0.0,
"product_brand_token_overlap": 0.0
```

Add one positive case:

```json
{
  "name": "brand_match_nike",
  "query": "nike mens running shoes",
  "product_brand": "Nike",
  "category_path": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Athletic", "Running"],
  "expected": {
    "tokenize": ["nike", "mens", "running", "shoes"],
    "query_length_tokens": 4.0,
    "query_has_brand": 1.0,
    "query_has_color": 0.0,
    "query_has_size_pattern": 0.0,
    "query_gender_intent": 1.0,
    "product_gender": 1.0,
    "gender_intent_match": 1.0,
    "gender_intent_mismatch": 0.0,
    "product_brand_match": 1.0,
    "product_brand_token_overlap": 1.0
  }
}
```

- [ ] **Step 2: Extend parity tests and verify red**

Update both parity tests to read `product_brand` and assert:

```python
product_brand_match(q, case.get("product_brand", ""))
product_brand_token_overlap(q, case.get("product_brand", ""))
```

Run:

```bash
cd services/api-go && go test ./internal/features
```

Expected: FAIL because `ProductBrandMatch` and `ProductBrandTokenOverlap` are undefined.

- [ ] **Step 3: Implement Python transforms**

Add to `pipelines/pipelines/features/transforms.py`:

```python
def product_brand_match(query: str | None, brand: str | None) -> float:
    query_tokens = set(tokenize(query))
    brand_tokens = set(tokenize(brand))
    return 1.0 if query_tokens and brand_tokens and bool(query_tokens & brand_tokens) else 0.0


def product_brand_token_overlap(query: str | None, brand: str | None) -> float:
    query_tokens = set(tokenize(query))
    brand_tokens = set(tokenize(brand))
    if not brand_tokens:
        return 0.0
    return float(len(query_tokens & brand_tokens) / len(brand_tokens))
```

- [ ] **Step 4: Implement Go transforms**

Add to `services/api-go/internal/features/transforms.go`:

```go
func ProductBrandMatch(query, brand string) float64 {
	queryTokens := tokenSet(Tokenize(query))
	brandTokens := tokenSet(Tokenize(brand))
	for t := range brandTokens {
		if _, ok := queryTokens[t]; ok {
			return 1
		}
	}
	return 0
}

func ProductBrandTokenOverlap(query, brand string) float64 {
	queryTokens := tokenSet(Tokenize(query))
	brandTokens := tokenSet(Tokenize(brand))
	if len(brandTokens) == 0 {
		return 0
	}
	matched := 0
	for t := range brandTokens {
		if _, ok := queryTokens[t]; ok {
			matched++
		}
	}
	return float64(matched) / float64(len(brandTokens))
}

func tokenSet(tokens []string) map[string]struct{} {
	out := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		out[t] = struct{}{}
	}
	return out
}
```

- [ ] **Step 5: Wire training and serving**

In `build_training_rows.py`, add:

```python
df["product_brand_match"] = [
    tf.product_brand_match(q, b) for q, b in zip(df["query"], df["brand"])
]
df["product_brand_token_overlap"] = [
    tf.product_brand_token_overlap(q, b) for q, b in zip(df["query"], df["brand"])
]
```

In `builder.go`, add:

```go
out[off+IdxProductBrandMatch] = ProductBrandMatch(query, h.Brand)
out[off+IdxProductBrandTokenOverlap] = ProductBrandTokenOverlap(query, h.Brand)
```

- [ ] **Step 6: Regenerate and verify**

Run:

```bash
make regen-feature-schema
gofmt -w services/api-go/internal/features/*.go
make check-feature-parity
cd services/api-go && go test ./internal/features
```

Expected: all commands exit 0.

---

### Task 3: Add Query Term Coverage Features

**Files:**
- Modify: `libs/schema/feature_schema.json`
- Modify: `libs/schema/fixtures/interaction_fixtures.json`
- Modify: `pipelines/pipelines/features/transforms.py`
- Modify: `services/api-go/internal/features/transforms.go`
- Modify: `pipelines/pipelines/label/build_training_rows.py`
- Modify: `services/api-go/internal/features/builder.go`

- [ ] **Step 1: Add failing fixtures**

Add a case:

```json
{
  "name": "title_and_category_coverage",
  "query": "nike trail running shoes",
  "product_brand": "Nike",
  "product_title": "Nike Wildhorse Trail Running Shoes",
  "category_path": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Athletic", "Running", "Trail Running"],
  "expected": {
    "tokenize": ["nike", "trail", "running", "shoes"],
    "query_length_tokens": 4.0,
    "query_has_brand": 1.0,
    "query_has_color": 0.0,
    "query_has_size_pattern": 0.0,
    "query_gender_intent": 0.0,
    "product_gender": 1.0,
    "gender_intent_match": 0.0,
    "gender_intent_mismatch": 0.0,
    "product_brand_match": 1.0,
    "product_brand_token_overlap": 1.0,
    "title_query_token_coverage": 1.0,
    "category_query_token_coverage": 1.0,
    "title_exact_query_match": 0.0
  }
}
```

- [ ] **Step 2: Implement transforms**

Python:

```python
def query_token_coverage(query: str | None, text: str | None, ignored_tokens: frozenset[str] = frozenset()) -> float:
    q = [t for t in tokenize(query) if t not in ignored_tokens]
    if not q:
        return 0.0
    text_tokens = set(tokenize(text))
    return float(sum(1 for t in q if t in text_tokens) / len(q))


def exact_query_phrase_match(query: str | None, text: str | None) -> float:
    q = " ".join(tokenize(query))
    haystack = " ".join(tokenize(text))
    return 1.0 if q and q in haystack else 0.0
```

Go:

```go
func QueryTokenCoverage(query, text string, ignored map[string]struct{}) float64 {
	queryTokens := Tokenize(query)
	if len(queryTokens) == 0 {
		return 0
	}
	textTokens := tokenSet(Tokenize(text))
	kept := 0
	matched := 0
	for _, t := range queryTokens {
		if _, skip := ignored[t]; skip {
			continue
		}
		kept++
		if _, ok := textTokens[t]; ok {
			matched++
		}
	}
	if kept == 0 {
		return 0
	}
	return float64(matched) / float64(kept)
}

func ExactQueryPhraseMatch(query, text string) float64 {
	q := strings.Join(Tokenize(query), " ")
	haystack := strings.Join(Tokenize(text), " ")
	if q != "" && strings.Contains(haystack, q) {
		return 1
	}
	return 0
}
```

- [ ] **Step 3: Wire builders**

Training:

```python
df["title_query_token_coverage"] = [
    tf.query_token_coverage(q, t, brand_token_set) for q, t in zip(df["query"], df["title"])
]
df["category_query_token_coverage"] = [
    tf.query_token_coverage(q, " ".join(cp or []), brand_token_set)
    for q, cp in zip(df["query"], df["category_path"])
]
df["title_exact_query_match"] = [
    tf.exact_query_phrase_match(q, t) for q, t in zip(df["query"], df["title"])
]
```

Serving:

```go
out[off+IdxTitleQueryTokenCoverage] = QueryTokenCoverage(query, h.Title, vocab.Brand)
out[off+IdxCategoryQueryTokenCoverage] = QueryTokenCoverage(query, strings.Join(h.CategoryPath, " "), vocab.Brand)
out[off+IdxTitleExactQueryMatch] = ExactQueryPhraseMatch(query, h.Title)
```

- [ ] **Step 4: Verify**

Run:

```bash
make regen-feature-schema
gofmt -w services/api-go/internal/features/*.go
make check-feature-parity
cd services/api-go && go test ./...
```

Expected: all commands exit 0.

---

### Task 4: Add Attribute Match Features

**Files:**
- Modify: `libs/schema/feature_schema.json`
- Modify: `libs/schema/fixtures/interaction_fixtures.json`
- Modify: `pipelines/pipelines/features/transforms.py`
- Modify: `services/api-go/internal/features/transforms.go`
- Modify: `pipelines/pipelines/label/build_training_rows.py`
- Modify: `services/api-go/internal/features/builder.go`

- [ ] **Step 1: Extend `Vocab`**

In `services/api-go/internal/features/builder.go`, extend:

```go
type Vocab struct {
	Brand    map[string]struct{}
	Color    map[string]struct{}
	Category map[string]struct{}
}
```

Update `LoadVocab` query:

```sql
SELECT COALESCE(brand,''), COALESCE(color,''), COALESCE(category_path, ARRAY[]::TEXT[]) FROM products
```

Tokenize each category path value into `v.Category`.

- [ ] **Step 2: Add Python category vocab**

In `build_training_rows.py`, add:

```python
category_token_set = frozenset(
    t
    for cp in products["category_path"].dropna()
    for part in (cp or [])
    for t in tf.tokenize(part)
)
```

- [ ] **Step 3: Implement transforms**

Python:

```python
def product_color_match(query: str | None, color: str | None) -> float:
    return product_brand_match(query, color)


def token_overlap_fraction(query: str | None, text: str | None) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    return float(len(query_tokens & text_tokens) / len(text_tokens))
```

Go:

```go
func ProductColorMatch(query, color string) float64 {
	return ProductBrandMatch(query, color)
}

func TokenOverlapFraction(query, text string) float64 {
	queryTokens := tokenSet(Tokenize(query))
	textTokens := tokenSet(Tokenize(text))
	if len(textTokens) == 0 {
		return 0
	}
	matched := 0
	for t := range textTokens {
		if _, ok := queryTokens[t]; ok {
			matched++
		}
	}
	return float64(matched) / float64(len(textTokens))
}
```

- [ ] **Step 4: Wire features**

Training:

```python
df["product_color_match"] = [
    tf.product_color_match(q, c) for q, c in zip(df["query"], df["color"])
]
df["query_has_category_token"] = df["query"].apply(
    lambda q: 1.0 if set(tf.tokenize(q)) & category_token_set else 0.0
)
df["product_category_token_overlap"] = [
    tf.token_overlap_fraction(q, " ".join(cp or []))
    for q, cp in zip(df["query"], df["category_path"])
]
```

Serving:

```go
out[off+IdxProductColorMatch] = ProductColorMatch(query, h.Color)
out[off+IdxQueryHasCategoryToken] = QueryHasAny(query, vocab.Category)
out[off+IdxProductCategoryTokenOverlap] = TokenOverlapFraction(query, strings.Join(h.CategoryPath, " "))
```

- [ ] **Step 5: Verify**

Run:

```bash
make regen-feature-schema
gofmt -w services/api-go/internal/features/*.go
make check-feature-parity
cd services/api-go && go test ./...
```

Expected: all commands exit 0.

---

### Task 5: Add Retrieval Boosts for Strong Structured Intent

**Files:**
- Modify: `services/api-go/internal/retrieval/opensearch.go`
- Modify: `services/api-go/internal/retrieval/opensearch_test.go`

- [ ] **Step 1: Write failing tests for brand boost**

Add a test proving a query with a known brand adds a `should` clause boosting the exact product brand:

```go
func TestBuildBM25BodyBoostsKnownBrand(t *testing.T) {
	body, err := buildBM25Body("nike running shoes", 50)
	if err != nil {
		t.Fatalf("buildBM25Body: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	query := got["query"].(map[string]any)
	boolQuery := query["bool"].(map[string]any)
	should := boolQuery["should"].([]any)
	if len(should) == 0 {
		t.Fatalf("expected brand boost should clause")
	}
}
```

- [ ] **Step 2: Implement query-builder support**

Extract the BM25 query builder so it can combine:

```go
"must": []any{baseMultiMatch},
"filter": []any{genderFilter},
"should": []any{brandBoost, categoryBoost},
"minimum_should_match": 0
```

Use a conservative `term` boost on `brand` only when the query token exactly matches one known brand token from the in-memory vocab. If retrieval has no vocab yet, skip this task and keep brand as an LTR-only feature.

- [ ] **Step 3: Verify original gender behavior still holds**

Run:

```bash
cd services/api-go && go test ./internal/retrieval
curl -s 'http://localhost:8080/search?q=mens%20nike%20running%20shoes&k=5' | jq '.results[].category_path'
```

Expected: retrieval tests pass; the API response remains all `Men` category in top 5.

---

### Task 6: Rebuild the Whole System and Promote the Model

**Files:**
- No source edits; uses Makefile targets.

- [ ] **Step 1: Rebuild containers**

Run:

```bash
docker compose down
docker compose up -d --build postgres redis opensearch minio mlflow embedder api
```

Expected: all core containers are running and healthy.

- [ ] **Step 2: Recreate data, index, and model from scratch**

Run the cold-start target if available:

```bash
make bootstrap-search
```

If `bootstrap-search` fails because a dependency is intentionally absent, run these targets in order:

```bash
make seed-databases
make index-bm25
make retrain-model-with-sim
```

Expected: Postgres has products/events/training rows, OpenSearch has a fresh product index, and the API reloads the promoted model.

- [ ] **Step 3: Run full verification after rebuild**

Run:

```bash
make check-feature-parity
cd services/api-go && go test ./...
docker compose --profile jobs run --rm pipelines python -m py_compile pipelines/features/transforms.py pipelines/features/__init__.py pipelines/label/build_training_rows.py pipelines/train/lgbm_ranker.py
```

Expected: all commands exit 0.

- [ ] **Step 4: Smoke test targeted searches**

Run:

```bash
curl -s 'http://localhost:8080/search?q=mens%20nike%20running%20shoes&k=5' | jq '.results[] | {title, brand, category_path, score, explain}'
curl -s 'http://localhost:8080/search?q=womens%20red%20adidas%20sneakers&k=5' | jq '.results[] | {title, brand, color, category_path, score, explain}'
curl -s 'http://localhost:8080/search?q=trail%20running%20shoes&k=5' | jq '.results[] | {title, brand, category_path, score, explain}'
```

Expected: gendered queries return matching gender categories; brand/color/category matches appear near the top when available; broad queries still return reasonable category matches.

---

### Task 7: Add Evaluation Gates for Feature Regressions

**Files:**
- Create: `pipelines/pipelines/evaluate/search_quality_smoke.py`
- Modify: `Makefile`

- [ ] **Step 1: Create a deterministic smoke evaluator**

Create `pipelines/pipelines/evaluate/search_quality_smoke.py`:

```python
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

BASE = "http://api:8080"

CASES = [
    {"query": "mens nike running shoes", "must_path": "Men", "must_brand": "Nike"},
    {"query": "womens running shoes", "must_path": "Women"},
    {"query": "boys shoes", "must_path": "Boys"},
    {"query": "girls sandals", "must_path": "Girls"},
]


def fetch(query: str) -> dict:
    url = f"{BASE}/search?{urllib.parse.urlencode({'q': query, 'k': 5})}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read())


def main() -> int:
    failures: list[str] = []
    for case in CASES:
        data = fetch(case["query"])
        results = data.get("results", [])
        if not results:
            failures.append(f"{case['query']}: no results")
            continue
        top = results[0]
        path = top.get("category_path") or []
        if case.get("must_path") and case["must_path"] not in path:
            failures.append(f"{case['query']}: top path {path} missing {case['must_path']}")
        if case.get("must_brand") and top.get("brand") != case["must_brand"]:
            failures.append(f"{case['query']}: top brand {top.get('brand')} != {case['must_brand']}")
    if failures:
        print("\\n".join(failures), file=sys.stderr)
        return 1
    print("search quality smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add Makefile target**

Add:

```make
.PHONY: search-quality-smoke
search-quality-smoke: ## Run deterministic API relevance smoke checks
	$(COMPOSE) --profile jobs run --rm pipelines python -m pipelines.evaluate.search_quality_smoke
```

- [ ] **Step 3: Verify gate**

Run:

```bash
docker compose up -d api embedder redis
make search-quality-smoke
```

Expected: `search quality smoke passed`.

---

## Rollout Checklist

- [ ] Replace schema intentionally and rerun the whole system; do not load old models against the new schema.
- [ ] Run `make regen-feature-schema` after every schema edit.
- [ ] Run `make check-feature-parity` before retraining.
- [ ] Run `cd services/api-go && go test ./...` before rebuilding the API.
- [ ] Rebuild API before serving any model whose feature count or order changed.
- [ ] Reindex, re-simulate, retrain, promote, and reload the model before judging quality.
- [ ] Verify representative queries by category, brand, color, and broad intent.
- [ ] Commit in small slices: brand features, coverage features, attribute features, retrieval boosts, evaluation gate.

## Self-Review

- Spec coverage: The plan covers schema changes, Python training features, Go serving features, retrieval shaping, retraining, and smoke evaluation.
- Placeholder scan: No task uses unresolved placeholders; feature indices are reset to the proposed v3 schema.
- Type consistency: Feature names use snake_case in schema/Python and `Idx...` constants in Go, matching the existing codegen/manual schema pattern.
