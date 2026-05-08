# Interaction features — cross-language reference

This file is the spec for every `FEATURE_SOURCE_INTERACTION` feature in `features.proto`. Each feature must have:

1. A precise textual spec.
2. A Python reference implementation (`pipelines/pipelines/features/transforms.py`).
3. A Go reference implementation (`services/api-go/internal/features/transforms.go`).
4. Fixture pairs of input → expected output in `libs/schema/fixtures/interaction_fixtures.json`.

Both the Python parity test (`pipelines/tests/test_interaction_parity.py`) and the Go parity test (`services/api-go/internal/features/parity_test.go`) read the same fixtures file and assert byte-equal output. CI runs both and fails on divergence. The companion drift guard for the feature schema itself lives in `feature_schema.json` + `codegen.py`; `make check-feature-parity` regenerates the language-specific files and fails on diff.

> Status: implemented in Phase 7. Schema is JSON-driven via `feature_schema.json` + `codegen.py`; both reference impls share `interaction_fixtures.json`.

## v1 interaction features

### `query_length_tokens` (FLOAT)
Tokenize the query on `\W+` (Unicode \W), filter empty strings, lowercase. Return the count as a float.

### `query_has_brand` (BOOL)
Lowercase the query, tokenize on `\W+`. Return 1.0 if any token is in the known-brand set (loaded from `products.brand` distinct values, lowercased), else 0.0.

### `query_has_color` (BOOL)
Same as `query_has_brand` against the known-color set.

### `query_has_size_pattern` (BOOL)
Return 1.0 if `re.search(r"\b\d+(\.\d+)?\s?(oz|ml|gb|tb|in|cm)\b", query, re.IGNORECASE)` matches, else 0.0. Both languages must compile the identical pattern.

### `query_affordability_intent` (BOOL)
Tokenize the query with the shared tokenization rules. Return 1.0 if the query contains one of:

- A direct affordability token: `affordable`, `affordability`, `cheap`, `cheaper`, `cheapest`, `budget`, `inexpensive`, `economical`, `value`, `price`, `priced`, `pricing`, `cost`, `costs`.
- A two-token affordability phrase: `low cost`, `low price`, `lower cost`, `lower price`, `lowest cost`, `lowest price`.

Return 0.0 otherwise.

### `affordability_price_score` (FLOAT)
If `query_affordability_intent` is 0.0 or `price_cents <= 0`, return 0.0. Otherwise return `1 / log1p(price_cents)`, so lower priced products receive a larger value only when the query expresses affordability intent.

### `session_last_query_overlap` (FLOAT)
Jaccard similarity between the token set of the current query and the token set of the most recent query in the session (per Redis hash `feat:session:{sid}.last_query`). Empty intersection or empty session → 0.0. Tokenization rules match `query_length_tokens`.
