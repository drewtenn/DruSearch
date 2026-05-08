# Search Performance Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DruSearch's local demo rehearse production-scale search behavior with alias-based indexing, configurable retrieval, partial degradation, Redis hot-path caches, synthetic large-catalog testing, horizontal API readiness, and repeatable local load tests.

**Architecture:** Keep the current Docker Compose stack, but remove application assumptions that only work for one tiny index and one API process. The API searches a read alias, reads scale knobs from config, caches expensive query work in Redis, and degrades per retrieval leg. Pipelines build versioned indexes and promote aliases, while docs and load-test targets explain the local defaults versus future production settings.

**Tech Stack:** Go API, OpenSearch 2.19, Redis 7, Postgres 16, Python pipelines, Docker Compose, Prometheus metrics, Makefile orchestration.

---

## File Map

- `services/api-go/internal/config/config.go`: add search/index/cache/eventbus scale configuration and parsers.
- `services/api-go/internal/config/config_test.go`: create config parser tests.
- `services/api-go/internal/retrieval/opensearch.go`: add retrieval options, source filtering, OpenSearch timeout, single-leg kNN public method, and partial hybrid result metadata.
- `services/api-go/internal/retrieval/opensearch_test.go`: expand request-body and partial-fusion tests.
- `services/api-go/internal/searchcache/cache.go`: create Redis-backed embedding and candidate cache helpers.
- `services/api-go/internal/searchcache/cache_test.go`: create deterministic key/serialization tests.
- `services/api-go/internal/httpapi/router.go`: carry search config and cache dependencies on `Server`.
- `services/api-go/internal/httpapi/search.go`: wire config, cache, partial retrieval modes, and cache metrics into `/search`.
- `services/api-go/cmd/api/main.go`: construct retrieval options, eventbus options, and search cache from config.
- `services/api-go/internal/obs/metrics.go`: add cache, partial-degradation, retrieval-leg error, and event queue metrics.
- `services/api-go/internal/eventbus/bus.go`: expose queue depth and wire queue gauge.
- `infra/opensearch/index_template.json`: keep local defaults but prepare for alias/versioned index settings.
- `pipelines/pipelines/index/aliases.py`: create alias/index lifecycle helpers.
- `pipelines/pipelines/index/bm25.py`: build versioned indexes, bulk load, validate, and promote aliases.
- `pipelines/pipelines/embed/build_vectors.py`: target write alias or concrete build index and support vector-skip scale runs.
- `pipelines/pipelines/ingest/synthetic.py`: create synthetic product-shaped catalog data for local scale tests.
- `pipelines/tests/test_index_aliases.py`: create alias helper tests.
- `pipelines/tests/test_synthetic_ingest.py`: create synthetic catalog tests.
- `Makefile`: add scale-rehearsal and load-test targets.
- `docs/ARCHITECTURE.md`: document alias lifecycle, scale knobs, cache layers, and partial degradation.
- `docs/RUNBOOK.md`: document local scale runs, multiple API replicas, alias rollback, and metrics.

---

### Task 1: Add Search Scale Configuration

**Files:**
- Modify: `services/api-go/internal/config/config.go`
- Create: `services/api-go/internal/config/config_test.go`

- [ ] **Step 1: Write failing config tests**

Create `services/api-go/internal/config/config_test.go`:

```go
package config

import (
	"testing"
	"time"
)

func TestFromEnvSearchScaleDefaults(t *testing.T) {
	t.Setenv("POSTGRES_USER", "drusearch")
	t.Setenv("POSTGRES_PASSWORD", "drusearch")
	t.Setenv("POSTGRES_HOST", "postgres")
	t.Setenv("POSTGRES_PORT", "5432")
	t.Setenv("POSTGRES_DB", "drusearch")

	cfg, err := FromEnv()
	if err != nil {
		t.Fatalf("FromEnv: %v", err)
	}

	if cfg.OpenSearchIndex != "products_read" {
		t.Fatalf("OpenSearchIndex=%q, want products_read", cfg.OpenSearchIndex)
	}
	if cfg.BM25Candidates != 200 || cfg.KNNCandidates != 200 {
		t.Fatalf("candidates bm25=%d knn=%d, want 200/200", cfg.BM25Candidates, cfg.KNNCandidates)
	}
	if cfg.RRFK != 60 {
		t.Fatalf("RRFK=%d, want 60", cfg.RRFK)
	}
	if cfg.SearchStageTimeout != 1500*time.Millisecond {
		t.Fatalf("SearchStageTimeout=%s, want 1.5s", cfg.SearchStageTimeout)
	}
	if !cfg.PartialRetrievalEnabled {
		t.Fatalf("PartialRetrievalEnabled=false, want true")
	}
	if cfg.EmbeddingCacheTTL != 10*time.Minute {
		t.Fatalf("EmbeddingCacheTTL=%s, want 10m", cfg.EmbeddingCacheTTL)
	}
	if cfg.CandidateCacheTTL != 30*time.Second {
		t.Fatalf("CandidateCacheTTL=%s, want 30s", cfg.CandidateCacheTTL)
	}
	if cfg.EventBuffer != 8192 || cfg.EventFlushSize != 500 || cfg.EventFlushEvery != 100*time.Millisecond {
		t.Fatalf("eventbus defaults buffer=%d flush=%d every=%s", cfg.EventBuffer, cfg.EventFlushSize, cfg.EventFlushEvery)
	}
}

func TestFromEnvSearchScaleOverrides(t *testing.T) {
	t.Setenv("OPENSEARCH_INDEX", "products_shadow")
	t.Setenv("BM25_CANDIDATES", "80")
	t.Setenv("KNN_CANDIDATES", "120")
	t.Setenv("RRF_K", "42")
	t.Setenv("SEARCH_STAGE_TIMEOUT", "750ms")
	t.Setenv("PARTIAL_RETRIEVAL_ENABLED", "false")
	t.Setenv("EMBEDDING_CACHE_TTL", "3m")
	t.Setenv("CANDIDATE_CACHE_TTL", "9s")
	t.Setenv("EVENT_BUFFER", "100")
	t.Setenv("EVENT_FLUSH_SIZE", "25")
	t.Setenv("EVENT_FLUSH_EVERY", "250ms")

	cfg, err := FromEnv()
	if err != nil {
		t.Fatalf("FromEnv: %v", err)
	}

	if cfg.OpenSearchIndex != "products_shadow" {
		t.Fatalf("OpenSearchIndex=%q", cfg.OpenSearchIndex)
	}
	if cfg.BM25Candidates != 80 || cfg.KNNCandidates != 120 || cfg.RRFK != 42 {
		t.Fatalf("retrieval overrides bm25=%d knn=%d rrf=%d", cfg.BM25Candidates, cfg.KNNCandidates, cfg.RRFK)
	}
	if cfg.SearchStageTimeout != 750*time.Millisecond {
		t.Fatalf("SearchStageTimeout=%s", cfg.SearchStageTimeout)
	}
	if cfg.PartialRetrievalEnabled {
		t.Fatalf("PartialRetrievalEnabled=true, want false")
	}
	if cfg.EmbeddingCacheTTL != 3*time.Minute || cfg.CandidateCacheTTL != 9*time.Second {
		t.Fatalf("cache ttl overrides embedding=%s candidates=%s", cfg.EmbeddingCacheTTL, cfg.CandidateCacheTTL)
	}
	if cfg.EventBuffer != 100 || cfg.EventFlushSize != 25 || cfg.EventFlushEvery != 250*time.Millisecond {
		t.Fatalf("event overrides buffer=%d flush=%d every=%s", cfg.EventBuffer, cfg.EventFlushSize, cfg.EventFlushEvery)
	}
}
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
cd services/api-go && go test ./internal/config
```

Expected: FAIL because the new `Config` fields do not exist.

- [ ] **Step 3: Add config fields and parsers**

Modify `services/api-go/internal/config/config.go`.

Add fields to `Config`:

```go
BM25Candidates          int
KNNCandidates           int
RRFK                    int
SearchStageTimeout      time.Duration
PartialRetrievalEnabled bool

EmbeddingCacheTTL time.Duration
CandidateCacheTTL time.Duration

EventBuffer     int
EventFlushSize  int
EventFlushEvery time.Duration
```

Change the `FromEnv` defaults:

```go
OpenSearchIndex: getenv("OPENSEARCH_INDEX", "products_read"),
BM25Candidates: mustAtoi(getenv("BM25_CANDIDATES", "200")),
KNNCandidates: mustAtoi(getenv("KNN_CANDIDATES", "200")),
RRFK: mustAtoi(getenv("RRF_K", "60")),
SearchStageTimeout: mustDuration(getenv("SEARCH_STAGE_TIMEOUT", "1500ms")),
PartialRetrievalEnabled: mustBool(getenv("PARTIAL_RETRIEVAL_ENABLED", "true")),
EmbeddingCacheTTL: mustDuration(getenv("EMBEDDING_CACHE_TTL", "10m")),
CandidateCacheTTL: mustDuration(getenv("CANDIDATE_CACHE_TTL", "30s")),
EventBuffer: mustAtoi(getenv("EVENT_BUFFER", "8192")),
EventFlushSize: mustAtoi(getenv("EVENT_FLUSH_SIZE", "500")),
EventFlushEvery: mustDuration(getenv("EVENT_FLUSH_EVERY", "100ms")),
```

Add helper functions:

```go
func mustDuration(s string) time.Duration {
	d, err := time.ParseDuration(s)
	if err != nil {
		panic(err)
	}
	return d
}

func mustBool(s string) bool {
	v, err := strconv.ParseBool(s)
	if err != nil {
		panic(err)
	}
	return v
}
```

- [ ] **Step 4: Verify config tests pass**

Run:

```bash
cd services/api-go && go test ./internal/config
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add services/api-go/internal/config/config.go services/api-go/internal/config/config_test.go
git commit -m "feat: add search scale configuration"
```

---

### Task 2: Make Retrieval Configurable and Partially Degradable

**Files:**
- Modify: `services/api-go/internal/retrieval/opensearch.go`
- Modify: `services/api-go/internal/retrieval/opensearch_test.go`

- [ ] **Step 1: Add failing retrieval option tests**

Append to `services/api-go/internal/retrieval/opensearch_test.go`:

```go
func TestBuildBM25BodyUsesTimeoutAndSourceFields(t *testing.T) {
	opts := Options{
		SourceFields: []string{"product_id", "title"},
		SearchTimeout: "750ms",
	}
	body, err := buildBM25Body("running shoes", 25, opts)
	if err != nil {
		t.Fatalf("buildBM25Body: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if got["timeout"] != "750ms" {
		t.Fatalf("timeout=%#v, want 750ms", got["timeout"])
	}
	src := got["_source"].([]any)
	if len(src) != 2 || src[0] != "product_id" || src[1] != "title" {
		t.Fatalf("_source=%#v, want product_id/title", src)
	}
}

func TestFuseHybridAllowsSingleLegResults(t *testing.T) {
	bm25 := []rawHit{{
		ID: "p1",
		Score: 2.5,
		Source: osSource{ProductID: "p1", Title: "Running Shoe"},
	}}

	hits := fuseHybrid(bm25, nil, 60)
	if len(hits) != 1 {
		t.Fatalf("len=%d, want 1", len(hits))
	}
	if hits[0].BM25Rank != 1 || hits[0].KNNRank != 0 || hits[0].RRF == 0 {
		t.Fatalf("hit=%#v, want bm25-only fused hit", hits[0])
	}
}
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
cd services/api-go && go test ./internal/retrieval
```

Expected: FAIL because `Options`, the new `buildBM25Body` signature, and `fuseHybrid` do not exist.

- [ ] **Step 3: Add retrieval options**

Modify `services/api-go/internal/retrieval/opensearch.go`.

Add:

```go
type Options struct {
	SourceFields  []string
	SearchTimeout string
}
```

Change `Engine`:

```go
type Engine struct {
	client *opensearch.Client
	index  string
	opts   Options
}

func New(c *opensearch.Client, index string, opts ...Options) *Engine {
	o := Options{}
	if len(opts) > 0 {
		o = opts[0]
	}
	return &Engine{client: c, index: index, opts: o}
}
```

Change `sourceFields`:

```go
func sourceFields(opts Options) []string {
	if len(opts.SourceFields) > 0 {
		return opts.SourceFields
	}
	return []string{"product_id", "title", "brand", "color", "category", "category_path", "price_cents", "popularity_prior"}
}
```

Update body builders:

```go
func buildBM25Body(query string, k int, opts Options) ([]byte, error) {
	// keep existing baseQuery/searchQuery construction
	body := map[string]any{
		"size":    k,
		"_source": sourceFields(opts),
		"query":   searchQuery,
	}
	if opts.SearchTimeout != "" {
		body["timeout"] = opts.SearchTimeout
	}
	return json.Marshal(body)
}

func buildKNNBody(query string, vec []float32, k int, opts Options) ([]byte, error) {
	// keep existing titleVec/filter construction
	body := map[string]any{
		"size":    k,
		"_source": sourceFields(opts),
		"query": map[string]any{
			"knn": map[string]any{
				"title_vec": titleVec,
			},
		},
	}
	if opts.SearchTimeout != "" {
		body["timeout"] = opts.SearchTimeout
	}
	return json.Marshal(body)
}
```

Update existing calls and tests to pass `Options{}`.

- [ ] **Step 4: Extract fusion and single-leg methods**

Add:

```go
func (e *Engine) KNN(ctx context.Context, query string, vec []float32, k int) ([]Hit, error) {
	raws, err := e.knnRaw(ctx, query, vec, k)
	if err != nil {
		return nil, err
	}
	hits := make([]Hit, 0, len(raws))
	for i, r := range raws {
		hits = append(hits, Hit{
			ProductID:       r.Source.ProductID,
			Title:           r.Source.Title,
			Brand:           r.Source.Brand,
			Color:           r.Source.Color,
			Category:        r.Source.Category,
			CategoryPath:    r.Source.CategoryPath,
			PriceCents:      r.Source.PriceCents,
			PopularityPrior: r.Source.PopularityPrior,
			KNN:             r.Score,
			KNNRank:         i + 1,
		})
	}
	return hits, nil
}
```

Move the current merge/sort body from `Hybrid` into:

```go
func fuseHybrid(bm25, knn []rawHit, rrfK int) []Hit {
	// existing merged map logic
	// existing sort by RRF descending
	return hits
}
```

Change `Hybrid` so it returns `fuseHybrid(bm25, knn, rrfK)` after both legs succeed.

- [ ] **Step 5: Verify retrieval tests pass**

Run:

```bash
gofmt -w services/api-go/internal/retrieval/opensearch.go services/api-go/internal/retrieval/opensearch_test.go
cd services/api-go && go test ./internal/retrieval
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add services/api-go/internal/retrieval/opensearch.go services/api-go/internal/retrieval/opensearch_test.go
git commit -m "feat: make retrieval configurable"
```

---

### Task 3: Add Redis Search Cache Helpers

**Files:**
- Create: `services/api-go/internal/searchcache/cache.go`
- Create: `services/api-go/internal/searchcache/cache_test.go`

- [ ] **Step 1: Write cache key and serialization tests**

Create `services/api-go/internal/searchcache/cache_test.go`:

```go
package searchcache

import (
	"testing"
	"time"

	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
)

func TestNormalizeQuery(t *testing.T) {
	got := NormalizeQuery("  Running   SHOES  ")
	if got != "running shoes" {
		t.Fatalf("NormalizeQuery=%q, want running shoes", got)
	}
}

func TestEmbeddingKeyIncludesModelAndQuery(t *testing.T) {
	got := EmbeddingKey("BAAI/bge-small-en-v1.5", "  Running   SHOES  ")
	want := "cache:embed:BAAI/bge-small-en-v1.5:running shoes"
	if got != want {
		t.Fatalf("EmbeddingKey=%q, want %q", got, want)
	}
}

func TestCandidateKeyIncludesIndexAndConfig(t *testing.T) {
	got := CandidateKey("products_read", "bm25=200;knn=200;rrf=60", "Running Shoes")
	want := "cache:candidates:products_read:bm25=200;knn=200;rrf=60:running shoes"
	if got != want {
		t.Fatalf("CandidateKey=%q, want %q", got, want)
	}
}

func TestCandidatePayloadRoundTrip(t *testing.T) {
	in := CandidatePayload{
		Mode:      "hybrid",
		CreatedAt: time.Unix(100, 0).UTC(),
		Hits: []retrieval.Hit{{
			ProductID: "p1",
			Title: "Running Shoe",
			BM25: 1.2,
			BM25Rank: 1,
			RRF: 0.03,
		}},
	}
	b, err := MarshalCandidates(in)
	if err != nil {
		t.Fatalf("MarshalCandidates: %v", err)
	}
	got, err := UnmarshalCandidates(b)
	if err != nil {
		t.Fatalf("UnmarshalCandidates: %v", err)
	}
	if got.Mode != "hybrid" || len(got.Hits) != 1 || got.Hits[0].ProductID != "p1" {
		t.Fatalf("round trip=%#v", got)
	}
}
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
cd services/api-go && go test ./internal/searchcache
```

Expected: FAIL because the package does not exist.

- [ ] **Step 3: Implement cache package**

Create `services/api-go/internal/searchcache/cache.go`:

```go
package searchcache

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
)

type Cache struct {
	rdb *redis.Client
}

type CandidatePayload struct {
	Mode      string          `json:"mode"`
	CreatedAt time.Time       `json:"created_at"`
	Hits      []retrieval.Hit `json:"hits"`
}

func New(rdb *redis.Client) *Cache {
	return &Cache{rdb: rdb}
}

func NormalizeQuery(q string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(q))), " ")
}

func EmbeddingKey(model, query string) string {
	return "cache:embed:" + model + ":" + NormalizeQuery(query)
}

func CandidateKey(index, configVersion, query string) string {
	return "cache:candidates:" + index + ":" + configVersion + ":" + NormalizeQuery(query)
}

func MarshalVector(vec []float32) ([]byte, error) {
	return json.Marshal(vec)
}

func UnmarshalVector(b []byte) ([]float32, error) {
	var vec []float32
	err := json.Unmarshal(b, &vec)
	return vec, err
}

func MarshalCandidates(p CandidatePayload) ([]byte, error) {
	return json.Marshal(p)
}

func UnmarshalCandidates(b []byte) (CandidatePayload, error) {
	var p CandidatePayload
	err := json.Unmarshal(b, &p)
	return p, err
}

func SafeVersion(s string) string {
	return base64.RawURLEncoding.EncodeToString([]byte(s))
}

func (c *Cache) GetEmbedding(ctx context.Context, key string) ([]float32, bool, error) {
	b, err := c.rdb.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	vec, err := UnmarshalVector(b)
	if err != nil {
		return nil, false, err
	}
	return vec, true, nil
}

func (c *Cache) SetEmbedding(ctx context.Context, key string, vec []float32, ttl time.Duration) error {
	b, err := MarshalVector(vec)
	if err != nil {
		return err
	}
	return c.rdb.Set(ctx, key, b, ttl).Err()
}

func (c *Cache) GetCandidates(ctx context.Context, key string) (CandidatePayload, bool, error) {
	b, err := c.rdb.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return CandidatePayload{}, false, nil
	}
	if err != nil {
		return CandidatePayload{}, false, err
	}
	p, err := UnmarshalCandidates(b)
	if err != nil {
		return CandidatePayload{}, false, err
	}
	return p, true, nil
}

func (c *Cache) SetCandidates(ctx context.Context, key string, p CandidatePayload, ttl time.Duration) error {
	b, err := MarshalCandidates(p)
	if err != nil {
		return err
	}
	return c.rdb.Set(ctx, key, b, ttl).Err()
}
```

- [ ] **Step 4: Verify cache tests pass**

Run:

```bash
gofmt -w services/api-go/internal/searchcache/*.go
cd services/api-go && go test ./internal/searchcache
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add services/api-go/internal/searchcache
git commit -m "feat: add search cache helpers"
```

---

### Task 4: Wire Config, Caches, and Partial Modes into `/search`

**Files:**
- Modify: `services/api-go/cmd/api/main.go`
- Modify: `services/api-go/internal/httpapi/router.go`
- Modify: `services/api-go/internal/httpapi/search.go`
- Modify: `services/api-go/internal/obs/metrics.go`

- [ ] **Step 1: Add API metrics**

Modify `services/api-go/internal/obs/metrics.go` by adding:

```go
CacheTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "drusearch_cache_total",
	Help: "Search cache lookups partitioned by cache and outcome.",
}, []string{"cache", "outcome"})

PartialRetrievalTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "drusearch_partial_retrieval_total",
	Help: "Partial retrieval degradations partitioned by failed leg.",
}, []string{"failed_leg"})

RetrievalLegErrors = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "drusearch_retrieval_leg_errors_total",
	Help: "OpenSearch retrieval errors partitioned by retrieval leg.",
}, []string{"leg"})
```

- [ ] **Step 2: Add server dependencies**

Modify `services/api-go/internal/httpapi/router.go`.

Add imports:

```go
"time"

"github.com/drewtenn/drusearch/services/api-go/internal/searchcache"
```

Add:

```go
type SearchConfig struct {
	BM25Candidates          int
	KNNCandidates           int
	RRFK                    int
	StageTimeout            time.Duration
	PartialRetrievalEnabled bool
	EmbeddingCacheTTL       time.Duration
	CandidateCacheTTL       time.Duration
	OpenSearchIndex         string
	EmbeddingModel          string
}
```

Add fields to `Server`:

```go
SearchConfig SearchConfig
SearchCache  *searchcache.Cache
```

- [ ] **Step 3: Wire main construction**

Modify `services/api-go/cmd/api/main.go`.

Add import:

```go
"github.com/drewtenn/drusearch/services/api-go/internal/searchcache"
```

Construct retrieval with options:

```go
ret := retrieval.New(stores.OS, cfg.OpenSearchIndex, retrieval.Options{
	SearchTimeout: cfg.SearchStageTimeout.String(),
})
```

Construct eventbus with config:

```go
bus := eventbus.New(stores.PG, logger, eventbus.Options{
	Buffer:     cfg.EventBuffer,
	FlushSize:  cfg.EventFlushSize,
	FlushEvery: cfg.EventFlushEvery,
})
```

Add to `httpapi.Server`:

```go
SearchConfig: httpapi.SearchConfig{
	BM25Candidates:          cfg.BM25Candidates,
	KNNCandidates:           cfg.KNNCandidates,
	RRFK:                    cfg.RRFK,
	StageTimeout:            cfg.SearchStageTimeout,
	PartialRetrievalEnabled: cfg.PartialRetrievalEnabled,
	EmbeddingCacheTTL:       cfg.EmbeddingCacheTTL,
	CandidateCacheTTL:       cfg.CandidateCacheTTL,
	OpenSearchIndex:         cfg.OpenSearchIndex,
	EmbeddingModel:          os.Getenv("EMBEDDER_MODEL"),
},
SearchCache: searchcache.New(stores.RDB),
```

If `EmbeddingModel` is blank, set it to `"BAAI/bge-small-en-v1.5"` before constructing the server.

- [ ] **Step 4: Replace hardcoded candidate constants in search**

Modify `services/api-go/internal/httpapi/search.go`.

Remove:

```go
defaultCandN = 200
rrfK = 60
```

Use `s.SearchConfig.BM25Candidates`, `s.SearchConfig.KNNCandidates`, and `s.SearchConfig.RRFK`.

Wrap embed/retrieval/cache calls with:

```go
stageCtx, cancel := context.WithTimeout(r.Context(), s.SearchConfig.StageTimeout)
defer cancel()
```

Use one stage context per stage so cancellation is local and metrics remain meaningful.

- [ ] **Step 5: Add embedding cache**

In `search.go`, before calling `s.Embedder.Embed`, compute:

```go
embedKey := searchcache.EmbeddingKey(s.SearchConfig.EmbeddingModel, q)
```

If `s.SearchCache != nil`, try `GetEmbedding`. On hit, set `vec` and increment:

```go
obs.CacheTotal.WithLabelValues("embedding", "hit").Inc()
```

On miss:

```go
obs.CacheTotal.WithLabelValues("embedding", "miss").Inc()
```

After successful embedder response, write the vector with `SetEmbedding`. If Redis read/write fails, log a warning and continue without failing the request.

- [ ] **Step 6: Add anonymous candidate cache**

Compute a config version string:

```go
candidateVersion := fmt.Sprintf("bm25=%d;knn=%d;rrf=%d", s.SearchConfig.BM25Candidates, s.SearchConfig.KNNCandidates, s.SearchConfig.RRFK)
candidateKey := searchcache.CandidateKey(s.SearchConfig.OpenSearchIndex, candidateVersion, q)
```

Try candidate cache after embedding succeeds or BM25-only mode is selected. Cache only retrieval candidates before rerank. On hit, use cached hits and cached mode, then still load user features and rerank.

Increment:

```go
obs.CacheTotal.WithLabelValues("candidates", "hit").Inc()
obs.CacheTotal.WithLabelValues("candidates", "miss").Inc()
```

On successful retrieval, write:

```go
s.SearchCache.SetCandidates(ctx, candidateKey, searchcache.CandidatePayload{
	Mode: mode,
	CreatedAt: time.Now().UTC(),
	Hits: hits,
}, s.SearchConfig.CandidateCacheTTL)
```

- [ ] **Step 7: Implement partial retrieval mode selection**

In `search.go`, replace direct `Hybrid` use with logic:

```go
if mode == "hybrid" {
	hits, err = s.Retrieval.Hybrid(r.Context(), q, vec, minCandidatePool(s.SearchConfig), s.SearchConfig.RRFK)
	if err != nil && s.SearchConfig.PartialRetrievalEnabled {
		s.Logger.Warn("hybrid retrieval failed; trying partial legs", zap.Error(err))
		bm25Hits, bm25Err := s.Retrieval.BM25(r.Context(), q, s.SearchConfig.BM25Candidates)
		knnHits, knnErr := s.Retrieval.KNN(r.Context(), q, vec, s.SearchConfig.KNNCandidates)
		switch {
		case bm25Err == nil:
			hits = bm25Hits
			mode = "bm25"
			err = nil
			obs.PartialRetrievalTotal.WithLabelValues("knn").Inc()
		case knnErr == nil:
			hits = knnHits
			mode = "knn"
			err = nil
			obs.PartialRetrievalTotal.WithLabelValues("bm25").Inc()
		default:
			obs.RetrievalLegErrors.WithLabelValues("bm25").Inc()
			obs.RetrievalLegErrors.WithLabelValues("knn").Inc()
		}
	}
}
```

Then refine this implementation so BM25 and kNN partial retries use stage timeouts and do not double-count errors when both legs fail.

Add helper:

```go
func minCandidatePool(cfg SearchConfig) int {
	if cfg.BM25Candidates > cfg.KNNCandidates {
		return cfg.BM25Candidates
	}
	return cfg.KNNCandidates
}
```

- [ ] **Step 8: Verify Go tests**

Run:

```bash
gofmt -w services/api-go/cmd/api/main.go services/api-go/internal/httpapi/*.go services/api-go/internal/obs/metrics.go
cd services/api-go && go test ./...
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add services/api-go/cmd/api/main.go services/api-go/internal/httpapi services/api-go/internal/obs/metrics.go
git commit -m "feat: cache and degrade search hot path"
```

---

### Task 5: Add Alias-Aware Index Lifecycle Helpers

**Files:**
- Create: `pipelines/pipelines/index/aliases.py`
- Create: `pipelines/tests/test_index_aliases.py`
- Modify: `pipelines/pipelines/index/bm25.py`
- Modify: `pipelines/pipelines/embed/build_vectors.py`

- [ ] **Step 1: Write alias helper tests**

Create `pipelines/tests/test_index_aliases.py`:

```python
from pipelines.index.aliases import (
    next_index_name,
    promote_alias_actions,
    read_alias,
    write_alias,
)


def test_next_index_name_starts_at_v1():
    assert next_index_name([]) == "products_v1"


def test_next_index_name_increments_highest_version():
    assert next_index_name(["products_v1", "products_v9", "other"]) == "products_v10"


def test_alias_names_are_stable():
    assert read_alias() == "products_read"
    assert write_alias() == "products_write"


def test_promote_alias_actions_removes_old_and_adds_new():
    body = promote_alias_actions("products_v2", old_indexes=["products_v1"])
    assert body == {
        "actions": [
            {"remove": {"index": "products_v1", "alias": "products_read"}},
            {"remove": {"index": "products_v1", "alias": "products_write"}},
            {"add": {"index": "products_v2", "alias": "products_read"}},
            {"add": {"index": "products_v2", "alias": "products_write"}},
        ]
    }
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
cd pipelines && python3 -m pytest tests/test_index_aliases.py -v
```

Expected: FAIL because `pipelines.index.aliases` does not exist.

- [ ] **Step 3: Implement alias helpers**

Create `pipelines/pipelines/index/aliases.py`:

```python
from __future__ import annotations

import re


PREFIX = "products_v"
READ_ALIAS = "products_read"
WRITE_ALIAS = "products_write"


def read_alias() -> str:
    return READ_ALIAS


def write_alias() -> str:
    return WRITE_ALIAS


def next_index_name(existing: list[str]) -> str:
    versions = []
    for name in existing:
        m = re.fullmatch(r"products_v(\d+)", name)
        if m:
            versions.append(int(m.group(1)))
    return f"{PREFIX}{(max(versions) + 1) if versions else 1}"


def promote_alias_actions(new_index: str, old_indexes: list[str]) -> dict:
    actions = []
    for old in sorted(old_indexes):
        actions.append({"remove": {"index": old, "alias": READ_ALIAS}})
        actions.append({"remove": {"index": old, "alias": WRITE_ALIAS}})
    actions.append({"add": {"index": new_index, "alias": READ_ALIAS}})
    actions.append({"add": {"index": new_index, "alias": WRITE_ALIAS}})
    return {"actions": actions}
```

- [ ] **Step 4: Modify BM25 indexer to build versioned indexes**

Modify `pipelines/pipelines/index/bm25.py`.

Replace `INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "products_v1")` with:

```python
from pipelines.index import aliases

INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "")
PROMOTE_ALIASES = os.getenv("INDEX_PROMOTE_ALIASES", "true").lower() == "true"
```

Add:

```python
def _existing_product_indexes(client) -> list[str]:
    indexes = client.indices.get(index="products_v*", ignore_unavailable=True)
    return sorted(indexes.keys())


def _target_index(client) -> str:
    if INDEX_NAME:
        return INDEX_NAME
    return aliases.next_index_name(_existing_product_indexes(client))
```

Change `_ensure_index(client)` to `_ensure_index(client, index_name: str)` and create the target index without deleting unrelated indexes:

```python
def _ensure_index(client, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        log.info("deleting existing target index %s", index_name)
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name)
    client.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "-1"}})
    log.info("created index %s", index_name)
```

Change `_iter_docs()` to `_iter_docs(index_name: str)` and set `"_index": index_name`.

Add:

```python
def _promote_aliases(client, index_name: str) -> None:
    old_indexes = []
    try:
        current = client.indices.get_alias(name=aliases.read_alias())
        old_indexes = sorted(current.keys())
    except Exception:
        old_indexes = []
    body = aliases.promote_alias_actions(index_name, old_indexes)
    client.indices.update_aliases(body=body)
    log.info("promoted %s to aliases %s/%s", index_name, aliases.read_alias(), aliases.write_alias())
```

In `main()`, compute `index_name = _target_index(client)`, pass it through bulk indexing, refresh that index, validate count, then call `_promote_aliases` when `PROMOTE_ALIASES` is true.

- [ ] **Step 5: Modify vector builder target**

Modify `pipelines/pipelines/embed/build_vectors.py`.

Change:

```python
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "products_v1")
```

to:

```python
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "products_write")
SKIP_VECTORS = os.getenv("SKIP_VECTORS", "false").lower() == "true"
```

At the start of `main()`, after logging OpenSearch info:

```python
if SKIP_VECTORS:
    log.info("SKIP_VECTORS=true; leaving %s without title_vec updates", INDEX_NAME)
    return 0
```

- [ ] **Step 6: Verify pipeline tests**

Run:

```bash
cd pipelines && python3 -m pytest tests/test_index_aliases.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add pipelines/pipelines/index/aliases.py pipelines/tests/test_index_aliases.py pipelines/pipelines/index/bm25.py pipelines/pipelines/embed/build_vectors.py
git commit -m "feat: index products through aliases"
```

---

### Task 6: Add Synthetic Catalog Generator for Local Scale Tests

**Files:**
- Create: `pipelines/pipelines/ingest/synthetic.py`
- Create: `pipelines/tests/test_synthetic_ingest.py`
- Modify: `Makefile`

- [ ] **Step 1: Write synthetic generator tests**

Create `pipelines/tests/test_synthetic_ingest.py`:

```python
from pipelines.ingest.synthetic import synthetic_product


def test_synthetic_product_is_deterministic():
    a = synthetic_product(42)
    b = synthetic_product(42)
    assert a == b


def test_synthetic_product_has_required_shape():
    p = synthetic_product(7)
    assert p["product_id"] == "synthetic-00000007"
    assert p["title"]
    assert p["description"]
    assert p["bullet_points"]
    assert p["brand"]
    assert p["color"]
    assert p["category"]
    assert isinstance(p["category_path"], list)
    assert p["price_cents"] > 0
    assert 0.0 <= p["popularity_prior"] <= 1.0
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
cd pipelines && python3 -m pytest tests/test_synthetic_ingest.py -v
```

Expected: FAIL because `pipelines.ingest.synthetic` does not exist.

- [ ] **Step 3: Implement synthetic ingest**

Create `pipelines/pipelines/ingest/synthetic.py`:

```python
from __future__ import annotations

import os
from typing import Iterator

from psycopg.rows import dict_row

from pipelines.common import db
from pipelines.common.logging import configure

log = configure("ingest.synthetic")

COUNT = int(os.getenv("SYNTHETIC_PRODUCTS", "100000"))
BATCH = int(os.getenv("SYNTHETIC_BATCH", "5000"))

BRANDS = ["Nike", "Adidas", "Sony", "Apple", "Anker", "Levi's", "KitchenAid", "Samsung"]
COLORS = ["black", "white", "blue", "red", "green", "silver", "gray", "brown"]
CATEGORIES = [
    ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Athletic", "Running"],
    ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Athletic", "Walking"],
    ["Electronics", "Headphones", "Earbud Headphones"],
    ["Home & Kitchen", "Kitchen & Dining", "Coffee"],
    ["Sports & Outdoors", "Exercise & Fitness", "Accessories"],
]
NOUNS = ["shoes", "headphones", "jacket", "bottle", "charger", "backpack", "coffee maker", "watch"]
ADJECTIVES = ["lightweight", "wireless", "waterproof", "premium", "compact", "breathable", "insulated"]


def synthetic_product(i: int) -> dict:
    brand = BRANDS[i % len(BRANDS)]
    color = COLORS[(i // len(BRANDS)) % len(COLORS)]
    category_path = CATEGORIES[i % len(CATEGORIES)]
    noun = NOUNS[(i // 3) % len(NOUNS)]
    adj = ADJECTIVES[(i // 5) % len(ADJECTIVES)]
    title = f"{brand} {adj} {color} {noun} model {i % 1000}"
    return {
        "product_id": f"synthetic-{i:08d}",
        "title": title,
        "description": f"{title} for local scale testing with stable generated content.",
        "bullet_points": f"{adj.title()} design\nColor: {color}\nBrand: {brand}",
        "brand": brand,
        "color": color,
        "category": category_path[-1],
        "category_path": category_path,
        "price_cents": 999 + ((i * 37) % 25000),
        "popularity_prior": ((i * 17) % 1000) / 1000.0,
    }


def iter_products(count: int) -> Iterator[dict]:
    for i in range(count):
        yield synthetic_product(i)


def chunks(count: int, batch: int) -> Iterator[list[dict]]:
    rows = []
    for p in iter_products(count):
        rows.append(p)
        if len(rows) >= batch:
            yield rows
            rows = []
    if rows:
        yield rows


def main() -> int:
    log.info("writing synthetic products count=%d batch=%d", COUNT, BATCH)
    with db.conn() as conn:
        conn.row_factory = dict_row
        with conn.cursor() as cur:
            for batch_rows in chunks(COUNT, BATCH):
                cur.executemany(
                    """
                    INSERT INTO products (
                        product_id, title, description, bullet_points, brand, color,
                        category, category_path, price_cents, popularity_prior
                    ) VALUES (
                        %(product_id)s, %(title)s, %(description)s, %(bullet_points)s,
                        %(brand)s, %(color)s, %(category)s, %(category_path)s,
                        %(price_cents)s, %(popularity_prior)s
                    )
                    ON CONFLICT (product_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        bullet_points = EXCLUDED.bullet_points,
                        brand = EXCLUDED.brand,
                        color = EXCLUDED.color,
                        category = EXCLUDED.category,
                        category_path = EXCLUDED.category_path,
                        price_cents = EXCLUDED.price_cents,
                        popularity_prior = EXCLUDED.popularity_prior
                    """,
                    batch_rows,
                )
                log.info("upserted synthetic batch=%d", len(batch_rows))
        conn.commit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add Makefile target**

Modify `Makefile` under Phase 1:

```make
.PHONY: seed-synthetic
seed-synthetic: ## Generate synthetic product-shaped records for local scale testing
	$(COMPOSE) run --rm pipelines python -m pipelines.ingest.synthetic
```

- [ ] **Step 5: Verify tests**

Run:

```bash
cd pipelines && python3 -m pytest tests/test_synthetic_ingest.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add pipelines/pipelines/ingest/synthetic.py pipelines/tests/test_synthetic_ingest.py Makefile
git commit -m "feat: add synthetic catalog scale data"
```

---

### Task 7: Add Load-Test and Scale-Rehearsal Targets

**Files:**
- Create: `pipelines/pipelines/evaluate/load_search.py`
- Modify: `Makefile`

- [ ] **Step 1: Create load-test script**

Create `pipelines/pipelines/evaluate/load_search.py`:

```python
from __future__ import annotations

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


API_URL = os.getenv("API_URL", "http://api:8080")
REQUESTS = int(os.getenv("LOAD_REQUESTS", "200"))
CONCURRENCY = int(os.getenv("LOAD_CONCURRENCY", "20"))
K = int(os.getenv("LOAD_K", "20"))
QUERIES = [
    "running shoes",
    "wireless earbuds",
    "black backpack",
    "coffee maker",
    "waterproof jacket",
    "nike mens shoes",
    "sony headphones",
    "insulated bottle",
]


def one(client: httpx.Client, i: int) -> tuple[int, float, str]:
    q = QUERIES[i % len(QUERIES)]
    start = time.perf_counter()
    r = client.get(f"{API_URL}/search", params={"q": q, "k": K}, timeout=10)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    mode = ""
    if r.status_code == 200:
        mode = r.json().get("mode", "")
    return r.status_code, elapsed_ms, mode


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, round((p / 100.0) * (len(values) - 1)))
    return values[idx]


def main() -> int:
    latencies: list[float] = []
    statuses: dict[int, int] = {}
    modes: dict[str, int] = {}
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(one, client, i) for i in range(REQUESTS)]
            for fut in as_completed(futures):
                status, elapsed, mode = fut.result()
                statuses[status] = statuses.get(status, 0) + 1
                if status == 200:
                    latencies.append(elapsed)
                    modes[mode] = modes.get(mode, 0) + 1

    print(f"requests={REQUESTS} concurrency={CONCURRENCY} statuses={statuses} modes={modes}")
    if latencies:
        print(
            "latency_ms "
            f"min={min(latencies):.1f} "
            f"mean={statistics.mean(latencies):.1f} "
            f"p50={percentile(latencies, 50):.1f} "
            f"p95={percentile(latencies, 95):.1f} "
            f"p99={percentile(latencies, 99):.1f} "
            f"max={max(latencies):.1f}"
        )
    return 0 if statuses.get(200, 0) == REQUESTS else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add Makefile targets**

Modify `Makefile`.

Add to Phase 7:

```make
.PHONY: load-search scale-rehearsal-local
load-search: ## Run a local concurrent /search load test and print latency percentiles
	$(COMPOSE) --profile jobs run --rm pipelines python -m pipelines.evaluate.load_search

scale-rehearsal-local: ## Generate synthetic data, index through aliases, skip vectors, and load-test BM25/LTR path
	$(MAKE) up
	$(MAKE) ready
	$(MAKE) seed-synthetic
	$(MAKE) index-bm25
	$(COMPOSE) --profile jobs run --rm -e SKIP_VECTORS=true pipelines python -m pipelines.embed.build_vectors
	$(MAKE) load-search
```

- [ ] **Step 3: Verify script syntax**

Run:

```bash
cd pipelines && python3 -m py_compile pipelines/evaluate/load_search.py
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add pipelines/pipelines/evaluate/load_search.py Makefile
git commit -m "feat: add local search load test"
```

---

### Task 8: Document Scale-Rehearsal Operations

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RUNBOOK.md`

- [ ] **Step 1: Update architecture docs**

Modify `docs/ARCHITECTURE.md`.

In the OpenSearch index section, replace references that say the API depends directly on `products_v1` with:

```markdown
### OpenSearch aliases and backing indexes

The API searches `products_read` by default. Indexing jobs build immutable backing indexes named `products_v<N>` and promote them through aliases:

| Name | Role |
|---|---|
| `products_read` | API search target |
| `products_write` | pipeline update target for vector writes during a build |
| `products_v<N>` | concrete backing index, retained for rollback until cleanup |

Local defaults keep `number_of_shards=1` and `number_of_replicas=0`. Production deployments should increase shards/replicas based on catalog size, query volume, heap, and vector memory.
```

In the latency and failure mode sections, add the new partial behavior:

```markdown
If one retrieval leg fails and `PARTIAL_RETRIEVAL_ENABLED=true`, `/search` returns the surviving leg (`bm25` or `knn`) and still applies LTR when a model is loaded. If both legs fail, the request returns 500.
```

In the feature/serving sections, add cache notes:

```markdown
Redis also stores short-lived hot-path caches:

- `cache:embed:<model>:<normalized_query>` for query vectors.
- `cache:candidates:<index>:<retrieval_config>:<normalized_query>` for anonymous candidate pools before personalization and LTR.
```

- [ ] **Step 2: Update runbook**

Modify `docs/RUNBOOK.md`.

Add a section:

````markdown
## Local scale rehearsal

The local stack can rehearse production-shaped search behavior without pretending a laptop is a production cluster.

```bash
SYNTHETIC_PRODUCTS=100000 make seed-synthetic
make index-bm25
SKIP_VECTORS=true docker compose --profile jobs run --rm pipelines python -m pipelines.embed.build_vectors
LOAD_REQUESTS=500 LOAD_CONCURRENCY=25 make load-search
```

The default API searches `products_read`. The indexer creates a new `products_v<N>` backing index and promotes `products_read` / `products_write` after successful bulk indexing.

To rollback an alias manually:

```bash
curl -s -X POST http://localhost:9200/_aliases \
  -H 'Content-Type: application/json' \
  -d '{"actions":[
    {"remove":{"index":"products_v2","alias":"products_read"}},
    {"add":{"index":"products_v1","alias":"products_read"}}
  ]}'
```

Run multiple API containers locally by assigning unique host ports or by using Compose scaling behind a local proxy. The API process is stateless; Redis, OpenSearch, Postgres, and mounted model artifacts carry shared state.
````

Update Observability list with:

```markdown
- `drusearch_cache_total{cache,outcome}` - embedding and candidate cache hits/misses
- `drusearch_partial_retrieval_total{failed_leg}` - degraded retrieval requests
- `drusearch_retrieval_leg_errors_total{leg}` - BM25/kNN leg errors
```

- [ ] **Step 3: Verify docs mention new commands**

Run:

```bash
rg -n "products_read|scale rehearsal|load-search|drusearch_cache_total|PARTIAL_RETRIEVAL_ENABLED" docs
```

Expected: output includes `docs/ARCHITECTURE.md` and `docs/RUNBOOK.md`.

- [ ] **Step 4: Commit**

Run:

```bash
git add docs/ARCHITECTURE.md docs/RUNBOOK.md
git commit -m "docs: document search scale rehearsal"
```

---

### Task 9: Final Verification

**Files:**
- No new files. Verify all changed files from Tasks 1-8.

- [ ] **Step 1: Run Go tests**

Run:

```bash
cd services/api-go && go test ./...
```

Expected: PASS.

- [ ] **Step 2: Run Python tests**

Run:

```bash
cd pipelines && python3 -m pytest -v
```

Expected: PASS.

- [ ] **Step 3: Run feature parity check**

Run:

```bash
make check-feature-parity
```

Expected: PASS with no generated-file diff.

- [ ] **Step 4: Run local smoke if Docker is available**

Run:

```bash
make up
make ready
make seed-synthetic
make index-bm25
make load-search
```

Expected: `/readyz` returns OK, indexing promotes `products_read`, and `load-search` prints status counts with all requests returning 200.

- [ ] **Step 5: Inspect metrics**

Run:

```bash
make metrics
```

Expected: output includes:

```text
drusearch_search_requests_total
drusearch_search_latency_seconds
drusearch_stage_latency_seconds
drusearch_cache_total
```

- [ ] **Step 6: Commit any final fixes**

Run:

```bash
git status --short
git add services/api-go pipelines Makefile docs/ARCHITECTURE.md docs/RUNBOOK.md
git commit -m "chore: verify search scale rehearsal"
```

Skip this commit if `git status --short` shows no remaining implementation changes.
