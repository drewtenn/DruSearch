package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"go.uber.org/zap"

	"github.com/drewtenn/drusearch/services/api-go/internal/embedder"
	"github.com/drewtenn/drusearch/services/api-go/internal/eventbus"
	"github.com/drewtenn/drusearch/services/api-go/internal/features"
	"github.com/drewtenn/drusearch/services/api-go/internal/neuralrerank"
	"github.com/drewtenn/drusearch/services/api-go/internal/obs"
	"github.com/drewtenn/drusearch/services/api-go/internal/rerank"
	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
)

type SearchResult struct {
	ProductID    string   `json:"product_id"`
	Title        string   `json:"title"`
	Brand        string   `json:"brand"`
	Color        string   `json:"color"`
	Category     string   `json:"category,omitempty"`
	CategoryPath []string `json:"category_path,omitempty"`
	PriceCents   int      `json:"price_cents"`
	Score        float64  `json:"score"`
	Explain      explain  `json:"explain"`
}

type explain struct {
	BM25     float64 `json:"bm25"`
	BM25Rank int     `json:"bm25_rank"`
	KNN      float64 `json:"knn"`
	KNNRank  int     `json:"knn_rank"`
	RRF      float64 `json:"rrf"`
	LTR      float64 `json:"ltr,omitempty"`
	LTRRank  int     `json:"ltr_rank,omitempty"`
	BGE      float64 `json:"bge,omitempty"`
	BGERank  int     `json:"bge_rank,omitempty"`
}

type SearchResponse struct {
	QueryID      string         `json:"query_id"`
	Query        string         `json:"query"`
	SessionID    string         `json:"session_id"`
	Mode         string         `json:"mode"`
	ModelVersion string         `json:"model_version,omitempty"`
	Results      []SearchResult `json:"results"`
	TookMs       int64          `json:"took_ms"`
}

const (
	defaultK     = 20
	maxK         = 100
	defaultCandN = 200
	rrfK         = 60
)

func (s *Server) search(w http.ResponseWriter, r *http.Request) {
	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "missing query parameter q"})
		return
	}
	ranker, err := rankerFromRequest(r.URL.Query().Get("ranker"), s.DefaultRanker)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	k := defaultK
	if v := r.URL.Query().Get("k"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= maxK {
			k = n
		}
	}

	userID := r.URL.Query().Get("user_id")
	sessionID := r.URL.Query().Get("session_id")
	if sessionID == "" {
		sessionID = "anon-" + newQueryID()
	}

	start := time.Now()

	mode := "hybrid"
	embedStart := time.Now()
	vec, err := s.Embedder.Embed(r.Context(), q)
	obs.StageLatency.WithLabelValues("embed").Observe(time.Since(embedStart).Seconds())
	if err != nil {
		// Circuit-open + sidecar errors both degrade gracefully.
		if errors.Is(err, embedder.ErrCircuitOpen) {
			s.Logger.Warn("embedder circuit open; degrading to bm25-only")
		} else {
			s.Logger.Warn("embedder failed; degrading to bm25-only", zap.Error(err))
		}
		mode = "bm25"
	}

	retrStart := time.Now()
	var hits []retrieval.Hit
	if mode == "hybrid" {
		hits, err = s.Retrieval.Hybrid(r.Context(), q, vec, defaultCandN, rrfK)
	} else {
		hits, err = s.Retrieval.BM25(r.Context(), q, defaultCandN)
	}
	obs.StageLatency.WithLabelValues("retrieve").Observe(time.Since(retrStart).Seconds())
	if err != nil {
		s.Logger.Error("retrieval failed", zap.String("q", q), zap.String("mode", mode), zap.Error(err))
		obs.SearchTotal.WithLabelValues(mode, "error").Inc()
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "search failed"})
		return
	}
	obs.CandidatesPerRequest.Observe(float64(len(hits)))

	rerankStart := time.Now()
	var (
		scored       []rerank.ScoredHit
		modelVersion string
		usedRanker   rankerMode
	)
	switch ranker {
	case rankerLTR:
		userStart := time.Now()
		userFeats := features.LoadUserFeatures(r.Context(), s.Stores.RDB, userID)
		obs.StageLatency.WithLabelValues("user_features").Observe(time.Since(userStart).Seconds())
		var used bool
		scored, modelVersion, used = s.maybeRerank(q, hits, userFeats)
		if used {
			usedRanker = rankerLTR
		}
	case rankerBGE:
		var used bool
		scored, modelVersion, used = s.maybeBGERerank(r.Context(), q, hits, k)
		if used {
			usedRanker = rankerBGE
		}
	default:
		scored = wrapHits(hits)
	}
	obs.StageLatency.WithLabelValues("rerank").Observe(time.Since(rerankStart).Seconds())
	if usedRanker != "" {
		mode = mode + "+" + string(usedRanker)
	}

	if len(scored) > k {
		scored = scored[:k]
	}

	queryID := newQueryID()
	results := make([]SearchResult, 0, len(scored))
	for i, sh := range scored {
		// pick the headline score: LTR if applied, else RRF/BM25
		score := sh.RRF
		switch {
		case usedRanker == rankerLTR:
			score = sh.LTR
		case usedRanker == rankerBGE:
			score = sh.BGE
		case mode == "bm25":
			score = sh.BM25
		}
		results = append(results, SearchResult{
			ProductID:    sh.ProductID,
			Title:        sh.Title,
			Brand:        sh.Brand,
			Color:        sh.Color,
			Category:     sh.Category,
			CategoryPath: sh.CategoryPath,
			PriceCents:   sh.PriceCents,
			Score:        score,
			Explain: explain{
				BM25:     sh.BM25,
				BM25Rank: sh.BM25Rank,
				KNN:      sh.KNN,
				KNNRank:  sh.KNNRank,
				RRF:      sh.RRF,
				LTR:      sh.LTR,
				LTRRank:  sh.LTRRank,
				BGE:      sh.BGE,
				BGERank:  sh.BGERank,
			},
		})

		// async impression event; one per returned product.
		if s.Bus != nil {
			scoreMap := map[string]float64{
				"bm25": sh.BM25,
				"knn":  sh.KNN,
				"rrf":  sh.RRF,
			}
			if usedRanker == rankerLTR {
				scoreMap["ltr"] = sh.LTR
			}
			if usedRanker == rankerBGE {
				scoreMap["bge"] = sh.BGE
			}
			s.Bus.Submit(eventbus.Event{
				Type:      "impression",
				UserID:    userID,
				SessionID: sessionID,
				Query:     q,
				QueryID:   queryID,
				ProductID: sh.ProductID,
				Position:  i,
				Scores:    scoreMap,
				Source:    "real",
			})
		}
	}

	took := time.Since(start)
	obs.SearchLatency.WithLabelValues(mode).Observe(took.Seconds())
	obs.SearchTotal.WithLabelValues(mode, "ok").Inc()

	writeJSON(w, http.StatusOK, SearchResponse{
		QueryID:      queryID,
		Query:        q,
		SessionID:    sessionID,
		Mode:         mode,
		ModelVersion: modelVersion,
		Results:      results,
		TookMs:       took.Milliseconds(),
	})
}

// maybeRerank applies the LTR model if one is loaded. Returns the (possibly
// re-sorted) scored hits, the loaded model version label, and whether
// re-ranking actually happened.
func (s *Server) maybeRerank(q string, hits []retrieval.Hit, user *features.UserFeatures) ([]rerank.ScoredHit, string, bool) {
	if s.Reranker == nil || s.Vocab == nil {
		return wrapHits(hits), "", false
	}
	loaded := s.Reranker.Get()
	if loaded == nil {
		return wrapHits(hits), "", false
	}
	scored, err := rerank.Apply(loaded, q, hits, s.Vocab, user)
	if err != nil {
		s.Logger.Warn("rerank failed; falling back to retrieval order", zap.Error(err))
		return wrapHits(hits), "", false
	}
	version := ""
	if v, ok := loaded.Meta["version"]; ok {
		version = toString(v)
	}
	return scored, version, true
}

func (s *Server) maybeBGERerank(ctx context.Context, q string, hits []retrieval.Hit, k int) ([]rerank.ScoredHit, string, bool) {
	if s.Neural == nil || len(hits) == 0 {
		return wrapHits(hits), "", false
	}
	limit := s.NeuralRerankCandidates
	if limit <= 0 {
		limit = 50
	}
	if k > limit {
		limit = k
	}
	if limit > len(hits) {
		limit = len(hits)
	}
	docs := make([]neuralrerank.Document, 0, limit)
	for _, h := range hits[:limit] {
		docs = append(docs, neuralrerank.Document{
			ID:   h.ProductID,
			Text: neuralrerank.ProductText(h),
		})
	}
	scores, model, err := s.Neural.Rerank(ctx, q, docs)
	if err != nil {
		s.Logger.Warn("BGE rerank failed; falling back to retrieval order", zap.Error(err))
		return wrapHits(hits), "", false
	}
	byID := make(map[string]float64, len(scores))
	for _, sc := range scores {
		byID[sc.ID] = sc.Score
	}
	out := wrapHits(hits)
	for i := 0; i < limit; i++ {
		score, ok := byID[out[i].ProductID]
		if !ok {
			score = math.Inf(-1)
		}
		out[i].BGE = score
	}
	sort.SliceStable(out[:limit], func(i, j int) bool {
		return out[i].BGE > out[j].BGE
	})
	for i := 0; i < limit; i++ {
		out[i].BGERank = i + 1
	}
	return out, model, true
}

func wrapHits(hits []retrieval.Hit) []rerank.ScoredHit {
	out := make([]rerank.ScoredHit, len(hits))
	for i, x := range hits {
		out[i] = rerank.ScoredHit{Hit: x}
	}
	return out
}

func toString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	default:
		return ""
	}
}

func (s *Server) productByID(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "missing product id"})
		return
	}
	p, err := s.Products.ByID(r.Context(), id)
	if err != nil {
		s.Logger.Error("product fetch", zap.String("id", id), zap.Error(err))
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "lookup failed"})
		return
	}
	if p == nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"error": "product not found"})
		return
	}
	writeJSON(w, http.StatusOK, p)
}

func newQueryID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return time.Now().UTC().Format("20060102150405.000000000")
	}
	return hex.EncodeToString(b[:])
}
