package retrieval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

	opensearch "github.com/opensearch-project/opensearch-go/v4"
)

type Engine struct {
	client *opensearch.Client
	index  string
}

func New(c *opensearch.Client, index string) *Engine {
	return &Engine{client: c, index: index}
}

type Hit struct {
	ProductID       string   `json:"product_id"`
	Title           string   `json:"title"`
	Brand           string   `json:"brand"`
	Color           string   `json:"color"`
	Category        string   `json:"category"`
	CategoryPath    []string `json:"category_path"`
	PriceCents      int      `json:"price_cents"`
	PopularityPrior float64  `json:"popularity_prior"`
	BM25            float64  `json:"bm25"`
	BM25Rank        int      `json:"bm25_rank"`
	KNN             float64  `json:"knn"`
	KNNRank         int      `json:"knn_rank"`
	RRF             float64  `json:"rrf"`
}

type osSource struct {
	ProductID       string   `json:"product_id"`
	Title           string   `json:"title"`
	Brand           string   `json:"brand"`
	Color           string   `json:"color"`
	Category        string   `json:"category"`
	CategoryPath    []string `json:"category_path"`
	PriceCents      int      `json:"price_cents"`
	PopularityPrior float64  `json:"popularity_prior"`
}

type rawHit struct {
	ID     string
	Score  float64
	Source osSource
}

func (e *Engine) doSearch(ctx context.Context, body []byte) ([]rawHit, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "/"+e.index+"/_search", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Perform(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("opensearch status=%d: %s", resp.StatusCode, string(b))
	}
	var raw struct {
		Hits struct {
			Hits []struct {
				ID     string          `json:"_id"`
				Score  float64         `json:"_score"`
				Source json.RawMessage `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}
	out := make([]rawHit, 0, len(raw.Hits.Hits))
	for _, h := range raw.Hits.Hits {
		var src osSource
		if err := json.Unmarshal(h.Source, &src); err != nil {
			return nil, err
		}
		if src.ProductID == "" {
			src.ProductID = h.ID
		}
		out = append(out, rawHit{ID: h.ID, Score: h.Score, Source: src})
	}
	return out, nil
}

func sourceFields() []string {
	return []string{"product_id", "title", "brand", "color", "category", "category_path", "price_cents", "popularity_prior"}
}

func buildBM25Body(query string, k int) ([]byte, error) {
	baseQuery := map[string]any{
		"multi_match": map[string]any{
			"query":  query,
			"fields": []string{"title^2", "category_path^2", "category^1.5", "bullets", "description"},
		},
	}
	return json.Marshal(map[string]any{
		"size":    k,
		"_source": sourceFields(),
		"query":   baseQuery,
	})
}

func (e *Engine) bm25Raw(ctx context.Context, query string, k int) ([]rawHit, error) {
	body, err := buildBM25Body(query, k)
	if err != nil {
		return nil, err
	}
	return e.doSearch(ctx, body)
}

func buildKNNBody(query string, vec []float32, k int) ([]byte, error) {
	titleVec := map[string]any{
		"vector": vec,
		"k":      k,
	}
	return json.Marshal(map[string]any{
		"size":    k,
		"_source": sourceFields(),
		"query": map[string]any{
			"knn": map[string]any{
				"title_vec": titleVec,
			},
		},
	})
}

func (e *Engine) knnRaw(ctx context.Context, query string, vec []float32, k int) ([]rawHit, error) {
	body, err := buildKNNBody(query, vec, k)
	if err != nil {
		return nil, err
	}
	return e.doSearch(ctx, body)
}

func (e *Engine) BM25(ctx context.Context, query string, k int) ([]Hit, error) {
	raws, err := e.bm25Raw(ctx, query, k)
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
			BM25:            r.Score,
			BM25Rank:        i + 1,
		})
	}
	return hits, nil
}

// Hybrid runs BM25 and k-NN in parallel and fuses with RRF.
// candN is the per-side candidate pool; the merged set is sorted by RRF and
// the entire fused list is returned (caller decides how many to keep).
//
// RRF: score(d) = sum_q 1 / (rrfK + rank_q(d)),  rrfK=60 by convention.
func (e *Engine) Hybrid(ctx context.Context, query string, vec []float32, candN, rrfK int) ([]Hit, error) {
	if rrfK <= 0 {
		rrfK = 60
	}

	var (
		wg              sync.WaitGroup
		bm25, knn       []rawHit
		bm25Err, knnErr error
	)
	wg.Add(2)
	go func() { defer wg.Done(); bm25, bm25Err = e.bm25Raw(ctx, query, candN) }()
	go func() { defer wg.Done(); knn, knnErr = e.knnRaw(ctx, query, vec, candN) }()
	wg.Wait()

	switch {
	case bm25Err != nil && knnErr != nil:
		return nil, fmt.Errorf("bm25=%v knn=%v", bm25Err, knnErr)
	case bm25Err != nil:
		return nil, fmt.Errorf("bm25 retrieval failed: %w", bm25Err)
	case knnErr != nil:
		return nil, fmt.Errorf("knn retrieval failed: %w", knnErr)
	}

	type fused struct {
		src      osSource
		bm25     float64
		knn      float64
		bm25Rank int
		knnRank  int
		rrf      float64
	}
	merged := make(map[string]*fused, candN)
	missingRank := candN + 1

	for i, h := range bm25 {
		f, ok := merged[h.ID]
		if !ok {
			f = &fused{src: h.Source, bm25Rank: missingRank, knnRank: missingRank}
			merged[h.ID] = f
		}
		f.bm25 = h.Score
		f.bm25Rank = i + 1
		f.rrf += 1.0 / float64(rrfK+i+1)
	}
	for i, h := range knn {
		f, ok := merged[h.ID]
		if !ok {
			f = &fused{src: h.Source, bm25Rank: missingRank, knnRank: missingRank}
			merged[h.ID] = f
		}
		f.knn = h.Score
		f.knnRank = i + 1
		f.rrf += 1.0 / float64(rrfK+i+1)
	}

	hits := make([]Hit, 0, len(merged))
	for _, f := range merged {
		hits = append(hits, Hit{
			ProductID:       f.src.ProductID,
			Title:           f.src.Title,
			Brand:           f.src.Brand,
			Color:           f.src.Color,
			Category:        f.src.Category,
			CategoryPath:    f.src.CategoryPath,
			PriceCents:      f.src.PriceCents,
			PopularityPrior: f.src.PopularityPrior,
			BM25:            f.bm25,
			BM25Rank:        f.bm25Rank,
			KNN:             f.knn,
			KNNRank:         f.knnRank,
			RRF:             f.rrf,
		})
	}
	// Sort by RRF desc.
	sortHitsByRRF(hits)
	return hits, nil
}

func sortHitsByRRF(h []Hit) {
	// insertion sort is fine for small N; switch to sort.Slice on growth
	for i := 1; i < len(h); i++ {
		j := i
		for j > 0 && h[j-1].RRF < h[j].RRF {
			h[j-1], h[j] = h[j], h[j-1]
			j--
		}
	}
}
