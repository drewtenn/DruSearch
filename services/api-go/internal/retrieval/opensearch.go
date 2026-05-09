package retrieval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"unicode"

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
	ProductID        string   `json:"product_id"`
	Title            string   `json:"title"`
	Brand            string   `json:"brand"`
	Color            string   `json:"color"`
	Category         string   `json:"category"`
	CategoryPath     []string `json:"category_path"`
	DerivedGender    string   `json:"derived_gender"`
	PriceCents       int      `json:"price_cents"`
	PopularityPrior  float64  `json:"popularity_prior"`
	BM25             float64  `json:"bm25"`
	BM25Rank         int      `json:"bm25_rank"`
	KNN              float64  `json:"knn"`
	KNNRank          int      `json:"knn_rank"`
	RRF              float64  `json:"rrf"`
	TitleBM25        float64  `json:"title_bm25"`
	CategoryPathBM25 float64  `json:"category_path_bm25"`
	CategoryBM25     float64  `json:"category_bm25"`
	BulletsBM25      float64  `json:"bullets_bm25"`
	DescriptionBM25  float64  `json:"description_bm25"`
	BrandBM25        float64  `json:"brand_bm25"`
}

type osSource struct {
	ProductID       string   `json:"product_id"`
	Title           string   `json:"title"`
	Brand           string   `json:"brand"`
	Color           string   `json:"color"`
	Category        string   `json:"category"`
	CategoryPath    []string `json:"category_path"`
	DerivedGender   string   `json:"derived_gender"`
	PriceCents      int      `json:"price_cents"`
	PopularityPrior float64  `json:"popularity_prior"`
}

type rawHit struct {
	ID            string
	Score         float64
	Source        osSource
	MatchedScores map[string]float64
}

var fieldScoreQueryNames = []string{
	"bm25_title",
	"bm25_category_path",
	"bm25_category",
	"bm25_bullets",
	"bm25_description",
	"bm25_brand",
}

func fieldScoreNames() []string {
	return fieldScoreQueryNames
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
				ID             string          `json:"_id"`
				Score          float64         `json:"_score"`
				Source         json.RawMessage `json:"_source"`
				MatchedQueries json.RawMessage `json:"matched_queries"`
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
		out = append(out, rawHit{
			ID:            h.ID,
			Score:         h.Score,
			Source:        src,
			MatchedScores: parseMatchedQueryScores(h.MatchedQueries),
		})
	}
	return out, nil
}

func parseMatchedQueryScores(raw json.RawMessage) map[string]float64 {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var scores map[string]float64
	if err := json.Unmarshal(raw, &scores); err == nil {
		return scores
	}
	var names []string
	if err := json.Unmarshal(raw, &names); err == nil {
		out := make(map[string]float64, len(names))
		for _, name := range names {
			out[name] = 1.0
		}
		return out
	}
	return nil
}

func sourceFields() []string {
	return []string{"product_id", "title", "brand", "color", "category", "category_path", "derived_gender", "price_cents", "popularity_prior"}
}

func buildBM25Body(query string, k int) ([]byte, error) {
	baseQuery := lexicalBM25Query(query)
	if intent := genderIntentLabel(query); intent != "" {
		baseQuery = genderBoostingQuery(baseQuery, intent)
	}
	return json.Marshal(map[string]any{
		"size":                        k,
		"_source":                     sourceFields(),
		"include_named_queries_score": true,
		"query":                       baseQuery,
	})
}

func lexicalBM25Query(query string) map[string]any {
	fieldMatch := func(name, field string, boost float64) map[string]any {
		return map[string]any{
			"match": map[string]any{
				field: map[string]any{
					"query": query,
					"boost": boost,
					"_name": name,
				},
			},
		}
	}
	return map[string]any{
		"bool": map[string]any{
			"must": []any{
				map[string]any{
					"dis_max": map[string]any{
						"tie_breaker": 0.1,
						"queries": []any{
							fieldMatch("bm25_title", "title", 2.0),
							fieldMatch("bm25_category_path", "category_path", 2.0),
							fieldMatch("bm25_category", "category", 1.5),
							fieldMatch("bm25_bullets", "bullets", 1.0),
							fieldMatch("bm25_description", "description", 1.0),
						},
					},
				},
			},
			"should": []any{
				map[string]any{
					"match": map[string]any{
						"brand.text": map[string]any{
							"query":          query,
							"boost":          2.5,
							"fuzziness":      "AUTO",
							"prefix_length":  1,
							"max_expansions": 20,
							"_name":          "bm25_brand",
						},
					},
				},
			},
		},
	}
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
			ProductID:        r.Source.ProductID,
			Title:            r.Source.Title,
			Brand:            r.Source.Brand,
			Color:            r.Source.Color,
			Category:         r.Source.Category,
			CategoryPath:     r.Source.CategoryPath,
			DerivedGender:    r.Source.DerivedGender,
			PriceCents:       r.Source.PriceCents,
			PopularityPrior:  r.Source.PopularityPrior,
			BM25:             r.Score,
			BM25Rank:         i + 1,
			TitleBM25:        r.MatchedScores["bm25_title"],
			CategoryPathBM25: r.MatchedScores["bm25_category_path"],
			CategoryBM25:     r.MatchedScores["bm25_category"],
			BulletsBM25:      r.MatchedScores["bm25_bullets"],
			DescriptionBM25:  r.MatchedScores["bm25_description"],
			BrandBM25:        r.MatchedScores["bm25_brand"],
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
		src         osSource
		bm25        float64
		knn         float64
		bm25Rank    int
		knnRank     int
		rrf         float64
		fieldScores map[string]float64
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
		f.fieldScores = h.MatchedScores
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
			ProductID:        f.src.ProductID,
			Title:            f.src.Title,
			Brand:            f.src.Brand,
			Color:            f.src.Color,
			Category:         f.src.Category,
			CategoryPath:     f.src.CategoryPath,
			DerivedGender:    f.src.DerivedGender,
			PriceCents:       f.src.PriceCents,
			PopularityPrior:  f.src.PopularityPrior,
			BM25:             f.bm25,
			BM25Rank:         f.bm25Rank,
			KNN:              f.knn,
			KNNRank:          f.knnRank,
			RRF:              f.rrf,
			TitleBM25:        f.fieldScores["bm25_title"],
			CategoryPathBM25: f.fieldScores["bm25_category_path"],
			CategoryBM25:     f.fieldScores["bm25_category"],
			BulletsBM25:      f.fieldScores["bm25_bullets"],
			DescriptionBM25:  f.fieldScores["bm25_description"],
			BrandBM25:        f.fieldScores["bm25_brand"],
		})
	}
	// Sort by RRF desc.
	sortHitsByRRF(hits)
	return hits, nil
}

func genderBoostingQuery(baseQuery map[string]any, intent string) map[string]any {
	should := []any{
		termBoost("derived_gender", intent, 8.0),
	}
	if intent == "men" || intent == "women" {
		should = append(should, termBoost("derived_gender", "unisex", 4.0))
	}
	return map[string]any{
		"boosting": map[string]any{
			"positive": map[string]any{
				"bool": map[string]any{
					"must":   []any{baseQuery},
					"should": should,
				},
			},
			"negative": map[string]any{
				"terms": map[string]any{
					"derived_gender": oppositeGenderLabels(intent),
				},
			},
			"negative_boost": 0.25,
		},
	}
}

func termBoost(field, value string, boost float64) map[string]any {
	return map[string]any{
		"term": map[string]any{
			field: map[string]any{
				"value": value,
				"boost": boost,
			},
		},
	}
}

func oppositeGenderLabels(intent string) []string {
	switch intent {
	case "men":
		return []string{"women", "boys", "girls"}
	case "women":
		return []string{"men", "boys", "girls"}
	case "boys":
		return []string{"men", "women", "girls", "unisex"}
	case "girls":
		return []string{"men", "women", "boys", "unisex"}
	case "unisex":
		return []string{"men", "women", "boys", "girls"}
	default:
		return nil
	}
}

func genderIntentLabel(query string) string {
	found := ""
	for _, token := range tokenize(query) {
		label := genderTokenLabel(token)
		if label == "" {
			continue
		}
		if found != "" && found != label {
			return ""
		}
		found = label
	}
	return found
}

func genderTokenLabel(token string) string {
	switch token {
	case "men", "mens", "man", "male":
		return "men"
	case "women", "womens", "woman", "female":
		return "women"
	case "boys", "boy":
		return "boys"
	case "girls", "girl":
		return "girls"
	case "unisex":
		return "unisex"
	default:
		return ""
	}
}

func tokenize(s string) []string {
	if s == "" {
		return nil
	}
	out := make([]string, 0, 8)
	var sb strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			sb.WriteRune(unicode.ToLower(r))
			continue
		}
		if sb.Len() > 0 {
			out = append(out, sb.String())
			sb.Reset()
		}
	}
	if sb.Len() > 0 {
		out = append(out, sb.String())
	}
	return out
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
