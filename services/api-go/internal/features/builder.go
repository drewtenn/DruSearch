package features

import (
	"context"
	"math"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
)

// Vocab is the catalog-derived token vocabulary used by interaction features.
type Vocab struct {
	Brand map[string]struct{}
	Color map[string]struct{}
}

// LoadVocab reads brand and color values from products and tokenizes them
// into lowercase token sets. Both Python and Go training/serving must use
// the same source.
func LoadVocab(ctx context.Context, pool *pgxpool.Pool) (*Vocab, error) {
	v := &Vocab{
		Brand: make(map[string]struct{}, 1024),
		Color: make(map[string]struct{}, 1024),
	}
	rows, err := pool.Query(ctx,
		"SELECT COALESCE(brand,''), COALESCE(color,'') FROM products")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var brand, color string
		if err := rows.Scan(&brand, &color); err != nil {
			return nil, err
		}
		for _, t := range Tokenize(brand) {
			v.Brand[t] = struct{}{}
		}
		for _, t := range Tokenize(color) {
			v.Color[t] = struct{}{}
		}
	}
	return v, rows.Err()
}

// BuildMatrix turns a slice of retrieval hits into a (nrows × NumFeatures)
// row-major dense matrix that LightGBM can score.
//
// Per-row inputs come from the hit (retrieval scores + product fields).
// Per-query inputs (token counts, brand/color/size flags) are computed
// once and broadcast across rows. Per-user inputs (brand affinity) come
// from a per-request UserFeatures snapshot.
func BuildMatrix(query string, hits []retrieval.Hit, vocab *Vocab, user *UserFeatures) []float64 {
	n := len(hits)
	out := make([]float64, n*NumFeatures)

	qLen := QueryLengthTokens(query)
	qBrand := QueryHasAny(query, vocab.Brand)
	qColor := QueryHasAny(query, vocab.Color)
	qSize := QueryHasSizePattern(query)
	qGender := QueryGenderIntent(query)

	var brandAff map[string]float64
	if user != nil {
		brandAff = user.BrandAff
	}

	for i, h := range hits {
		off := i * NumFeatures
		out[off+IdxBM25Score] = h.BM25
		out[off+IdxBM25Rank] = float64(h.BM25Rank)
		out[off+IdxKNNScore] = h.KNN
		out[off+IdxKNNRank] = float64(h.KNNRank)
		out[off+IdxRRFScore] = h.RRF
		out[off+IdxPopularityPrior] = h.PopularityPrior
		out[off+IdxPriceLogCents] = math.Log1p(float64(h.PriceCents))
		out[off+IdxTitleLengthTokens] = float64(len(Tokenize(h.Title)))
		out[off+IdxQueryLengthTokens] = qLen
		out[off+IdxQueryHasBrand] = qBrand
		out[off+IdxQueryHasColor] = qColor
		out[off+IdxQueryHasSizePat] = qSize
		pGender := ProductGender(h.CategoryPath)
		out[off+IdxQueryGenderIntent] = qGender
		out[off+IdxProductGender] = pGender
		out[off+IdxGenderIntentMatch] = GenderIntentMatch(qGender, pGender)
		out[off+IdxGenderIntentMis] = GenderIntentMismatch(qGender, pGender)
		if brandAff != nil {
			out[off+IdxUserBrandAffinity] = brandAff[h.Brand]
		}
	}
	return out
}
