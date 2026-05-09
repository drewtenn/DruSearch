package features

import (
	"context"
	"math"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
)

// Vocab is the catalog-derived token vocabulary used by interaction features.
type Vocab struct {
	Brand    map[string]struct{}
	Color    map[string]struct{}
	Category map[string]struct{}
}

// LoadVocab reads brand and color values from products and tokenizes them
// into lowercase token sets. Both Python and Go training/serving must use
// the same source.
func LoadVocab(ctx context.Context, pool *pgxpool.Pool) (*Vocab, error) {
	v := &Vocab{
		Brand:    make(map[string]struct{}, 1024),
		Color:    make(map[string]struct{}, 1024),
		Category: make(map[string]struct{}, 4096),
	}
	rows, err := pool.Query(ctx,
		"SELECT COALESCE(brand,''), COALESCE(color,''), COALESCE(category_path, ARRAY[]::TEXT[]) FROM products")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var brand, color string
		var categoryPath []string
		if err := rows.Scan(&brand, &color, &categoryPath); err != nil {
			return nil, err
		}
		for _, t := range BrandTokens(brand) {
			v.Brand[t] = struct{}{}
		}
		for _, t := range Tokenize(color) {
			v.Color[t] = struct{}{}
		}
		for _, part := range categoryPath {
			for _, t := range Tokenize(part) {
				v.Category[t] = struct{}{}
			}
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

	if vocab == nil {
		vocab = &Vocab{}
	}
	qLen := QueryLengthTokens(query)
	qBrand := QueryHasAny(query, vocab.Brand)
	qColor := QueryHasAny(query, vocab.Color)
	qCategory := QueryHasAny(query, vocab.Category)
	qSize := QueryHasSizePattern(query)
	qAffordability := QueryAffordabilityIntent(query)
	qGender := QueryGenderIntent(query)

	var brandAff map[string]float64
	if user != nil {
		brandAff = user.BrandAff
	}
	missingRank := missingRetrievalRank(hits)

	for i, h := range hits {
		off := i * NumFeatures
		out[off+IdxBM25Score] = h.BM25
		out[off+IdxBM25Rank] = float64(rankFeature(h.BM25Rank, missingRank))
		out[off+IdxKNNScore] = h.KNN
		out[off+IdxKNNRank] = float64(rankFeature(h.KNNRank, missingRank))
		out[off+IdxRRFScore] = h.RRF
		out[off+IdxPopularityPrior] = h.PopularityPrior
		out[off+IdxPriceLogCents] = math.Log1p(float64(h.PriceCents))
		out[off+IdxTitleLengthTokens] = float64(len(Tokenize(h.Title)))
		out[off+IdxQueryLengthTokens] = qLen
		out[off+IdxQueryHasBrand] = qBrand
		out[off+IdxQueryHasColor] = qColor
		out[off+IdxQueryHasCategoryToken] = qCategory
		out[off+IdxQueryHasSizePat] = qSize
		pGender := ProductGender(h.CategoryPath)
		out[off+IdxQueryGenderIntent] = qGender
		out[off+IdxProductGender] = pGender
		out[off+IdxGenderIntentMatch] = GenderIntentMatch(qGender, pGender)
		out[off+IdxGenderIntentMis] = GenderIntentMismatch(qGender, pGender)
		out[off+IdxProductBrandMatch] = ProductBrandMatch(query, h.Brand)
		out[off+IdxProductBrandTokenOverlap] = ProductBrandTokenOverlap(query, h.Brand)
		out[off+IdxProductColorMatch] = ProductColorMatch(query, h.Color)
		categoryText := strings.Join(h.CategoryPath, " ")
		out[off+IdxTitleQueryTokenCoverage] = QueryTokenCoverage(query, h.Title, vocab.Brand)
		out[off+IdxCategoryQueryTokenCoverage] = QueryTokenCoverage(query, categoryText, vocab.Brand)
		out[off+IdxProductCategoryTokenOverlap] = TokenOverlapFraction(query, categoryText)
		out[off+IdxTitleExactQueryMatch] = ExactQueryPhraseMatch(query, h.Title)
		if brandAff != nil {
			out[off+IdxUserBrandAffinity] = brandAff[h.Brand]
		}
		out[off+IdxQueryAffordabilityIntent] = qAffordability
		out[off+IdxAffordabilityPriceScore] = AffordabilityPriceScore(qAffordability, h.PriceCents)
		out[off+IdxBrandFamilyMatch] = BrandFamilyMatch(query, h.Brand, h.Title)
		out[off+IdxSubbrandTitleMatch] = SubbrandTitleMatch(query, h.Title)
	}
	return out
}

func missingRetrievalRank(hits []retrieval.Hit) int {
	maxRank := 0
	for _, h := range hits {
		if h.BM25Rank > maxRank {
			maxRank = h.BM25Rank
		}
		if h.KNNRank > maxRank {
			maxRank = h.KNNRank
		}
	}
	if maxRank <= 0 {
		return len(hits) + 1
	}
	return maxRank + 1
}

func rankFeature(rank, missingRank int) int {
	if rank > 0 {
		return rank
	}
	return missingRank
}
