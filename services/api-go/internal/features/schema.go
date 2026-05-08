// Package features holds the LTR feature schema and reference transforms.
//
// The constants below are the hand-written, idiomatic Go view of the
// schema. They are kept honest by schema_generated.go (produced by
// libs/schema/codegen.py from libs/schema/feature_schema.json) and the
// drift-guard test in schema_test.go. To extend the schema: edit the
// JSON, run `make check-feature-parity`, then update Names/Idx*
// constants here and in pipelines/pipelines/features/__init__.py.
package features

const (
	IdxBM25Score          = 0
	IdxBM25Rank           = 1
	IdxKNNScore           = 2
	IdxKNNRank            = 3
	IdxRRFScore           = 4
	IdxPopularityPrior    = 5
	IdxPriceLogCents      = 6
	IdxTitleLengthTokens  = 7
	IdxQueryLengthTokens  = 8
	IdxQueryHasBrand      = 9
	IdxQueryHasColor      = 10
	IdxQueryHasSizePat    = 11
	IdxUserBrandAffinity  = 12

	NumFeatures = 13
)

// Names ordered by their feature index above.
var Names = [...]string{
	"bm25_score",
	"bm25_rank",
	"knn_score",
	"knn_rank",
	"rrf_score",
	"popularity_prior",
	"price_log_cents",
	"title_length_tokens",
	"query_length_tokens",
	"query_has_brand",
	"query_has_color",
	"query_has_size_pattern",
	"user_brand_affinity",
}
