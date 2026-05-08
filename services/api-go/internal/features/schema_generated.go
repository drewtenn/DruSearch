// Generated from libs/schema/feature_schema.json by libs/schema/codegen.py.
// DO NOT EDIT BY HAND. Run `make check-feature-parity` after schema changes.

package features

const SchemaVersion = "v1"

const (
	GenBm25Score = 0
	GenBm25Rank = 1
	GenKnnScore = 2
	GenKnnRank = 3
	GenRrfScore = 4
	GenPopularityPrior = 5
	GenPriceLogCents = 6
	GenTitleLengthTokens = 7
	GenQueryLengthTokens = 8
	GenQueryHasBrand = 9
	GenQueryHasColor = 10
	GenQueryHasSizePattern = 11
	GenUserBrandAffinity = 12

	GenNumFeatures = 13
)

// GenNames is the ordered feature-name list aligned with the indices above.
var GenNames = [...]string{
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
