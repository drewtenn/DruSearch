// Generated from libs/schema/feature_schema.json by libs/schema/codegen.py.
// DO NOT EDIT BY HAND. Run `make check-feature-parity` after schema changes.

package features

const SchemaVersion = "v3"

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
	GenQueryHasCategoryToken = 11
	GenQueryHasSizePattern = 12
	GenQueryGenderIntent = 13
	GenProductGender = 14
	GenGenderIntentMatch = 15
	GenGenderIntentMismatch = 16
	GenProductBrandMatch = 17
	GenProductBrandTokenOverlap = 18
	GenProductColorMatch = 19
	GenTitleQueryTokenCoverage = 20
	GenCategoryQueryTokenCoverage = 21
	GenProductCategoryTokenOverlap = 22
	GenTitleExactQueryMatch = 23
	GenUserBrandAffinity = 24

	GenNumFeatures = 25
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
	"query_has_category_token",
	"query_has_size_pattern",
	"query_gender_intent",
	"product_gender",
	"gender_intent_match",
	"gender_intent_mismatch",
	"product_brand_match",
	"product_brand_token_overlap",
	"product_color_match",
	"title_query_token_coverage",
	"category_query_token_coverage",
	"product_category_token_overlap",
	"title_exact_query_match",
	"user_brand_affinity",
}
