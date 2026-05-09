// Generated from libs/schema/feature_schema.json by libs/schema/codegen.py.
// DO NOT EDIT BY HAND. Run `make check-feature-parity` after schema changes.

package features

const SchemaVersion = "v8"

const (
	GenBm25Score = 0
	GenBm25Rank = 1
	GenTitleBm25Score = 2
	GenCategoryPathBm25Score = 3
	GenCategoryBm25Score = 4
	GenBulletsBm25Score = 5
	GenDescriptionBm25Score = 6
	GenBrandBm25Score = 7
	GenKnnScore = 8
	GenKnnRank = 9
	GenRrfScore = 10
	GenPopularityPrior = 11
	GenPriceLogCents = 12
	GenTitleLengthTokens = 13
	GenQueryLengthTokens = 14
	GenQueryHasBrand = 15
	GenQueryHasColor = 16
	GenQueryHasCategoryToken = 17
	GenQueryHasSizePattern = 18
	GenQueryGenderIntent = 19
	GenProductGender = 20
	GenGenderIntentMatch = 21
	GenGenderIntentMismatch = 22
	GenProductBrandMatch = 23
	GenProductBrandTokenOverlap = 24
	GenProductColorMatch = 25
	GenTitleQueryTokenCoverage = 26
	GenCategoryQueryTokenCoverage = 27
	GenProductCategoryTokenOverlap = 28
	GenTitleExactQueryMatch = 29
	GenUserBrandAffinity = 30
	GenQueryAffordabilityIntent = 31
	GenAffordabilityPriceScore = 32
	GenBrandFamilyMatch = 33
	GenSubbrandTitleMatch = 34

	GenNumFeatures = 35
)

// GenNames is the ordered feature-name list aligned with the indices above.
var GenNames = [...]string{
	"bm25_score",
	"bm25_rank",
	"title_bm25_score",
	"category_path_bm25_score",
	"category_bm25_score",
	"bullets_bm25_score",
	"description_bm25_score",
	"brand_bm25_score",
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
	"query_affordability_intent",
	"affordability_price_score",
	"brand_family_match",
	"subbrand_title_match",
}
