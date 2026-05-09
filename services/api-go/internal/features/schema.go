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
	IdxBM25Score                   = 0
	IdxBM25Rank                    = 1
	IdxKNNScore                    = 2
	IdxKNNRank                     = 3
	IdxRRFScore                    = 4
	IdxPopularityPrior             = 5
	IdxPriceLogCents               = 6
	IdxTitleLengthTokens           = 7
	IdxQueryLengthTokens           = 8
	IdxQueryHasBrand               = 9
	IdxQueryHasColor               = 10
	IdxQueryHasCategoryToken       = 11
	IdxQueryHasSizePat             = 12
	IdxQueryGenderIntent           = 13
	IdxProductGender               = 14
	IdxGenderIntentMatch           = 15
	IdxGenderIntentMis             = 16
	IdxProductBrandMatch           = 17
	IdxProductBrandTokenOverlap    = 18
	IdxProductColorMatch           = 19
	IdxTitleQueryTokenCoverage     = 20
	IdxCategoryQueryTokenCoverage  = 21
	IdxProductCategoryTokenOverlap = 22
	IdxTitleExactQueryMatch        = 23
	IdxUserBrandAffinity           = 24
	IdxQueryAffordabilityIntent    = 25
	IdxAffordabilityPriceScore     = 26
	IdxBrandFamilyMatch            = 27
	IdxSubbrandTitleMatch          = 28

	NumFeatures = 29
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
