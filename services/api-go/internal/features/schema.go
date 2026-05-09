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
	IdxTitleBM25Score              = 2
	IdxCategoryPathBM25Score       = 3
	IdxCategoryBM25Score           = 4
	IdxBulletsBM25Score            = 5
	IdxDescriptionBM25Score        = 6
	IdxBrandBM25Score              = 7
	IdxKNNScore                    = 8
	IdxKNNRank                     = 9
	IdxRRFScore                    = 10
	IdxPopularityPrior             = 11
	IdxPriceLogCents               = 12
	IdxTitleLengthTokens           = 13
	IdxQueryLengthTokens           = 14
	IdxQueryHasBrand               = 15
	IdxQueryHasColor               = 16
	IdxQueryHasCategoryToken       = 17
	IdxQueryHasSizePat             = 18
	IdxQueryGenderIntent           = 19
	IdxProductGender               = 20
	IdxGenderIntentMatch           = 21
	IdxGenderIntentMis             = 22
	IdxProductBrandMatch           = 23
	IdxProductBrandTokenOverlap    = 24
	IdxProductColorMatch           = 25
	IdxTitleQueryTokenCoverage     = 26
	IdxCategoryQueryTokenCoverage  = 27
	IdxProductCategoryTokenOverlap = 28
	IdxTitleExactQueryMatch        = 29
	IdxUserBrandAffinity           = 30
	IdxQueryAffordabilityIntent    = 31
	IdxAffordabilityPriceScore     = 32
	IdxBrandFamilyMatch            = 33
	IdxSubbrandTitleMatch          = 34

	NumFeatures = 35
)

// Names ordered by their feature index above.
var Names = [...]string{
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
