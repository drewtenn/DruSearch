package features

import "testing"

// Drift guard: hand-written schema (schema.go) must match generated schema
// (schema_generated.go, built from libs/schema/feature_schema.json by
// libs/schema/codegen.py). If this fails, run `make check-feature-parity`
// and reconcile.
func TestSchemaMatchesGenerated(t *testing.T) {
	if NumFeatures != GenNumFeatures {
		t.Fatalf("NumFeatures=%d, GenNumFeatures=%d", NumFeatures, GenNumFeatures)
	}
	if len(Names) != len(GenNames) {
		t.Fatalf("len(Names)=%d, len(GenNames)=%d", len(Names), len(GenNames))
	}
	for i := range Names {
		if Names[i] != GenNames[i] {
			t.Errorf("Names[%d]=%q, GenNames[%d]=%q", i, Names[i], i, GenNames[i])
		}
	}
	indexExpect := map[int]string{
		IdxBM25Score:                   "bm25_score",
		IdxBM25Rank:                    "bm25_rank",
		IdxKNNScore:                    "knn_score",
		IdxKNNRank:                     "knn_rank",
		IdxRRFScore:                    "rrf_score",
		IdxPopularityPrior:             "popularity_prior",
		IdxPriceLogCents:               "price_log_cents",
		IdxTitleLengthTokens:           "title_length_tokens",
		IdxQueryLengthTokens:           "query_length_tokens",
		IdxQueryHasBrand:               "query_has_brand",
		IdxQueryHasColor:               "query_has_color",
		IdxQueryHasCategoryToken:       "query_has_category_token",
		IdxQueryHasSizePat:             "query_has_size_pattern",
		IdxQueryGenderIntent:           "query_gender_intent",
		IdxProductGender:               "product_gender",
		IdxGenderIntentMatch:           "gender_intent_match",
		IdxGenderIntentMis:             "gender_intent_mismatch",
		IdxProductBrandMatch:           "product_brand_match",
		IdxProductBrandTokenOverlap:    "product_brand_token_overlap",
		IdxProductColorMatch:           "product_color_match",
		IdxTitleQueryTokenCoverage:     "title_query_token_coverage",
		IdxCategoryQueryTokenCoverage:  "category_query_token_coverage",
		IdxProductCategoryTokenOverlap: "product_category_token_overlap",
		IdxTitleExactQueryMatch:        "title_exact_query_match",
		IdxUserBrandAffinity:           "user_brand_affinity",
		IdxQueryAffordabilityIntent:    "query_affordability_intent",
		IdxAffordabilityPriceScore:     "affordability_price_score",
	}
	for idx, name := range indexExpect {
		if Names[idx] != name {
			t.Errorf("Names[%d]=%q, want %q", idx, Names[idx], name)
		}
	}
}
