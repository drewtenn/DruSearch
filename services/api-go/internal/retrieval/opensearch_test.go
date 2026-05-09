package retrieval

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestBuildBM25BodyLeavesQueriesUnfiltered(t *testing.T) {
	for _, queryText := range []string{
		"nike running shoes",
	} {
		t.Run(queryText, func(t *testing.T) {
			body, err := buildBM25Body(queryText, 50)
			if err != nil {
				t.Fatalf("buildBM25Body: %v", err)
			}

			var got map[string]any
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatalf("unmarshal body: %v", err)
			}

			query := got["query"].(map[string]any)
			if _, ok := query["multi_match"]; !ok {
				t.Fatalf("query=%#v, want plain multi_match", query)
			}
			if _, ok := query["bool"]; ok {
				t.Fatalf("query=%#v, want no hard filter", query)
			}
		})
	}
}

func TestBuildBM25BodyBoostsDerivedGenderIntent(t *testing.T) {
	body, err := buildBM25Body("nike mens shoes", 50)
	if err != nil {
		t.Fatalf("buildBM25Body: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	query := got["query"].(map[string]any)
	boosting := query["boosting"].(map[string]any)
	positive := boosting["positive"].(map[string]any)
	boolQuery := positive["bool"].(map[string]any)
	should := boolQuery["should"].([]any)

	if len(should) != 2 {
		t.Fatalf("should=%#v, want men and unisex derived gender boosts", should)
	}
	if boosting["negative_boost"] != 0.25 {
		t.Fatalf("negative_boost=%#v, want 0.25", boosting["negative_boost"])
	}

	negative := boosting["negative"].(map[string]any)
	terms := negative["terms"].(map[string]any)
	wantOpposite := []any{"women", "boys", "girls"}
	if gotOpposite := terms["derived_gender"]; !reflect.DeepEqual(gotOpposite, wantOpposite) {
		t.Fatalf("negative derived_gender=%#v, want %#v", gotOpposite, wantOpposite)
	}
}

func TestBuildKNNBodyLeavesGenderedQueriesUnfiltered(t *testing.T) {
	body, err := buildKNNBody("women's nike running shoes", []float32{0.1, 0.2}, 50)
	if err != nil {
		t.Fatalf("buildKNNBody: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	query := got["query"].(map[string]any)
	knn := query["knn"].(map[string]any)
	titleVec := knn["title_vec"].(map[string]any)
	if _, ok := titleVec["filter"]; ok {
		t.Fatalf("title_vec=%#v, want no hard filter", titleVec)
	}
}
