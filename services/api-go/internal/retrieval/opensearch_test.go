package retrieval

import (
	"encoding/json"
	"testing"
)

func TestBuildBM25BodyFiltersMensQueriesToMenCategory(t *testing.T) {
	body, err := buildBM25Body("mens nike running shoes", 50)
	if err != nil {
		t.Fatalf("buildBM25Body: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	query := got["query"].(map[string]any)
	boolQuery := query["bool"].(map[string]any)
	filter := boolQuery["filter"].([]any)
	if len(filter) != 1 {
		t.Fatalf("filter len=%d, want 1", len(filter))
	}
	term := filter[0].(map[string]any)["term"].(map[string]any)
	if term["category_path.raw"] != "Men" {
		t.Fatalf("category_path.raw filter=%#v, want Men", term["category_path.raw"])
	}
}

func TestBuildBM25BodyLeavesUngenderedQueriesUnfiltered(t *testing.T) {
	body, err := buildBM25Body("nike running shoes", 50)
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
}

func TestBuildKNNBodyFiltersWomensQueriesToWomenCategory(t *testing.T) {
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
	filter := titleVec["filter"].(map[string]any)
	term := filter["term"].(map[string]any)
	if term["category_path.raw"] != "Women" {
		t.Fatalf("category_path.raw filter=%#v, want Women", term["category_path.raw"])
	}
}
