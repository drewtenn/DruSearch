package retrieval

import (
	"encoding/json"
	"testing"
)

func TestBuildBM25BodyLeavesQueriesUnfiltered(t *testing.T) {
	for _, queryText := range []string{
		"mens nike running shoes",
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
