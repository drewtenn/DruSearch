package neuralrerank

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
)

func TestClientRerankPostsQueryAndDocuments(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rerank" {
			t.Fatalf("path = %q, want /rerank", r.URL.Path)
		}
		var req rerankRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Query != "nike shoes" {
			t.Fatalf("query = %q, want nike shoes", req.Query)
		}
		if len(req.Documents) != 2 {
			t.Fatalf("documents len = %d, want 2", len(req.Documents))
		}
		if req.Documents[0].ID != "p1" || req.Documents[0].Text != "doc one" {
			t.Fatalf("first doc = %#v", req.Documents[0])
		}

		_ = json.NewEncoder(w).Encode(rerankResponse{
			Model: "BAAI/bge-reranker-v2-m3",
			Scores: []Score{
				{ID: "p1", Score: 0.2},
				{ID: "p2", Score: 0.9},
			},
		})
	}))
	defer srv.Close()

	client := New(srv.URL, time.Second)
	got, model, err := client.Rerank(t.Context(), "nike shoes", []Document{
		{ID: "p1", Text: "doc one"},
		{ID: "p2", Text: "doc two"},
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if model != "BAAI/bge-reranker-v2-m3" {
		t.Fatalf("model = %q", model)
	}
	if len(got) != 2 || got[0].ID != "p1" || got[0].Score != 0.2 || got[1].ID != "p2" || got[1].Score != 0.9 {
		t.Fatalf("scores = %#v", got)
	}
}

func TestProductTextIncludesStructuredFields(t *testing.T) {
	got := ProductText(retrieval.Hit{
		Title:        "Nike Baby Booties",
		Brand:        "Nike",
		CategoryPath: []string{"Baby Boys", "Shoes", "Boots"},
		PriceCents:   2492,
	})

	for _, want := range []string{
		"Title: Nike Baby Booties",
		"Brand: Nike",
		"Category: Baby Boys > Shoes > Boots",
		"Price: $24.92",
	} {
		if !contains(got, want) {
			t.Fatalf("ProductText missing %q in %q", want, got)
		}
	}
}

func contains(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && index(s, sub) >= 0)
}

func index(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
