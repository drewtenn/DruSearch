package rerank

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/drewtenn/drusearch/services/api-go/internal/features"
)

const tinyXGBoostModel = `{
  "learner": {
    "attributes": {},
    "feature_names": [],
    "feature_types": [],
    "learner_model_param": {
      "base_score": "0E0",
      "num_class": "0",
      "num_feature": "27",
      "num_target": "1"
    },
    "objective": {"name": "rank:ndcg", "reg_loss_param": {"scale_pos_weight": "1"}},
    "gradient_booster": {
      "name": "gbtree",
      "model": {
        "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": "1"},
        "iteration_indptr": [0, 1],
        "tree_info": [0],
        "trees": [
          {
            "base_weights": [0, -1, 1],
            "categories": [],
            "categories_nodes": [],
            "categories_segments": [],
            "categories_sizes": [],
            "default_left": [0, 0, 0],
            "id": 0,
            "left_children": [1, -1, -1],
            "loss_changes": [1, 0, 0],
            "parents": [2147483647, 0, 0],
            "right_children": [2, -1, -1],
            "split_conditions": [2, -1, 1],
            "split_indices": [0, 0, 0],
            "split_type": [0, 0, 0],
            "sum_hessian": [1, 1, 1],
            "tree_param": {"num_deleted": "0", "num_feature": "27", "num_nodes": "3", "size_leaf_vector": "1"}
          }
        ]
      }
    }
  },
  "version": [2, 1, 3]
}`

func TestXGBoostScoreEvaluatesNativeJSONTrees(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.xgb.json")
	if err := os.WriteFile(path, []byte(tinyXGBoostModel), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	model, err := LoadXGBoost(path)
	if err != nil {
		t.Fatalf("LoadXGBoost: %v", err)
	}

	matrix := make([]float64, 2*features.NumFeatures)
	matrix[0] = 0.5
	matrix[features.NumFeatures] = 2.5
	got, err := model.Score(matrix, 2)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	want := []float64{-1, 1}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("row %d: got %g want %g", i, got[i], want[i])
		}
	}
}

func TestRerankerReloadLoadsXGBoostFromMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "ltr_reranker.xgb.json"), []byte(tinyXGBoostModel), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	meta := `{"name":"ltr_reranker","version":"9","model_backend":"xgboost"}`
	if err := os.WriteFile(filepath.Join(dir, "ltr_reranker.json"), []byte(meta), 0o644); err != nil {
		t.Fatalf("write meta: %v", err)
	}

	loaded, err := New(dir, "ltr_reranker").Reload()
	if err != nil {
		t.Fatalf("Reload: %v", err)
	}
	if loaded.Path != filepath.Join(dir, "ltr_reranker.xgb.json") {
		t.Fatalf("loaded path = %s", loaded.Path)
	}
	if loaded.Meta["model_backend"] != "xgboost" {
		t.Fatalf("model_backend = %v", loaded.Meta["model_backend"])
	}
}
