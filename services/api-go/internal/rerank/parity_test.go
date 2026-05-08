package rerank

import (
	"math"
	"os"
	"testing"

	"github.com/drewtenn/drusearch/services/api-go/internal/features"
)

// Cross-language parity check.
//
// Inputs are five hand-picked feature rows (matching the Python feature index
// in pipelines/pipelines/features/__init__.py); expected predictions were
// generated with python lightgbm against the same on-disk model file
// (/var/lib/drusearch/models/ltr_reranker.txt) — see promote.py for how the
// file is rewritten. Any divergence above 1e-9 indicates a serialization or
// feature-ordering bug.
func TestPythonGoPredictionParity(t *testing.T) {
	const modelPath = "/var/lib/drusearch/models/ltr_reranker.txt"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("no model at %s; run `make promote-model` first", modelPath)
	}

	rr := New("/var/lib/drusearch/models", "ltr_reranker")
	loaded, err := rr.Reload()
	if err != nil {
		t.Fatalf("Reload: %v", err)
	}

	// (rows × NumFeatures) row-major, ordering must match features.Names.
	rows := [][features.NumFeatures]float64{
		{12.5, 1, 0.7, 1, 0.030, 0.0, 9.0, 12, 3, 0, 0, 0},
		{10.0, 5, 0.5, 8, 0.022, 0.0, 8.5, 10, 3, 1, 0, 0},
		{8.0, 9, 0.4, 3, 0.020, 0.0, 7.0, 8, 2, 0, 1, 1},
		{5.0, 40, 0.3, 99, 0.012, 0.0, 6.5, 6, 4, 0, 0, 0},
		{15.0, 2, 0.0, 0, 0.018, 0.0, 9.5, 15, 5, 1, 1, 0},
	}
	wantPreds := []float64{
		-3.523572325969941,
		-4.346344157064133,
		-5.676620555580454,
		-5.345505535219655,
		-4.406090248137971,
	}

	mat := make([]float64, 0, len(rows)*features.NumFeatures)
	for _, r := range rows {
		mat = append(mat, r[:]...)
	}

	got, err := loaded.Score(mat, len(rows))
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if len(got) != len(wantPreds) {
		t.Fatalf("got %d preds, want %d", len(got), len(wantPreds))
	}

	const tol = 1e-9
	for i, w := range wantPreds {
		if d := math.Abs(got[i] - w); d > tol {
			t.Errorf("row %d: got %.15f want %.15f diff %g (tol %g)", i, got[i], w, d, tol)
		}
	}
}
