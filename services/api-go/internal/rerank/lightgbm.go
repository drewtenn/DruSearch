// Package rerank wraps a LightGBM model loaded via dmitryikh/leaves
// and applies it to a list of retrieval candidates. The model file
// path is provided by promote.py and reload-able via the admin API.
package rerank

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync/atomic"
	"time"

	"github.com/dmitryikh/leaves"

	"github.com/drewtenn/drusearch/services/api-go/internal/features"
	"github.com/drewtenn/drusearch/services/api-go/internal/obs"
	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
)

type Loaded struct {
	Ensemble *leaves.Ensemble
	Path     string
	Meta     map[string]any
	LoadedAt time.Time
}

// Reranker holds the currently-loaded ensemble. Callers Read with Get
// (cheap atomic load); a successful Reload replaces it atomically. A nil
// Get result means "no model loaded" — callers should fall back to RRF.
type Reranker struct {
	current atomic.Pointer[Loaded]
	dir     string
	name    string
}

func New(dir, name string) *Reranker {
	return &Reranker{dir: dir, name: name}
}

// modelPath / metaPath return the conventional artifact paths set by
// pipelines.register.promote.
func (r *Reranker) modelPath() string {
	return filepath.Join(r.dir, r.name+".txt")
}
func (r *Reranker) metaPath() string {
	return filepath.Join(r.dir, r.name+".json")
}

// Reload reads the artifact from disk and atomically swaps the pointer.
// If no model file is present, returns os.ErrNotExist.
func (r *Reranker) Reload() (*Loaded, error) {
	path := r.modelPath()
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	ens, err := leaves.LGEnsembleFromFile(path, false)
	if err != nil {
		return nil, fmt.Errorf("leaves load %s: %w", path, err)
	}
	meta := map[string]any{}
	if b, err := os.ReadFile(r.metaPath()); err == nil {
		_ = json.Unmarshal(b, &meta)
	}
	loaded := &Loaded{
		Ensemble: ens,
		Path:     path,
		Meta:     meta,
		LoadedAt: time.Now().UTC(),
	}
	// Reset the model-loaded gauge before re-arming so /metrics doesn't
	// hold a stale (name, version) label after a model swap.
	obs.ModelLoaded.Reset()
	if name, ok := loaded.Meta["name"].(string); ok {
		version := ""
		switch v := loaded.Meta["version"].(type) {
		case string:
			version = v
		case float64:
			version = fmt.Sprintf("%v", v)
		}
		obs.ModelLoaded.WithLabelValues(name, version).Set(1)
	}
	r.current.Store(loaded)
	return loaded, nil
}

func (r *Reranker) Get() *Loaded {
	return r.current.Load()
}

// Score runs the loaded model on a per-row dense feature matrix and returns
// one prediction per row. Caller is responsible for sorting by score.
func (l *Loaded) Score(matrix []float64, nrows int) ([]float64, error) {
	if l == nil || l.Ensemble == nil {
		return nil, fmt.Errorf("no model loaded")
	}
	preds := make([]float64, nrows)
	if err := l.Ensemble.PredictDense(
		matrix, nrows, features.NumFeatures, preds,
		0, // 0 = use all trees
		1, // single-threaded; reranker hot-path
	); err != nil {
		return nil, err
	}
	return preds, nil
}

// Apply scores `hits` and returns a new slice sorted by descending LTR
// score. The original slice is not modified. If l is nil this is a no-op
// (returns hits unchanged).
//
// The LTR score is also written into Hit.LTR / Hit.LTRRank for downstream
// `explain` reporting.
type ScoredHit struct {
	retrieval.Hit
	LTR     float64
	LTRRank int
	BGE     float64
	BGERank int
}

func Apply(l *Loaded, query string, hits []retrieval.Hit, vocab *features.Vocab, user *features.UserFeatures) ([]ScoredHit, error) {
	out := make([]ScoredHit, len(hits))
	for i, h := range hits {
		out[i] = ScoredHit{Hit: h}
	}
	if l == nil || len(hits) == 0 {
		return out, nil
	}
	matrix := features.BuildMatrix(query, hits, vocab, user)
	preds, err := l.Score(matrix, len(hits))
	if err != nil {
		return out, err
	}
	for i := range out {
		out[i].LTR = preds[i]
	}
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].LTR > out[j].LTR
	})
	for i := range out {
		out[i].LTRRank = i + 1
	}
	return out, nil
}
