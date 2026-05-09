// Package rerank wraps a promoted LTR model and applies it to retrieval
// candidates. LightGBM is loaded via dmitryikh/leaves; XGBoost is loaded
// from the native JSON artifact written by Booster.save_model.
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
	Scorer   Scorer
	Path     string
	Meta     map[string]any
	LoadedAt time.Time
}

type Scorer interface {
	Score(matrix []float64, nrows int) ([]float64, error)
}

type lightGBMScorer struct {
	ensemble *leaves.Ensemble
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
func (r *Reranker) xgboostModelPath() string {
	return filepath.Join(r.dir, r.name+".xgb.json")
}
func (r *Reranker) metaPath() string {
	return filepath.Join(r.dir, r.name+".json")
}

// Reload reads the artifact from disk and atomically swaps the pointer.
// If no model file is present, returns os.ErrNotExist.
func (r *Reranker) Reload() (*Loaded, error) {
	meta := map[string]any{}
	if b, err := os.ReadFile(r.metaPath()); err == nil {
		_ = json.Unmarshal(b, &meta)
	}
	backend := modelBackend(meta)
	var (
		path   string
		scorer Scorer
		ens    *leaves.Ensemble
		err    error
	)
	switch backend {
	case "lgbm":
		path = r.modelPath()
		if _, err := os.Stat(path); err != nil {
			return nil, err
		}
		if err := validateFeatureSchema(meta); err != nil {
			return nil, err
		}
		ens, err = leaves.LGEnsembleFromFile(path, false)
		if err != nil {
			return nil, fmt.Errorf("leaves load %s: %w", path, err)
		}
		scorer = lightGBMScorer{ensemble: ens}
	case "xgboost":
		path = r.xgboostModelPath()
		if _, err := os.Stat(path); err != nil {
			return nil, err
		}
		if err := validateFeatureSchema(meta); err != nil {
			return nil, err
		}
		scorer, err = LoadXGBoost(path)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unsupported model_backend %q", backend)
	}
	loaded := &Loaded{
		Ensemble: ens,
		Scorer:   scorer,
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

func validateFeatureSchema(meta map[string]any) error {
	version, _ := meta["feature_schema_version"].(string)
	if version == "" {
		return fmt.Errorf("model metadata missing feature_schema_version; serving schema is %s", features.SchemaVersion)
	}
	if version != features.SchemaVersion {
		return fmt.Errorf("model feature_schema_version %s != serving schema %s", version, features.SchemaVersion)
	}
	return nil
}

func (r *Reranker) Get() *Loaded {
	return r.current.Load()
}

func modelBackend(meta map[string]any) string {
	if v, ok := meta["model_backend"].(string); ok && v != "" {
		switch v {
		case "lgbm", "lightgbm":
			return "lgbm"
		case "xgboost", "xgb":
			return "xgboost"
		default:
			return v
		}
	}
	return "lgbm"
}

// Score runs the loaded model on a per-row dense feature matrix and returns
// one prediction per row. Caller is responsible for sorting by score.
func (l *Loaded) Score(matrix []float64, nrows int) ([]float64, error) {
	if l == nil || l.Scorer == nil {
		return nil, fmt.Errorf("no model loaded")
	}
	return l.Scorer.Score(matrix, nrows)
}

func (s lightGBMScorer) Score(matrix []float64, nrows int) ([]float64, error) {
	if s.ensemble == nil {
		return nil, fmt.Errorf("no model loaded")
	}
	preds := make([]float64, nrows)
	if err := s.ensemble.PredictDense(
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
