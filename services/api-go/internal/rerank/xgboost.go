package rerank

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/drewtenn/drusearch/services/api-go/internal/features"
)

type XGBoost struct {
	BaseScore  float64
	NumFeature int
	Trees      []xgbTree
	TreeLimit  int
}

type xgbTree struct {
	LeftChildren    []int     `json:"left_children"`
	RightChildren   []int     `json:"right_children"`
	DefaultLeft     []int     `json:"default_left"`
	SplitConditions []float64 `json:"split_conditions"`
	SplitIndices    []int     `json:"split_indices"`
	SplitType       []int     `json:"split_type"`
}

type xgbModelJSON struct {
	Learner struct {
		Attributes        map[string]string `json:"attributes"`
		LearnerModelParam struct {
			BaseScore  string `json:"base_score"`
			NumFeature string `json:"num_feature"`
			NumTarget  string `json:"num_target"`
		} `json:"learner_model_param"`
		GradientBooster struct {
			Name  string `json:"name"`
			Model struct {
				IterationIndptr []int     `json:"iteration_indptr"`
				Trees           []xgbTree `json:"trees"`
			} `json:"model"`
		} `json:"gradient_booster"`
	} `json:"learner"`
}

func LoadXGBoost(path string) (*XGBoost, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw xgbModelJSON
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("xgboost parse %s: %w", path, err)
	}
	if raw.Learner.GradientBooster.Name != "gbtree" {
		return nil, fmt.Errorf("xgboost %s: unsupported booster %q", path, raw.Learner.GradientBooster.Name)
	}
	numFeature, err := strconv.Atoi(raw.Learner.LearnerModelParam.NumFeature)
	if err != nil {
		return nil, fmt.Errorf("xgboost %s: invalid num_feature %q", path, raw.Learner.LearnerModelParam.NumFeature)
	}
	if numFeature > features.NumFeatures {
		return nil, fmt.Errorf("xgboost %s: model has %d features, serving schema has %d", path, numFeature, features.NumFeatures)
	}
	if nt := raw.Learner.LearnerModelParam.NumTarget; nt != "" && nt != "1" {
		return nil, fmt.Errorf("xgboost %s: unsupported num_target %q", path, nt)
	}
	baseScore, err := parseXGBFloat(raw.Learner.LearnerModelParam.BaseScore)
	if err != nil {
		return nil, fmt.Errorf("xgboost %s: invalid base_score %q", path, raw.Learner.LearnerModelParam.BaseScore)
	}
	for i := range raw.Learner.GradientBooster.Model.Trees {
		if err := validateXGBTree(raw.Learner.GradientBooster.Model.Trees[i]); err != nil {
			return nil, fmt.Errorf("xgboost %s tree %d: %w", path, i, err)
		}
	}
	treeLimit := len(raw.Learner.GradientBooster.Model.Trees)
	if best, ok := raw.Learner.Attributes["best_iteration"]; ok {
		if bestIter, err := strconv.Atoi(best); err == nil {
			indptr := raw.Learner.GradientBooster.Model.IterationIndptr
			if bestIter+1 >= 0 && bestIter+1 < len(indptr) && indptr[bestIter+1] <= treeLimit {
				treeLimit = indptr[bestIter+1]
			}
		}
	}
	return &XGBoost{
		BaseScore:  baseScore,
		NumFeature: numFeature,
		Trees:      raw.Learner.GradientBooster.Model.Trees,
		TreeLimit:  treeLimit,
	}, nil
}

func parseXGBFloat(s string) (float64, error) {
	if s == "" {
		return 0, nil
	}
	return strconv.ParseFloat(s, 64)
}

func validateXGBTree(t xgbTree) error {
	n := len(t.LeftChildren)
	if n == 0 {
		return fmt.Errorf("empty tree")
	}
	if len(t.RightChildren) != n || len(t.DefaultLeft) != n ||
		len(t.SplitConditions) != n || len(t.SplitIndices) != n || len(t.SplitType) != n {
		return fmt.Errorf("tree arrays have inconsistent lengths")
	}
	for _, typ := range t.SplitType {
		if typ != 0 {
			return fmt.Errorf("categorical splits are not supported")
		}
	}
	for i := 0; i < n; i++ {
		if t.SplitIndices[i] < 0 || t.SplitIndices[i] >= features.NumFeatures {
			return fmt.Errorf("split index %d out of serving schema range", t.SplitIndices[i])
		}
		for _, child := range []int{t.LeftChildren[i], t.RightChildren[i]} {
			if child >= n {
				return fmt.Errorf("child index %d out of tree range", child)
			}
		}
	}
	return nil
}

func (m *XGBoost) Score(matrix []float64, nrows int) ([]float64, error) {
	if m == nil {
		return nil, fmt.Errorf("no model loaded")
	}
	if len(matrix) < nrows*features.NumFeatures {
		return nil, fmt.Errorf("matrix has %d values, need at least %d", len(matrix), nrows*features.NumFeatures)
	}
	preds := make([]float64, nrows)
	for row := 0; row < nrows; row++ {
		score := m.BaseScore
		offset := row * features.NumFeatures
		for i := 0; i < m.TreeLimit; i++ {
			score += m.Trees[i].predict(matrix[offset : offset+features.NumFeatures])
		}
		preds[row] = score
	}
	return preds, nil
}

func (t xgbTree) predict(row []float64) float64 {
	node := 0
	for {
		left := t.LeftChildren[node]
		right := t.RightChildren[node]
		if left < 0 && right < 0 {
			return t.SplitConditions[node]
		}
		featureIndex := t.SplitIndices[node]
		value := math.NaN()
		if featureIndex >= 0 && featureIndex < len(row) {
			value = row[featureIndex]
		}
		if math.IsNaN(value) {
			if t.DefaultLeft[node] != 0 {
				node = left
			} else {
				node = right
			}
			continue
		}
		if value < t.SplitConditions[node] {
			node = left
		} else {
			node = right
		}
	}
}
