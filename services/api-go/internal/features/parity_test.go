package features

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

// Locating the repo root from the test file lets the test run with `go
// test ./...` from anywhere — no env var needed.
func fixturesPath(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("could not resolve caller path")
	}
	// services/api-go/internal/features/parity_test.go -> repo root is 4 levels up
	repo := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", "..", ".."))
	return filepath.Join(repo, "libs", "schema", "fixtures", "interaction_fixtures.json")
}

type fixtureFile struct {
	Vocab struct {
		Brands []string `json:"brands"`
		Colors []string `json:"colors"`
	} `json:"vocab"`
	Cases []fixtureCase `json:"cases"`
}

type fixtureCase struct {
	Name         string         `json:"name"`
	Query        string         `json:"query"`
	CategoryPath []string       `json:"category_path"`
	Expected     fixtureExpects `json:"expected"`
}

type fixtureExpects struct {
	Tokenize             []string `json:"tokenize"`
	QueryLengthTokens    float64  `json:"query_length_tokens"`
	QueryHasBrand        float64  `json:"query_has_brand"`
	QueryHasColor        float64  `json:"query_has_color"`
	QueryHasSizePattern  float64  `json:"query_has_size_pattern"`
	QueryGenderIntent    float64  `json:"query_gender_intent"`
	ProductGender        float64  `json:"product_gender"`
	GenderIntentMatch    float64  `json:"gender_intent_match"`
	GenderIntentMismatch float64  `json:"gender_intent_mismatch"`
}

// TestInteractionParityFixtures runs the Go reference transforms against the
// canonical fixtures shared with the Python reference impl. Both sides must
// produce identical output on the same fixtures, otherwise the trained model
// is consuming a different feature distribution than the serving path.
func TestInteractionParityFixtures(t *testing.T) {
	raw, err := os.ReadFile(fixturesPath(t))
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var ff fixtureFile
	if err := json.Unmarshal(raw, &ff); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}
	if len(ff.Cases) == 0 {
		t.Fatal("no fixtures found")
	}
	brands := setLower(ff.Vocab.Brands)
	colors := setLower(ff.Vocab.Colors)

	for _, c := range ff.Cases {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			got := Tokenize(c.Query)
			want := c.Expected.Tokenize
			if !slicesEqual(got, want) {
				t.Errorf("Tokenize(%q) = %#v, want %#v", c.Query, got, want)
			}
			if v := QueryLengthTokens(c.Query); v != c.Expected.QueryLengthTokens {
				t.Errorf("QueryLengthTokens(%q) = %v, want %v", c.Query, v, c.Expected.QueryLengthTokens)
			}
			if v := QueryHasAny(c.Query, brands); v != c.Expected.QueryHasBrand {
				t.Errorf("QueryHasAny[brand](%q) = %v, want %v", c.Query, v, c.Expected.QueryHasBrand)
			}
			if v := QueryHasAny(c.Query, colors); v != c.Expected.QueryHasColor {
				t.Errorf("QueryHasAny[color](%q) = %v, want %v", c.Query, v, c.Expected.QueryHasColor)
			}
			if v := QueryHasSizePattern(c.Query); v != c.Expected.QueryHasSizePattern {
				t.Errorf("QueryHasSizePattern(%q) = %v, want %v", c.Query, v, c.Expected.QueryHasSizePattern)
			}
			qg := QueryGenderIntent(c.Query)
			if qg != c.Expected.QueryGenderIntent {
				t.Errorf("QueryGenderIntent(%q) = %v, want %v", c.Query, qg, c.Expected.QueryGenderIntent)
			}
			pg := ProductGender(c.CategoryPath)
			if pg != c.Expected.ProductGender {
				t.Errorf("ProductGender(%#v) = %v, want %v", c.CategoryPath, pg, c.Expected.ProductGender)
			}
			if v := GenderIntentMatch(qg, pg); v != c.Expected.GenderIntentMatch {
				t.Errorf("GenderIntentMatch(%v, %v) = %v, want %v", qg, pg, v, c.Expected.GenderIntentMatch)
			}
			if v := GenderIntentMismatch(qg, pg); v != c.Expected.GenderIntentMismatch {
				t.Errorf("GenderIntentMismatch(%v, %v) = %v, want %v", qg, pg, v, c.Expected.GenderIntentMismatch)
			}
		})
	}
}

func setLower(xs []string) map[string]struct{} {
	m := make(map[string]struct{}, len(xs))
	for _, x := range xs {
		m[x] = struct{}{}
	}
	return m
}

// Compare slices treating nil and len-0 as equal so the JSON [] case (Go nil
// from Tokenize("")) parses cleanly.
func slicesEqual(a, b []string) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}
