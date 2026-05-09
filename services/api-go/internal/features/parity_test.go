package features

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
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
	ProductBrand string         `json:"product_brand"`
	ProductColor string         `json:"product_color"`
	ProductTitle string         `json:"product_title"`
	CategoryPath []string       `json:"category_path"`
	Expected     fixtureExpects `json:"expected"`
}

type fixtureExpects struct {
	Tokenize               []string `json:"tokenize"`
	QueryLengthTokens      float64  `json:"query_length_tokens"`
	QueryHasBrand          float64  `json:"query_has_brand"`
	QueryHasColor          float64  `json:"query_has_color"`
	QueryHasCategory       float64  `json:"query_has_category_token"`
	QueryHasSizePattern    float64  `json:"query_has_size_pattern"`
	QueryAffordability     float64  `json:"query_affordability_intent"`
	QueryGenderIntent      float64  `json:"query_gender_intent"`
	ProductGender          float64  `json:"product_gender"`
	GenderIntentMatch      float64  `json:"gender_intent_match"`
	GenderIntentMismatch   float64  `json:"gender_intent_mismatch"`
	ProductBrandMatch      float64  `json:"product_brand_match"`
	ProductBrandOverlap    float64  `json:"product_brand_token_overlap"`
	ProductColorMatch      float64  `json:"product_color_match"`
	TitleQueryCoverage     float64  `json:"title_query_token_coverage"`
	CategoryQueryCoverage  float64  `json:"category_query_token_coverage"`
	ProductCategoryOverlap float64  `json:"product_category_token_overlap"`
	TitleExactQueryMatch   float64  `json:"title_exact_query_match"`
	BrandFamilyMatch       *float64 `json:"brand_family_match"`
	SubbrandTitleMatch     *float64 `json:"subbrand_title_match"`
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
	categories := categoryTokens(ff.Cases)

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
			if v := QueryHasAny(c.Query, categories); v != c.Expected.QueryHasCategory {
				t.Errorf("QueryHasAny[category](%q) = %v, want %v", c.Query, v, c.Expected.QueryHasCategory)
			}
			if v := QueryHasSizePattern(c.Query); v != c.Expected.QueryHasSizePattern {
				t.Errorf("QueryHasSizePattern(%q) = %v, want %v", c.Query, v, c.Expected.QueryHasSizePattern)
			}
			if v := QueryAffordabilityIntent(c.Query); v != c.Expected.QueryAffordability {
				t.Errorf("QueryAffordabilityIntent(%q) = %v, want %v", c.Query, v, c.Expected.QueryAffordability)
			}
			qg := QueryGenderIntent(c.Query)
			if qg != c.Expected.QueryGenderIntent {
				t.Errorf("QueryGenderIntent(%q) = %v, want %v", c.Query, qg, c.Expected.QueryGenderIntent)
			}
			pg := ProductGender(c.CategoryPath, c.ProductTitle)
			if pg != c.Expected.ProductGender {
				t.Errorf("ProductGender(%#v, %q) = %v, want %v", c.CategoryPath, c.ProductTitle, pg, c.Expected.ProductGender)
			}
			if v := GenderIntentMatch(qg, pg); v != c.Expected.GenderIntentMatch {
				t.Errorf("GenderIntentMatch(%v, %v) = %v, want %v", qg, pg, v, c.Expected.GenderIntentMatch)
			}
			if v := GenderIntentMismatch(qg, pg); v != c.Expected.GenderIntentMismatch {
				t.Errorf("GenderIntentMismatch(%v, %v) = %v, want %v", qg, pg, v, c.Expected.GenderIntentMismatch)
			}
			if v := ProductBrandMatch(c.Query, c.ProductBrand); v != c.Expected.ProductBrandMatch {
				t.Errorf("ProductBrandMatch(%q, %q) = %v, want %v", c.Query, c.ProductBrand, v, c.Expected.ProductBrandMatch)
			}
			if v := ProductBrandTokenOverlap(c.Query, c.ProductBrand); v != c.Expected.ProductBrandOverlap {
				t.Errorf("ProductBrandTokenOverlap(%q, %q) = %v, want %v", c.Query, c.ProductBrand, v, c.Expected.ProductBrandOverlap)
			}
			if v := ProductColorMatch(c.Query, c.ProductColor); v != c.Expected.ProductColorMatch {
				t.Errorf("ProductColorMatch(%q, %q) = %v, want %v", c.Query, c.ProductColor, v, c.Expected.ProductColorMatch)
			}
			categoryText := strings.Join(c.CategoryPath, " ")
			if v := QueryTokenCoverage(c.Query, c.ProductTitle, brands); v != c.Expected.TitleQueryCoverage {
				t.Errorf("QueryTokenCoverage[title](%q, %q) = %v, want %v", c.Query, c.ProductTitle, v, c.Expected.TitleQueryCoverage)
			}
			if v := QueryTokenCoverage(c.Query, categoryText, brands); v != c.Expected.CategoryQueryCoverage {
				t.Errorf("QueryTokenCoverage[category](%q, %q) = %v, want %v", c.Query, categoryText, v, c.Expected.CategoryQueryCoverage)
			}
			if v := TokenOverlapFraction(c.Query, categoryText); v != c.Expected.ProductCategoryOverlap {
				t.Errorf("TokenOverlapFraction(%q, %q) = %v, want %v", c.Query, categoryText, v, c.Expected.ProductCategoryOverlap)
			}
			if v := ExactQueryPhraseMatch(c.Query, c.ProductTitle); v != c.Expected.TitleExactQueryMatch {
				t.Errorf("ExactQueryPhraseMatch(%q, %q) = %v, want %v", c.Query, c.ProductTitle, v, c.Expected.TitleExactQueryMatch)
			}
			if c.Expected.BrandFamilyMatch != nil {
				if v := BrandFamilyMatch(c.Query, c.ProductBrand, c.ProductTitle); v != *c.Expected.BrandFamilyMatch {
					t.Errorf("BrandFamilyMatch(%q, %q, %q) = %v, want %v", c.Query, c.ProductBrand, c.ProductTitle, v, *c.Expected.BrandFamilyMatch)
				}
			}
			if c.Expected.SubbrandTitleMatch != nil {
				if v := SubbrandTitleMatch(c.Query, c.ProductTitle); v != *c.Expected.SubbrandTitleMatch {
					t.Errorf("SubbrandTitleMatch(%q, %q) = %v, want %v", c.Query, c.ProductTitle, v, *c.Expected.SubbrandTitleMatch)
				}
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

func categoryTokens(cases []fixtureCase) map[string]struct{} {
	m := make(map[string]struct{})
	for _, c := range cases {
		for _, part := range c.CategoryPath {
			for _, t := range Tokenize(part) {
				m[t] = struct{}{}
			}
		}
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
