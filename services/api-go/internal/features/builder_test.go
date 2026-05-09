package features

import (
	"math"
	"testing"

	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
)

func TestBuildMatrixAffordabilityPriceScore(t *testing.T) {
	hits := []retrieval.Hit{
		{ProductID: "cheap", PriceCents: 1_000},
		{ProductID: "premium", PriceCents: 10_000},
	}

	matrix := BuildMatrix("affordable running shoes", hits, nil, nil)
	cheapOff := 0
	premiumOff := NumFeatures

	if got := matrix[cheapOff+IdxQueryAffordabilityIntent]; got != 1 {
		t.Fatalf("cheap row affordability intent = %v, want 1", got)
	}
	if got := matrix[premiumOff+IdxQueryAffordabilityIntent]; got != 1 {
		t.Fatalf("premium row affordability intent = %v, want 1", got)
	}

	cheapScore := matrix[cheapOff+IdxAffordabilityPriceScore]
	premiumScore := matrix[premiumOff+IdxAffordabilityPriceScore]
	if !(cheapScore > premiumScore) {
		t.Fatalf("affordability price score should prefer cheaper products: cheap=%v premium=%v", cheapScore, premiumScore)
	}
	wantCheap := 1 / math.Log1p(1000)
	if math.Abs(cheapScore-wantCheap) > 1e-12 {
		t.Fatalf("cheap affordability score = %.15f, want %.15f", cheapScore, wantCheap)
	}
}

func TestBuildMatrixAffordabilityPriceScoreInactiveWithoutIntent(t *testing.T) {
	matrix := BuildMatrix("running shoes", []retrieval.Hit{{ProductID: "p", PriceCents: 1_000}}, nil, nil)

	if got := matrix[IdxQueryAffordabilityIntent]; got != 0 {
		t.Fatalf("affordability intent = %v, want 0", got)
	}
	if got := matrix[IdxAffordabilityPriceScore]; got != 0 {
		t.Fatalf("affordability price score = %v, want 0", got)
	}
}

func TestBuildMatrixEncodesMissingRanksAsWorseThanSeenRanks(t *testing.T) {
	hits := []retrieval.Hit{
		{ProductID: "both", BM25Rank: 1, KNNRank: 2},
		{ProductID: "bm25_only", BM25Rank: 2, KNNRank: 0},
		{ProductID: "knn_only", BM25Rank: 0, KNNRank: 1},
	}

	matrix := BuildMatrix("running shoes", hits, nil, nil)

	if got := matrix[0+IdxBM25Rank]; got != 1 {
		t.Fatalf("both bm25 rank = %v, want 1", got)
	}
	if got := matrix[0+IdxKNNRank]; got != 2 {
		t.Fatalf("both knn rank = %v, want 2", got)
	}
	if got := matrix[NumFeatures+IdxKNNRank]; got != 3 {
		t.Fatalf("bm25-only knn rank = %v, want 3", got)
	}
	if got := matrix[2*NumFeatures+IdxBM25Rank]; got != 3 {
		t.Fatalf("knn-only bm25 rank = %v, want 3", got)
	}
}

func TestBuildMatrixUsesDerivedGenderWithUnisexPartialMatch(t *testing.T) {
	hits := []retrieval.Hit{
		{ProductID: "unisex", Title: "Nike Unisex Running Shoes", DerivedGender: "unisex"},
		{ProductID: "women", Title: "Nike Women's Running Shoes", DerivedGender: "women"},
	}

	matrix := BuildMatrix("nike mens shoes", hits, nil, nil)

	if got := matrix[0+IdxProductGender]; got != GenderUnisex {
		t.Fatalf("unisex product gender = %v, want %v", got, GenderUnisex)
	}
	if got := matrix[0+IdxGenderIntentMatch]; got != 0.5 {
		t.Fatalf("unisex gender intent match = %v, want 0.5", got)
	}
	if got := matrix[0+IdxGenderIntentMis]; got != 0 {
		t.Fatalf("unisex gender intent mismatch = %v, want 0", got)
	}

	if got := matrix[NumFeatures+IdxProductGender]; got != GenderWomen {
		t.Fatalf("women product gender = %v, want %v", got, GenderWomen)
	}
	if got := matrix[NumFeatures+IdxGenderIntentMatch]; got != 0 {
		t.Fatalf("women gender intent match = %v, want 0", got)
	}
	if got := matrix[NumFeatures+IdxGenderIntentMis]; got != 1 {
		t.Fatalf("women gender intent mismatch = %v, want 1", got)
	}
}

func TestQueryTokenCoverageKeepsBrandTokensWhenQueryIsOnlyBrand(t *testing.T) {
	brands := map[string]struct{}{"jordan": {}}

	if got := QueryTokenCoverage("jordan", "Air Jordan Future", brands); got != 1 {
		t.Fatalf("coverage for title match = %v, want 1", got)
	}
	if got := QueryTokenCoverage("jordan", "Anti Crease Shoe Guard", brands); got != 0 {
		t.Fatalf("coverage for title mismatch = %v, want 0", got)
	}
}

func TestGenericBrandTokensDoNotCountAsBrandMatches(t *testing.T) {
	if got := ProductBrandMatch("mens jordan basketball", "Altra Running Mens"); got != 0 {
		t.Fatalf("generic brand match = %v, want 0", got)
	}
	if got := ProductBrandTokenOverlap("mens jordan basketball", "Altra Running Mens"); got != 0 {
		t.Fatalf("generic brand overlap = %v, want 0", got)
	}
	if got := ProductBrandMatch("mens jordan basketball", "Jordan"); got != 1 {
		t.Fatalf("jordan brand match = %v, want 1", got)
	}
}

func TestBrandFamilyMatchRequiresSubbrandEvidenceForParentBrand(t *testing.T) {
	if got := BrandFamilyMatch("mens jordan basketball", "Nike", "Nike Men's Air Jordan 1 Mid Shoes"); got != 1 {
		t.Fatalf("nike air jordan family match = %v, want 1", got)
	}
	if got := BrandFamilyMatch("mens jordan basketball", "Nike", "Nike Mens PG 5 Basketball Shoe"); got != 0 {
		t.Fatalf("generic nike basketball family match = %v, want 0", got)
	}
	if got := BrandFamilyMatch("mens jordan basketball", "Jordan", "Air Jordan Future"); got != 1 {
		t.Fatalf("jordan brand family match = %v, want 1", got)
	}
	if got := SubbrandTitleMatch("mens jordan basketball", "Nike Men's Air Jordan 1 Mid Shoes"); got != 1 {
		t.Fatalf("air jordan title subbrand match = %v, want 1", got)
	}
	if got := SubbrandTitleMatch("mens jordan basketball", "Nike Mens PG 5 Basketball Shoe"); got != 0 {
		t.Fatalf("generic nike title subbrand match = %v, want 0", got)
	}
}
