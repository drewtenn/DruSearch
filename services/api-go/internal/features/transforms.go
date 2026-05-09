package features

import (
	"math"
	"regexp"
	"strings"
	"unicode"
)

const (
	GenderNone  = 0.0
	GenderMen   = 1.0
	GenderWomen = 2.0
	GenderBoys  = 3.0
	GenderGirls = 4.0
)

// Tokenize splits on Unicode word boundaries (matches Python re.findall(r"\w+")
// when both sides interpret \w as Unicode word). All tokens are lowercased.
func Tokenize(s string) []string {
	if s == "" {
		return nil
	}
	out := make([]string, 0, 8)
	var sb strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			sb.WriteRune(unicode.ToLower(r))
		} else {
			if sb.Len() > 0 {
				out = append(out, sb.String())
				sb.Reset()
			}
		}
	}
	if sb.Len() > 0 {
		out = append(out, sb.String())
	}
	return out
}

func QueryLengthTokens(query string) float64 {
	return float64(len(Tokenize(query)))
}

func QueryHasAny(query string, vocab map[string]struct{}) float64 {
	if len(vocab) == 0 || query == "" {
		return 0
	}
	for _, t := range Tokenize(query) {
		if _, ok := vocab[t]; ok {
			return 1
		}
	}
	return 0
}

// Identical regex to the Python reference impl (transforms.py).
var sizeRe = regexp.MustCompile(`(?i)\b\d+(?:\.\d+)?\s?(?:oz|ml|gb|tb|in|cm|mm|kg|lb|l|g)\b`)

func QueryHasSizePattern(query string) float64 {
	if query == "" {
		return 0
	}
	if sizeRe.FindStringIndex(query) != nil {
		return 1
	}
	return 0
}

func QueryAffordabilityIntent(query string) float64 {
	tokens := Tokenize(query)
	for i, t := range tokens {
		if affordabilityToken(t) {
			return 1
		}
		if i == 0 {
			continue
		}
		prev := tokens[i-1]
		if (prev == "low" || prev == "lower" || prev == "lowest") && (t == "cost" || t == "price") {
			return 1
		}
	}
	return 0
}

func AffordabilityPriceScore(queryAffordability float64, priceCents int) float64 {
	if queryAffordability == 0 || priceCents <= 0 {
		return 0
	}
	return 1 / math.Log1p(float64(priceCents))
}

func QueryGenderIntent(query string) float64 {
	found := GenderNone
	for _, t := range Tokenize(query) {
		g := queryGenderToken(t)
		if g == GenderNone {
			continue
		}
		if found != GenderNone && found != g {
			return GenderNone
		}
		found = g
	}
	return found
}

func affordabilityToken(t string) bool {
	switch t {
	case "affordable", "affordability", "cheap", "cheaper", "cheapest",
		"budget", "inexpensive", "economical", "value", "price", "priced",
		"pricing", "cost", "costs":
		return true
	default:
		return false
	}
}

func ProductGender(categoryPath []string) float64 {
	for _, part := range categoryPath {
		switch strings.ToLower(strings.TrimSpace(part)) {
		case "men":
			return GenderMen
		case "women":
			return GenderWomen
		case "boys":
			return GenderBoys
		case "girls":
			return GenderGirls
		}
	}
	return GenderNone
}

func GenderIntentMatch(queryGender, productGender float64) float64 {
	if queryGender != GenderNone && queryGender == productGender {
		return 1
	}
	return 0
}

func GenderIntentMismatch(queryGender, productGender float64) float64 {
	if queryGender != GenderNone && productGender != GenderNone && queryGender != productGender {
		return 1
	}
	return 0
}

func ProductBrandMatch(query, brand string) float64 {
	queryTokens := tokenSet(Tokenize(query))
	brandTokens := tokenSet(Tokenize(brand))
	for t := range brandTokens {
		if _, ok := queryTokens[t]; ok {
			return 1
		}
	}
	return 0
}

func ProductBrandTokenOverlap(query, brand string) float64 {
	queryTokens := tokenSet(Tokenize(query))
	brandTokens := tokenSet(Tokenize(brand))
	if len(brandTokens) == 0 {
		return 0
	}
	matched := 0
	for t := range brandTokens {
		if _, ok := queryTokens[t]; ok {
			matched++
		}
	}
	return float64(matched) / float64(len(brandTokens))
}

func ProductColorMatch(query, color string) float64 {
	return ProductBrandMatch(query, color)
}

func QueryTokenCoverage(query, text string, ignored map[string]struct{}) float64 {
	queryTokens := Tokenize(query)
	if len(queryTokens) == 0 {
		return 0
	}
	textTokens := tokenSet(Tokenize(text))
	keptTokens := make([]string, 0, len(queryTokens))
	for _, t := range queryTokens {
		if _, skip := ignored[t]; skip {
			continue
		}
		keptTokens = append(keptTokens, t)
	}
	if len(keptTokens) == 0 {
		keptTokens = queryTokens
	}
	matched := 0
	for _, t := range keptTokens {
		if _, ok := textTokens[t]; ok {
			matched++
		}
	}
	return float64(matched) / float64(len(keptTokens))
}

func ExactQueryPhraseMatch(query, text string) float64 {
	q := strings.Join(Tokenize(query), " ")
	haystack := strings.Join(Tokenize(text), " ")
	if q != "" && strings.Contains(haystack, q) {
		return 1
	}
	return 0
}

func TokenOverlapFraction(query, text string) float64 {
	queryTokens := tokenSet(Tokenize(query))
	textTokens := tokenSet(Tokenize(text))
	if len(textTokens) == 0 {
		return 0
	}
	matched := 0
	for t := range textTokens {
		if _, ok := queryTokens[t]; ok {
			matched++
		}
	}
	return float64(matched) / float64(len(textTokens))
}

func tokenSet(tokens []string) map[string]struct{} {
	out := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		out[t] = struct{}{}
	}
	return out
}

func queryGenderToken(t string) float64 {
	switch t {
	case "men", "mens", "man", "male":
		return GenderMen
	case "women", "womens", "woman", "female":
		return GenderWomen
	case "boys", "boy":
		return GenderBoys
	case "girls", "girl":
		return GenderGirls
	default:
		return GenderNone
	}
}
