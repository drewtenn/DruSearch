package features

import (
	"math"
	"regexp"
	"strings"
	"unicode"
)

const (
	GenderNone   = 0.0
	GenderMen    = 1.0
	GenderWomen  = 2.0
	GenderBoys   = 3.0
	GenderGirls  = 4.0
	GenderUnisex = 5.0
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

func ProductGender(categoryPath []string, title string) float64 {
	for _, part := range categoryPath {
		if gender := ProductGenderLabel(part); gender != GenderNone {
			return gender
		}
	}
	return titleGender(title)
}

func ProductGenderLabel(label string) float64 {
	switch strings.ToLower(strings.TrimSpace(label)) {
	case "men":
		return GenderMen
	case "women":
		return GenderWomen
	case "boys":
		return GenderBoys
	case "girls":
		return GenderGirls
	case "unisex":
		return GenderUnisex
	default:
		return GenderNone
	}
}

func GenderIntentMatch(queryGender, productGender float64) float64 {
	if queryGender != GenderNone && queryGender == productGender {
		return 1
	}
	if (queryGender == GenderMen || queryGender == GenderWomen) && productGender == GenderUnisex {
		return 0.5
	}
	return 0
}

func GenderIntentMismatch(queryGender, productGender float64) float64 {
	if productGender == GenderUnisex {
		return 0
	}
	if queryGender != GenderNone && productGender != GenderNone && queryGender != productGender {
		return 1
	}
	return 0
}

func ProductBrandMatch(query, brand string) float64 {
	queryTokens := tokenSet(BrandTokens(query))
	brandTokens := tokenSet(BrandTokens(brand))
	for t := range brandTokens {
		if _, ok := queryTokens[t]; ok {
			return 1
		}
	}
	return 0
}

func ProductBrandTokenOverlap(query, brand string) float64 {
	queryTokens := tokenSet(BrandTokens(query))
	brandTokens := tokenSet(BrandTokens(brand))
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

func SubbrandTitleMatch(query, title string) float64 {
	queryTokens := tokenSet(BrandTokens(query))
	if len(queryTokens) == 0 {
		return 0
	}
	titleTokens := Tokenize(title)
	for subbrand := range queryTokens {
		if !knownSubbrand(subbrand) {
			continue
		}
		if titleContainsSubbrandAlias(titleTokens, subbrand) {
			return 1
		}
	}
	return 0
}

func BrandFamilyMatch(query, brand, title string) float64 {
	queryTokens := tokenSet(BrandTokens(query))
	brandTokens := tokenSet(BrandTokens(brand))
	if len(queryTokens) == 0 || len(brandTokens) == 0 {
		return 0
	}
	for t := range brandTokens {
		if _, ok := queryTokens[t]; ok {
			return 1
		}
	}
	for subbrand := range queryTokens {
		if !subbrandParentBrandMatch(subbrand, brandTokens) {
			continue
		}
		if SubbrandTitleMatch(query, title) == 1 {
			return 1
		}
	}
	return 0
}

func ProductColorMatch(query, color string) float64 {
	return ProductBrandMatch(query, color)
}

func BrandTokens(text string) []string {
	tokens := Tokenize(text)
	out := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if brandStopToken(token) {
			continue
		}
		out = append(out, token)
	}
	return out
}

func brandStopToken(token string) bool {
	switch token {
	case "accessories", "accessory", "active", "athletic", "basketball",
		"boy", "boys", "clothing", "fashion", "girl", "girls", "jewelry",
		"men", "mens", "running", "shoe", "shoes", "sneaker", "sneakers",
		"sports", "team", "watch", "watches", "woman", "women", "womens":
		return true
	case "unisex":
		return true
	default:
		return false
	}
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

func knownSubbrand(subbrand string) bool {
	switch subbrand {
	case "jordan":
		return true
	default:
		return false
	}
}

func subbrandParentBrandMatch(subbrand string, brandTokens map[string]struct{}) bool {
	switch subbrand {
	case "jordan":
		_, ok := brandTokens["nike"]
		return ok
	default:
		return false
	}
}

func titleContainsSubbrandAlias(titleTokens []string, subbrand string) bool {
	switch subbrand {
	case "jordan":
		return containsTokenSequence(titleTokens, []string{"air", "jordan"}) ||
			containsTokenSequence(titleTokens, []string{"jordan"})
	default:
		return false
	}
}

func containsTokenSequence(tokens, sequence []string) bool {
	if len(sequence) == 0 || len(sequence) > len(tokens) {
		return false
	}
	for i := 0; i <= len(tokens)-len(sequence); i++ {
		match := true
		for j, want := range sequence {
			if tokens[i+j] != want {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
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
	case "unisex":
		return GenderUnisex
	default:
		return GenderNone
	}
}

func titleGender(title string) float64 {
	found := GenderNone
	for _, t := range Tokenize(title) {
		g := queryGenderToken(t)
		if g == GenderNone {
			continue
		}
		if g == GenderUnisex {
			return GenderUnisex
		}
		if found != GenderNone && found != g {
			if (found == GenderMen && g == GenderWomen) || (found == GenderWomen && g == GenderMen) {
				return GenderUnisex
			}
			return GenderNone
		}
		found = g
	}
	return found
}
