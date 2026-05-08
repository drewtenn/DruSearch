package features

import (
	"regexp"
	"strings"
	"unicode"
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
