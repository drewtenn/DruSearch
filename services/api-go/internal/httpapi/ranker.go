package httpapi

import (
	"fmt"
	"strings"
)

type rankerMode string

const (
	rankerHybrid rankerMode = "hybrid"
	rankerLTR    rankerMode = "ltr"
	rankerBGE    rankerMode = "bge"
)

func rankerFromRequest(requested, def string) (rankerMode, error) {
	value := strings.ToLower(strings.TrimSpace(requested))
	if value == "" {
		value = strings.ToLower(strings.TrimSpace(def))
	}
	if value == "" {
		return rankerHybrid, nil
	}
	switch value {
	case "hybrid", "rrf", "bm25_knn", "bm25/knn":
		return rankerHybrid, nil
	case "ltr":
		return rankerLTR, nil
	case "bge":
		return rankerBGE, nil
	default:
		return "", fmt.Errorf("unknown ranker %q", requested)
	}
}
