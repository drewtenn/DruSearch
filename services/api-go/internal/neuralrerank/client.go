// Package neuralrerank calls the Python inference sidecar for transformer
// cross-encoder reranking.
package neuralrerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
)

type Client struct {
	baseURL string
	hc      *http.Client
}

func New(baseURL string, timeout time.Duration) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		hc:      &http.Client{Timeout: timeout},
	}
}

type Document struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type Score struct {
	ID    string  `json:"id"`
	Score float64 `json:"score"`
}

type rerankRequest struct {
	Query     string     `json:"query"`
	Documents []Document `json:"documents"`
}

type rerankResponse struct {
	Scores []Score `json:"scores"`
	Model  string  `json:"model"`
}

func (c *Client) Rerank(ctx context.Context, query string, docs []Document) ([]Score, string, error) {
	body, _ := json.Marshal(rerankRequest{Query: query, Documents: docs})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, "", fmt.Errorf("neural reranker %d: %s", resp.StatusCode, string(b))
	}
	var out rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, "", err
	}
	return out.Scores, out.Model, nil
}

func ProductText(h retrieval.Hit) string {
	parts := []string{
		"Title: " + h.Title,
		"Brand: " + h.Brand,
	}
	category := strings.Join(h.CategoryPath, " > ")
	if category == "" {
		category = h.Category
	}
	if category != "" {
		parts = append(parts, "Category: "+category)
	}
	if h.PriceCents > 0 {
		parts = append(parts, fmt.Sprintf("Price: $%.2f", float64(h.PriceCents)/100))
	} else {
		parts = append(parts, "Price: unknown")
	}
	return strings.Join(parts, "\n")
}
