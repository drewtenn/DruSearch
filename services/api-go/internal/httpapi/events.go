package httpapi

import (
	"encoding/json"
	"net/http"
	"strings"

	"go.uber.org/zap"

	"github.com/drewtenn/drusearch/services/api-go/internal/eventbus"
	"github.com/drewtenn/drusearch/services/api-go/internal/obs"
)

type eventReq struct {
	EventType string             `json:"event_type"`
	QueryID   string             `json:"query_id"`
	Query     string             `json:"query"`
	SessionID string             `json:"session_id"`
	UserID    string             `json:"user_id"`
	ProductID string             `json:"product_id"`
	Position  int                `json:"position"`
	Scores    map[string]float64 `json:"retrieval_scores"`
	Source    string             `json:"source"`
}

var validEventTypes = map[string]bool{
	"impression": true,
	"click":      true,
	"purchase":   true,
}

func (s *Server) events(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var req eventReq
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4*1024)).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid json"})
		return
	}

	req.EventType = strings.TrimSpace(req.EventType)
	if !validEventTypes[req.EventType] {
		obs.EventsTotal.WithLabelValues(req.EventType, "invalid").Inc()
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "event_type must be impression|click|purchase"})
		return
	}
	if req.QueryID == "" || req.Query == "" || req.SessionID == "" || req.ProductID == "" {
		obs.EventsTotal.WithLabelValues(req.EventType, "invalid").Inc()
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "query_id, query, session_id, product_id required"})
		return
	}
	if req.Position < 0 {
		obs.EventsTotal.WithLabelValues(req.EventType, "invalid").Inc()
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "position must be >= 0"})
		return
	}
	if req.Source == "" {
		req.Source = "real"
	}

	s.Bus.Submit(eventbus.Event{
		Type:      req.EventType,
		UserID:    req.UserID,
		SessionID: req.SessionID,
		Query:     req.Query,
		QueryID:   req.QueryID,
		ProductID: req.ProductID,
		Position:  req.Position,
		Scores:    req.Scores,
		Source:    req.Source,
	})

	obs.EventsTotal.WithLabelValues(req.EventType, "ok").Inc()
	s.Logger.Debug("event accepted",
		zap.String("type", req.EventType),
		zap.String("query_id", req.QueryID),
		zap.String("product_id", req.ProductID),
	)
	writeJSON(w, http.StatusAccepted, map[string]any{"status": "queued"})
}
