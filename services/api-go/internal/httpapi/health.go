package httpapi

import (
	"encoding/json"
	"net/http"
)

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

type readyResp struct {
	Status     string `json:"status"`
	Postgres   bool   `json:"postgres"`
	Redis      bool   `json:"redis"`
	OpenSearch bool   `json:"opensearch"`
	Embedder   bool   `json:"embedder"`
}

func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	h := s.Stores.Ping(r.Context())
	emb := s.Embedder.Healthy(r.Context())
	resp := readyResp{
		Status:     "ok",
		Postgres:   h.Postgres,
		Redis:      h.Redis,
		OpenSearch: h.OpenSearch,
		Embedder:   emb,
	}
	code := http.StatusOK
	if !(h.Postgres && h.Redis && h.OpenSearch && emb) {
		resp.Status = "not_ready"
		code = http.StatusServiceUnavailable
	}
	writeJSON(w, code, resp)
}

func writeJSON(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(body)
}
