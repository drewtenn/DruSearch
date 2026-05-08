package httpapi

import (
	"net/http"

	"go.uber.org/zap"
)

func (s *Server) reloadModel(w http.ResponseWriter, _ *http.Request) {
	if s.Reranker == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "reranker not configured"})
		return
	}
	loaded, err := s.Reranker.Reload()
	if err != nil {
		s.Logger.Warn("reload-model failed", zap.Error(err))
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":    "ok",
		"path":      loaded.Path,
		"loaded_at": loaded.LoadedAt,
		"meta":      loaded.Meta,
	})
}

func (s *Server) reindex(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{"error": "reindex: phase 7"})
}
