package httpapi

import "net/http"

// Phase 0 stubs. Phase 1+ replaces these with real implementations.

func (s *Server) search(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{"error": "search: phase 1"})
}

func (s *Server) events(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{"error": "events: phase 3"})
}

func (s *Server) productByID(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{"error": "products: phase 1"})
}

func (s *Server) reloadModel(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{"error": "reload-model: phase 5"})
}

func (s *Server) reindex(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{"error": "reindex: phase 1"})
}
