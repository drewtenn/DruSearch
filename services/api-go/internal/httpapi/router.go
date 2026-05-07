package httpapi

import (
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"go.uber.org/zap"

	"github.com/drewtenn/drusearch/services/api-go/internal/embedder"
	"github.com/drewtenn/drusearch/services/api-go/internal/store"
)

type Server struct {
	Logger   *zap.Logger
	Stores   *store.Stores
	Embedder *embedder.Client
}

func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(5 * time.Second))

	r.Get("/healthz", s.healthz)
	r.Get("/readyz", s.readyz)

	r.Get("/search", s.search)
	r.Post("/events", s.events)
	r.Get("/products/{id}", s.productByID)

	r.Route("/admin", func(ar chi.Router) {
		ar.Post("/reload-model", s.reloadModel)
		ar.Post("/reindex", s.reindex)
	})

	return r
}
