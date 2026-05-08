package httpapi

import (
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"

	"github.com/drewtenn/drusearch/services/api-go/internal/embedder"
	"github.com/drewtenn/drusearch/services/api-go/internal/eventbus"
	"github.com/drewtenn/drusearch/services/api-go/internal/features"
	"github.com/drewtenn/drusearch/services/api-go/internal/products"
	"github.com/drewtenn/drusearch/services/api-go/internal/rerank"
	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
	"github.com/drewtenn/drusearch/services/api-go/internal/store"
)

type Server struct {
	Logger     *zap.Logger
	Stores     *store.Stores
	Embedder   embedder.Interface
	Retrieval  *retrieval.Engine
	Products   *products.Store
	Bus        *eventbus.Bus
	Reranker   *rerank.Reranker
	Vocab      *features.Vocab
	AdminToken string
}

func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(5 * time.Second))

	r.Get("/healthz", s.healthz)
	r.Get("/readyz", s.readyz)
	r.Handle("/metrics", promhttp.Handler())

	r.Get("/search", s.search)
	r.Post("/events", s.events)
	r.Get("/products/{id}", s.productByID)

	r.Route("/admin", func(ar chi.Router) {
		ar.Use(s.requireAdminToken)
		ar.Post("/reload-model", s.reloadModel)
		ar.Post("/reindex", s.reindex)
	})

	return r
}
