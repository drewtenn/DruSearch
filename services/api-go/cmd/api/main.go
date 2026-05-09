package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.uber.org/zap"

	"github.com/drewtenn/drusearch/services/api-go/internal/config"
	"github.com/drewtenn/drusearch/services/api-go/internal/embedder"
	"github.com/drewtenn/drusearch/services/api-go/internal/eventbus"
	"github.com/drewtenn/drusearch/services/api-go/internal/features"
	"github.com/drewtenn/drusearch/services/api-go/internal/httpapi"
	"github.com/drewtenn/drusearch/services/api-go/internal/products"
	"github.com/drewtenn/drusearch/services/api-go/internal/rerank"
	"github.com/drewtenn/drusearch/services/api-go/internal/retrieval"
	"github.com/drewtenn/drusearch/services/api-go/internal/store"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "healthcheck" {
		os.Exit(runHealthcheck())
	}

	logger, _ := zap.NewProduction()
	defer logger.Sync() //nolint:errcheck

	cfg, err := config.FromEnv()
	if err != nil {
		logger.Fatal("config", zap.Error(err))
	}

	rootCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	stores, err := store.Open(rootCtx, cfg.PostgresURL, cfg.RedisAddr, cfg.OpenSearchURL)
	if err != nil {
		logger.Fatal("stores", zap.Error(err))
	}
	defer stores.Close()

	embClient := embedder.New(cfg.EmbedderURL, cfg.EmbedderTimeout)
	emb := embedder.NewBreaker(embClient)
	ret := retrieval.New(stores.OS, cfg.OpenSearchIndex)
	prods := products.New(stores.PG)
	bus := eventbus.New(stores.PG, logger, eventbus.Options{})

	go bus.Run(rootCtx)

	// Catalog vocab for interaction features (brand/color/category token sets).
	vocab, err := features.LoadVocab(rootCtx, stores.PG)
	if err != nil {
		logger.Warn("vocab load failed; reranker will not run until /admin/reload-model after products exist",
			zap.Error(err))
		vocab = &features.Vocab{
			Brand:    map[string]struct{}{},
			Color:    map[string]struct{}{},
			Category: map[string]struct{}{},
		}
	} else {
		logger.Info("vocab loaded",
			zap.Int("brand_tokens", len(vocab.Brand)),
			zap.Int("color_tokens", len(vocab.Color)),
			zap.Int("category_tokens", len(vocab.Category)),
		)
	}

	// Optional LTR reranker; nil-ok if no model is on disk yet.
	rr := rerank.New(cfg.LTRModelDir, cfg.LTRModelName)
	if loaded, err := rr.Reload(); err != nil {
		logger.Warn("LTR model not loaded at boot; /search will use RRF only",
			zap.String("dir", cfg.LTRModelDir), zap.Error(err))
	} else {
		logger.Info("LTR model loaded",
			zap.String("path", loaded.Path), zap.Any("meta", loaded.Meta))
	}

	if cfg.AdminToken == "" {
		logger.Warn("ADMIN_TOKEN not set; /admin/* will reject all requests")
	}
	srv := &httpapi.Server{
		Logger:         logger,
		Stores:         stores,
		Embedder:       emb,
		Retrieval:      ret,
		Products:       prods,
		Bus:            bus,
		Reranker:       rr,
		Vocab:          vocab,
		AdminToken:     cfg.AdminToken,
		DefaultRanker:  cfg.DefaultRanker,
		RequestTimeout: cfg.APITimeout,
	}
	httpServer := &http.Server{
		Addr:              fmt.Sprintf("%s:%d", cfg.APIHost, cfg.APIPort),
		Handler:           srv.Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		logger.Info("listening", zap.String("addr", httpServer.Addr))
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Fatal("listen", zap.Error(err))
		}
	}()

	<-rootCtx.Done()
	logger.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownCtx)
}

func runHealthcheck() int {
	port := os.Getenv("API_PORT")
	if port == "" {
		port = "8080"
	}
	c := &http.Client{Timeout: 2 * time.Second}
	resp, err := c.Get(fmt.Sprintf("http://127.0.0.1:%s/healthz", port))
	if err != nil {
		return 1
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 1
	}
	return 0
}
