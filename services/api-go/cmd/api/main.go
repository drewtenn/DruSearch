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
	"github.com/drewtenn/drusearch/services/api-go/internal/httpapi"
	"github.com/drewtenn/drusearch/services/api-go/internal/store"
)

func main() {
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

	emb := embedder.New(cfg.EmbedderURL, cfg.EmbedderTimeout)

	srv := &httpapi.Server{Logger: logger, Stores: stores, Embedder: emb}
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
