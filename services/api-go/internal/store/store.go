package store

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	opensearch "github.com/opensearch-project/opensearch-go/v4"
	"github.com/redis/go-redis/v9"
)

type Stores struct {
	PG  *pgxpool.Pool
	RDB *redis.Client
	OS  *opensearch.Client
}

func Open(ctx context.Context, pgURL, redisAddr, osURL string) (*Stores, error) {
	pgCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	pg, err := pgxpool.New(pgCtx, pgURL)
	if err != nil {
		return nil, fmt.Errorf("postgres: %w", err)
	}

	rdb := redis.NewClient(&redis.Options{Addr: redisAddr})

	os, err := opensearch.NewClient(opensearch.Config{
		Addresses: []string{osURL},
		Transport: &http.Transport{
			MaxIdleConns:        10,
			IdleConnTimeout:     90 * time.Second,
			TLSHandshakeTimeout: 5 * time.Second,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("opensearch: %w", err)
	}

	return &Stores{PG: pg, RDB: rdb, OS: os}, nil
}

func (s *Stores) Close() {
	if s.PG != nil {
		s.PG.Close()
	}
	if s.RDB != nil {
		_ = s.RDB.Close()
	}
}

// Ping reports component health for /readyz.
type Health struct {
	Postgres   bool `json:"postgres"`
	Redis      bool `json:"redis"`
	OpenSearch bool `json:"opensearch"`
}

func (s *Stores) Ping(ctx context.Context) Health {
	h := Health{}
	{
		ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
		defer cancel()
		if err := s.PG.Ping(ctx); err == nil {
			h.Postgres = true
		}
	}
	{
		ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
		defer cancel()
		if _, err := s.RDB.Ping(ctx).Result(); err == nil {
			h.Redis = true
		}
	}
	{
		ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
		defer cancel()
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "/_cluster/health", nil)
		resp, err := s.OS.Perform(req)
		if err == nil && resp.StatusCode < 300 {
			h.OpenSearch = true
		}
		if resp != nil {
			_ = resp.Body.Close()
		}
	}
	return h
}
