package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	APIHost    string
	APIPort    int
	APITimeout time.Duration

	PostgresURL string

	RedisAddr string

	OpenSearchURL   string
	OpenSearchIndex string

	EmbedderURL     string
	EmbedderTimeout time.Duration

	DefaultRanker string

	AdminToken string

	LTRModelDir  string
	LTRModelName string
}

func FromEnv() (Config, error) {
	c := Config{
		APIHost:         getenv("API_HOST", "0.0.0.0"),
		APIPort:         mustAtoi(getenv("API_PORT", "8080")),
		APITimeout:      time.Duration(mustAtoi(getenv("API_TIMEOUT_SECONDS", "30"))) * time.Second,
		PostgresURL:     buildPostgresURL(),
		RedisAddr:       fmt.Sprintf("%s:%s", getenv("REDIS_HOST", "redis"), getenv("REDIS_PORT", "6379")),
		OpenSearchURL:   fmt.Sprintf("%s://%s:%s", getenv("OPENSEARCH_SCHEME", "http"), getenv("OPENSEARCH_HOST", "opensearch"), getenv("OPENSEARCH_PORT", "9200")),
		OpenSearchIndex: getenv("OPENSEARCH_INDEX", "products_v1"),
		EmbedderURL:     fmt.Sprintf("http://%s:%s", getenv("EMBEDDER_HOST", "embedder"), getenv("EMBEDDER_PORT", "8000")),
		EmbedderTimeout: 2 * time.Second,
		DefaultRanker:   getenv("DEFAULT_RANKER", "ltr"),
		AdminToken:      os.Getenv("ADMIN_TOKEN"),
		LTRModelDir:     getenv("LTR_MODEL_DIR", "/var/lib/drusearch/models"),
		LTRModelName:    getenv("LTR_MODEL_NAME", "ltr_reranker"),
	}
	return c, nil
}

func buildPostgresURL() string {
	return fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=disable",
		getenv("POSTGRES_USER", "drusearch"),
		getenv("POSTGRES_PASSWORD", "drusearch"),
		getenv("POSTGRES_HOST", "postgres"),
		getenv("POSTGRES_PORT", "5432"),
		getenv("POSTGRES_DB", "drusearch"),
	)
}

func getenv(k, def string) string {
	if v, ok := os.LookupEnv(k); ok && v != "" {
		return v
	}
	return def
}

func mustAtoi(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		panic(err)
	}
	return n
}
