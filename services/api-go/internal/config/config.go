package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	APIHost string
	APIPort int

	PostgresURL string

	RedisAddr string

	OpenSearchURL string

	EmbedderURL     string
	EmbedderTimeout time.Duration

	AdminToken string
}

func FromEnv() (Config, error) {
	c := Config{
		APIHost:         getenv("API_HOST", "0.0.0.0"),
		APIPort:         mustAtoi(getenv("API_PORT", "8080")),
		PostgresURL:     buildPostgresURL(),
		RedisAddr:       fmt.Sprintf("%s:%s", getenv("REDIS_HOST", "redis"), getenv("REDIS_PORT", "6379")),
		OpenSearchURL:   fmt.Sprintf("%s://%s:%s", getenv("OPENSEARCH_SCHEME", "http"), getenv("OPENSEARCH_HOST", "opensearch"), getenv("OPENSEARCH_PORT", "9200")),
		EmbedderURL:     fmt.Sprintf("http://%s:%s", getenv("EMBEDDER_HOST", "embedder"), getenv("EMBEDDER_PORT", "8000")),
		EmbedderTimeout: 2 * time.Second,
		AdminToken:      os.Getenv("ADMIN_TOKEN"),
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
