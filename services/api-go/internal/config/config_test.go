package config

import "testing"

func TestFromEnvDefaultsToLTRRanker(t *testing.T) {
	t.Setenv("DEFAULT_RANKER", "")

	cfg, err := FromEnv()
	if err != nil {
		t.Fatalf("FromEnv: %v", err)
	}

	if cfg.DefaultRanker != "ltr" {
		t.Fatalf("DefaultRanker = %q, want %q", cfg.DefaultRanker, "ltr")
	}
}
