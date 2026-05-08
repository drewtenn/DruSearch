// Package obs centralises Prometheus metrics for the API.
//
// Metrics are registered against the default registry so the chi router
// can simply expose `/metrics` via promhttp.Handler().
package obs

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// SearchTotal counts /search requests by mode (hybrid, hybrid+ltr, bm25)
	// and outcome (ok, error).
	SearchTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "drusearch_search_requests_total",
		Help: "Number of /search requests partitioned by mode and outcome.",
	}, []string{"mode", "outcome"})

	// SearchLatency is the end-to-end /search latency (handler time).
	SearchLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "drusearch_search_latency_seconds",
		Help:    "End-to-end /search latency in seconds.",
		Buckets: prometheus.ExponentialBucketsRange(0.001, 1.0, 12),
	}, []string{"mode"})

	// StageLatency breaks the request down per pipeline stage.
	StageLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "drusearch_stage_latency_seconds",
		Help:    "Per-stage /search latency in seconds (embed, retrieve, user_features, rerank).",
		Buckets: prometheus.ExponentialBucketsRange(0.0005, 0.5, 12),
	}, []string{"stage"})

	// CandidatesPerRequest tracks how many candidates the retriever returned.
	CandidatesPerRequest = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "drusearch_retrieval_candidates",
		Help:    "Candidate pool size returned by hybrid retrieval before rerank.",
		Buckets: []float64{1, 10, 25, 50, 100, 200, 400, 800},
	})

	// EventsTotal counts inbound events on /events.
	EventsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "drusearch_events_total",
		Help: "Number of /events submissions partitioned by type and outcome.",
	}, []string{"type", "outcome"})

	// EventBus stats.
	EventBusWritten = promauto.NewCounter(prometheus.CounterOpts{
		Name: "drusearch_eventbus_written_total",
		Help: "Events successfully flushed to Postgres by the event bus.",
	})
	EventBusDropped = promauto.NewCounter(prometheus.CounterOpts{
		Name: "drusearch_eventbus_dropped_total",
		Help: "Events dropped due to a full buffer.",
	})

	// EmbedderCircuit reflects the live state of the embedder circuit
	// breaker: 0=closed (healthy), 1=half-open, 2=open (tripped).
	EmbedderCircuit = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "drusearch_embedder_circuit_state",
		Help: "Embedder circuit-breaker state (0=closed, 1=half-open, 2=open).",
	})

	// ModelLoaded reports the LTR model version currently in memory
	// (a single info-style metric labelled with name + version).
	ModelLoaded = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "drusearch_ltr_model_loaded_info",
		Help: "Constant 1 per loaded LTR model labelled with name and version.",
	}, []string{"name", "version"})
)
