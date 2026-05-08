package embedder

import (
	"context"
	"errors"
	"time"

	"github.com/sony/gobreaker"

	"github.com/drewtenn/drusearch/services/api-go/internal/obs"
)

// Breaker wraps a Client and trips OPEN after consecutive failures.
// While OPEN, Embed returns ErrCircuitOpen immediately so the caller
// can degrade to BM25-only without paying the upstream timeout.
type Breaker struct {
	inner *Client
	cb    *gobreaker.CircuitBreaker
}

var ErrCircuitOpen = errors.New("embedder circuit open")

func NewBreaker(inner *Client, settings ...func(*gobreaker.Settings)) *Breaker {
	st := gobreaker.Settings{
		Name:        "embedder",
		MaxRequests: 1,                // probe one request when half-open
		Interval:    60 * time.Second, // reset failure count window
		Timeout:     10 * time.Second, // OPEN -> half-open after this
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures >= 5
		},
		OnStateChange: func(_ string, _ gobreaker.State, to gobreaker.State) {
			obs.EmbedderCircuit.Set(stateToFloat(to))
		},
	}
	for _, opt := range settings {
		opt(&st)
	}
	cb := gobreaker.NewCircuitBreaker(st)
	obs.EmbedderCircuit.Set(stateToFloat(cb.State()))
	return &Breaker{inner: inner, cb: cb}
}

func (b *Breaker) Embed(ctx context.Context, text string) ([]float32, error) {
	out, err := b.cb.Execute(func() (any, error) {
		return b.inner.Embed(ctx, text)
	})
	if err != nil {
		if errors.Is(err, gobreaker.ErrOpenState) || errors.Is(err, gobreaker.ErrTooManyRequests) {
			return nil, ErrCircuitOpen
		}
		return nil, err
	}
	return out.([]float32), nil
}

// Healthy is a passthrough: it does NOT exercise the breaker (so /readyz
// can keep reporting accurate sidecar liveness independent of the
// trip-and-degrade behaviour on the hot path).
func (b *Breaker) Healthy(ctx context.Context) bool {
	return b.inner.Healthy(ctx)
}

// State returns the current breaker state for observability.
func (b *Breaker) State() gobreaker.State {
	return b.cb.State()
}

func stateToFloat(s gobreaker.State) float64 {
	switch s {
	case gobreaker.StateClosed:
		return 0
	case gobreaker.StateHalfOpen:
		return 1
	case gobreaker.StateOpen:
		return 2
	}
	return -1
}
