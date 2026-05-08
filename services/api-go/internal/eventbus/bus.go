// Package eventbus buffers search events and flushes them to Postgres
// in batches. Submit is non-blocking: a full buffer logs and drops the
// event so the request hot path never stalls on the writer.
package eventbus

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"

	"github.com/drewtenn/drusearch/services/api-go/internal/obs"
)

type Event struct {
	Type      string
	Ts        time.Time
	UserID    string
	SessionID string
	Query     string
	QueryID   string
	ProductID string
	Position  int
	Scores    map[string]float64
	Source    string
}

type Bus struct {
	pool       *pgxpool.Pool
	log        *zap.Logger
	ch         chan Event
	flushSize  int
	flushEvery time.Duration

	dropped atomic.Uint64
	written atomic.Uint64
}

type Options struct {
	Buffer     int
	FlushSize  int
	FlushEvery time.Duration
}

func New(pool *pgxpool.Pool, log *zap.Logger, opts Options) *Bus {
	if opts.Buffer <= 0 {
		opts.Buffer = 8192
	}
	if opts.FlushSize <= 0 {
		opts.FlushSize = 500
	}
	if opts.FlushEvery <= 0 {
		opts.FlushEvery = 100 * time.Millisecond
	}
	return &Bus{
		pool:       pool,
		log:        log,
		ch:         make(chan Event, opts.Buffer),
		flushSize:  opts.FlushSize,
		flushEvery: opts.FlushEvery,
	}
}

func (b *Bus) Submit(e Event) {
	if e.Ts.IsZero() {
		e.Ts = time.Now().UTC()
	}
	if e.Source == "" {
		e.Source = "real"
	}
	select {
	case b.ch <- e:
	default:
		b.dropped.Add(1)
		obs.EventBusDropped.Inc()
	}
}

// Stats returns a snapshot of submitted/dropped counters.
func (b *Bus) Stats() (written, dropped uint64) {
	return b.written.Load(), b.dropped.Load()
}

// Run blocks until ctx is cancelled, draining and flushing the buffer
// before returning.
func (b *Bus) Run(ctx context.Context) {
	ticker := time.NewTicker(b.flushEvery)
	defer ticker.Stop()
	batch := make([]Event, 0, b.flushSize)

	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := b.insertBatch(ctx, batch); err != nil {
			b.log.Error("eventbus flush failed", zap.Int("n", len(batch)), zap.Error(err))
		} else {
			b.written.Add(uint64(len(batch)))
			obs.EventBusWritten.Add(float64(len(batch)))
		}
		batch = batch[:0]
	}

	drainAndExit := func() {
		for {
			select {
			case e := <-b.ch:
				batch = append(batch, e)
				if len(batch) >= b.flushSize {
					flush()
				}
			default:
				flush()
				return
			}
		}
	}

	for {
		select {
		case <-ctx.Done():
			drainAndExit()
			return
		case e := <-b.ch:
			batch = append(batch, e)
			if len(batch) >= b.flushSize {
				flush()
			}
		case <-ticker.C:
			flush()
		}
	}
}

func (b *Bus) insertBatch(ctx context.Context, batch []Event) error {
	rows := make([][]any, 0, len(batch))
	for _, e := range batch {
		var scoresJSON []byte
		if len(e.Scores) > 0 {
			scoresJSON, _ = json.Marshal(e.Scores)
		}
		var userID any
		if e.UserID != "" {
			userID = e.UserID
		}
		rows = append(rows, []any{
			e.Type, e.Ts, userID, e.SessionID, e.Query, e.QueryID,
			e.ProductID, e.Position, scoresJSON, e.Source,
		})
	}
	_, err := b.pool.CopyFrom(
		ctx,
		pgx.Identifier{"search_events"},
		[]string{
			"event_type", "ts", "user_id", "session_id", "query", "query_id",
			"product_id", "position", "retrieval_scores", "source",
		},
		pgx.CopyFromRows(rows),
	)
	return err
}
