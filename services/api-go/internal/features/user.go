package features

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// UserFeatures is the per-request snapshot of online features for a single user.
//
// BrandAff is a map from exact brand string (matching products.brand) to
// the user's click share for that brand in [0, 1].
type UserFeatures struct {
	BrandAff map[string]float64
}

// LoadUserFeatures fetches feat:user:<id> from Redis. Returns an empty
// (zero-affinity) snapshot when userID is empty, the key is missing, or
// Redis is unreachable — degraded ranking is preferred to a 500.
func LoadUserFeatures(ctx context.Context, rdb *redis.Client, userID string) *UserFeatures {
	out := &UserFeatures{BrandAff: map[string]float64{}}
	if userID == "" || rdb == nil {
		return out
	}
	cctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()

	h, err := rdb.HGetAll(cctx, fmt.Sprintf("feat:user:%s", userID)).Result()
	if err != nil {
		if !errors.Is(err, redis.Nil) {
			// Caller can log; we degrade silently to "no personalization".
		}
		return out
	}
	for k, v := range h {
		if strings.HasPrefix(k, "brand_aff:") {
			f, perr := strconv.ParseFloat(v, 64)
			if perr == nil {
				out.BrandAff[k[len("brand_aff:"):]] = f
			}
		}
	}
	return out
}
