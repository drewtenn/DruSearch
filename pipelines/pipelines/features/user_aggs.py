"""Compute per-user brand affinity from clicks and write to Redis.

Affinity definition:
    affinity(user, brand) = clicks(user, brand) / (total_clicks(user) + 1)

This is the user's click-share for the brand (smoothed by +1 in the
denominator so brand-new users don't divide-by-zero). The Go reranker
does an HGETALL per request keyed by `feat:user:{user_id}` and looks
up by exact brand string.

Run: docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.features.user_aggs
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

import redis

from pipelines.common import db
from pipelines.common.config import load
from pipelines.common.logging import configure

log = configure("features.user_aggs")

REDIS_KEY_PREFIX = os.getenv("USER_FEAT_PREFIX", "feat:user:")
TTL_SECONDS = int(os.getenv("USER_FEAT_TTL_SECONDS", "0"))  # 0 = no expiry


def _build_brand_share() -> dict[str, dict[str, float]]:
    """Returns {user_id: {brand: share}}."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with db.conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT e.user_id, COALESCE(p.brand, '')
            FROM search_events e
            JOIN products p USING (product_id)
            WHERE e.event_type = 'click'
              AND e.user_id IS NOT NULL
            """
        )
        for u, b in cur.fetchall():
            if not u or not b:
                continue
            counts[u][b] += 1

    out: dict[str, dict[str, float]] = {}
    for u, c in counts.items():
        total = sum(c.values()) + 1
        out[u] = {brand: cnt / total for brand, cnt in c.items()}
    return out


def main() -> int:
    cfg = load()
    rdb = redis.Redis(host=cfg.redis_host, port=cfg.redis_port, decode_responses=True)
    log.info("connected to redis at %s:%d", cfg.redis_host, cfg.redis_port)

    affinity = _build_brand_share()
    log.info("computed brand affinity for %d users", len(affinity))
    if not affinity:
        log.warning("no clicks found; nothing to write")
        return 0

    pipe = rdb.pipeline()
    written = 0
    for u, brands in affinity.items():
        if not brands:
            continue
        key = f"{REDIS_KEY_PREFIX}{u}"
        pipe.delete(key)
        # Store as field "brand_aff:<brand>" so we can co-tenant other features later.
        mapping = {f"brand_aff:{b}": f"{v:.6f}" for b, v in brands.items()}
        pipe.hset(key, mapping=mapping)
        if TTL_SECONDS > 0:
            pipe.expire(key, TTL_SECONDS)
        written += 1
        if written % 500 == 0:
            pipe.execute()
            pipe = rdb.pipeline()
    pipe.execute()

    log.info("wrote %d user feature hashes to redis", written)
    if written:
        sample_user = next(iter(affinity))
        sample = rdb.hgetall(f"{REDIS_KEY_PREFIX}{sample_user}")
        top = sorted(sample.items(), key=lambda kv: -float(kv[1]))[:5]
        log.info("sample user=%s top affinities=%s", sample_user, top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
