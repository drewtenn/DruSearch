"""Compute product-level aggregate features (CTR, purchase rate) from the
event log and write them to product_features for online lookup.

Run: docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.features.aggregates
"""

from __future__ import annotations

import os

from pipelines.common import db
from pipelines.common.logging import configure

log = configure("features.aggregates")

LOOKBACK_DAYS = int(os.getenv("AGG_LOOKBACK_DAYS", "30"))

SQL = """
WITH agg AS (
  SELECT
    e.product_id,
    SUM(CASE WHEN e.event_type = 'impression' THEN 1 ELSE 0 END) AS impressions,
    SUM(CASE WHEN e.event_type = 'click'      THEN 1 ELSE 0 END) AS clicks,
    SUM(CASE WHEN e.event_type = 'purchase'   THEN 1 ELSE 0 END) AS purchases
  FROM search_events e
  JOIN products USING (product_id)
  WHERE e.ts > now() - (%s || ' days')::interval
  GROUP BY e.product_id
)
INSERT INTO product_features (product_id, impressions_30d, clicks_30d, purchases_30d, ctr_prior, pos_corrected_ctr, updated_at)
SELECT
  product_id,
  impressions,
  clicks,
  purchases,
  -- Beta(1, 99) smoothing
  (clicks + 1.0) / (impressions + 100.0)::real AS ctr_prior,
  -- placeholder: equal to ctr_prior for v1; IPS-weighted version arrives in Phase 7
  (clicks + 1.0) / (impressions + 100.0)::real AS pos_corrected_ctr,
  now()
FROM agg
ON CONFLICT (product_id) DO UPDATE SET
  impressions_30d   = EXCLUDED.impressions_30d,
  clicks_30d        = EXCLUDED.clicks_30d,
  purchases_30d     = EXCLUDED.purchases_30d,
  ctr_prior         = EXCLUDED.ctr_prior,
  pos_corrected_ctr = EXCLUDED.pos_corrected_ctr,
  updated_at        = EXCLUDED.updated_at
"""


def main() -> int:
    log.info("computing product aggregates lookback_days=%d", LOOKBACK_DAYS)
    with db.conn() as c, c.cursor() as cur:
        cur.execute(SQL, (LOOKBACK_DAYS,))
        affected = cur.rowcount
        cur.execute("SELECT COUNT(*), SUM(impressions_30d), SUM(clicks_30d), SUM(purchases_30d) FROM product_features")
        n, imps, clicks, purchases = cur.fetchone()
    c.commit()
    log.info(
        "wrote %d rows; covered %d products  impressions=%s clicks=%s purchases=%s",
        affected, n, imps, clicks, purchases,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
