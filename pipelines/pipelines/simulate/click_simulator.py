"""Synthetic click simulator that drives the real /search and /events APIs.

Click model (per impression of product p at rank r for user u on query q):

    P(click) = examination(r) * relevance(u, q, p)
    examination(r) = 1 / (r + 1)^eta
    relevance(u, q, p) = sigmoid(
        w_judg * judgment_score(q, p)
      + w_pop  * popularity_prior(p)
      + w_brand * brand_pref(u, brand(p))
      + N(0, sigma)
    )
    P(purchase | click) = clip(0.05 + 0.10 * relevance, 0.05, 0.15)

Per-user brand affinity drifts +0.05 toward clicked brands (clipped to [-0.5, 0.5]).
ESCI judgment score: E=1.0, S=0.5, C=0.1, I=-0.5, missing=0.

Events are POSTed back to /events. Impressions are auto-logged by the
search handler — the simulator only emits clicks and purchases.

Run:
    docker compose --profile jobs run --rm pipelines \\
        python -m pipelines.simulate.click_simulator
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import random
import secrets
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import httpx

from pipelines.common import db
from pipelines.common.logging import configure

log = configure("simulate.click")


# ---------------------------------------------------------------------------
# Tunable model parameters (env-overridable for experiments)
# ---------------------------------------------------------------------------

JUDG_SCORE = {"E": 1.0, "S": 0.5, "C": 0.1, "I": -0.5}

W_JUDG = float(os.getenv("SIM_W_JUDG", "1.5"))
W_POP = float(os.getenv("SIM_W_POP", "0.4"))
W_BRAND = float(os.getenv("SIM_W_BRAND", "0.6"))
SIGMA = float(os.getenv("SIM_SIGMA", "0.25"))
ETA = float(os.getenv("SIM_ETA", "1.0"))
AFFINITY_STEP = float(os.getenv("SIM_AFFINITY_STEP", "0.05"))
AFFINITY_CLIP = float(os.getenv("SIM_AFFINITY_CLIP", "0.5"))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Catalog state
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Catalog:
    queries: list[str]
    judgments: dict[str, dict[str, str]]  # query -> {product_id: 'E'/'S'/'C'/'I'}
    product_pop: dict[str, float]
    product_brand: dict[str, str]


def _load_catalog() -> Catalog:
    log.info("loading catalog from postgres")
    judgments: dict[str, dict[str, str]] = {}
    pop: dict[str, float] = {}
    brand: dict[str, str] = {}

    with db.conn() as c:
        with c.cursor() as cur:
            cur.execute("SELECT query, product_id, esci_label FROM esci_judgments")
            for q, pid, lbl in cur:
                judgments.setdefault(q, {})[pid] = lbl
        with c.cursor() as cur:
            cur.execute(
                "SELECT product_id, COALESCE(brand, ''), COALESCE(popularity_prior, 0)"
                " FROM products"
            )
            for pid, br, p in cur:
                brand[pid] = br
                pop[pid] = float(p)

    queries = sorted(judgments.keys())
    log.info("catalog: queries=%d products=%d", len(queries), len(brand))
    return Catalog(queries=queries, judgments=judgments, product_pop=pop, product_brand=brand)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class UserStats:
    user_id: str
    impressions: int = 0
    clicks: int = 0
    purchases: int = 0


def _post_event(client: httpx.Client, base_url: str, payload: dict) -> None:
    try:
        r = client.post(f"{base_url}/events", json=payload, timeout=5.0)
        r.raise_for_status()
    except Exception as exc:  # pragma: no cover - best-effort
        log.warning("event post failed type=%s err=%s", payload.get("event_type"), exc)


def _search_params(query: str, user_id: str, session_id: str, k: int, ranker: str) -> dict:
    params = {"q": query, "user_id": user_id, "session_id": session_id, "k": k}
    if ranker:
        params["ranker"] = ranker
    return params


def _simulate_user(
    user_idx: int,
    rng: random.Random,
    cat: Catalog,
    base_url: str,
    queries_per_user: int,
    k: int,
    ranker: str,
) -> UserStats:
    user_id = f"u_{user_idx:05d}"
    session_id = f"sess_{user_id}_{secrets.token_hex(4)}"
    brand_aff: dict[str, float] = {}
    stats = UserStats(user_id=user_id)

    queries = rng.sample(cat.queries, min(queries_per_user, len(cat.queries)))

    with httpx.Client(http2=False, timeout=10.0) as client:
        for q in queries:
            try:
                resp = client.get(
                    f"{base_url}/search",
                    params=_search_params(q, user_id, session_id, k, ranker),
                )
                resp.raise_for_status()
            except Exception as exc:
                log.warning("search failed q=%r err=%s", q, exc)
                continue

            body = resp.json()
            results = body.get("results", [])
            query_id = body["query_id"]
            q_judg = cat.judgments.get(q, {})

            for rank, hit in enumerate(results):
                stats.impressions += 1
                pid = hit["product_id"]
                br = hit.get("brand", "") or cat.product_brand.get(pid, "")
                judgment = JUDG_SCORE.get(q_judg.get(pid), 0.0)
                pop = cat.product_pop.get(pid, 0.0)
                aff = brand_aff.get(br, 0.0)

                relevance = _sigmoid(
                    W_JUDG * judgment
                    + W_POP * pop
                    + W_BRAND * aff
                    + rng.gauss(0.0, SIGMA)
                )
                examination = 1.0 / ((rank + 1) ** ETA)
                p_click = examination * relevance

                if rng.random() < p_click:
                    stats.clicks += 1
                    _post_event(
                        client,
                        base_url,
                        {
                            "event_type": "click",
                            "query_id": query_id,
                            "query": q,
                            "session_id": session_id,
                            "user_id": user_id,
                            "product_id": pid,
                            "position": rank,
                            "retrieval_scores": {
                                "bm25": hit["explain"]["bm25"],
                                "knn":  hit["explain"]["knn"],
                                "rrf":  hit["explain"]["rrf"],
                            },
                            "source": "synthetic",
                        },
                    )
                    if br:
                        new = max(-AFFINITY_CLIP, min(AFFINITY_CLIP, aff + AFFINITY_STEP))
                        brand_aff[br] = new

                    p_purchase = max(0.05, min(0.15, 0.05 + 0.10 * relevance))
                    if rng.random() < p_purchase:
                        stats.purchases += 1
                        _post_event(
                            client,
                            base_url,
                            {
                                "event_type": "purchase",
                                "query_id": query_id,
                                "query": q,
                                "session_id": session_id,
                                "user_id": user_id,
                                "product_id": pid,
                                "position": rank,
                                "source": "synthetic",
                            },
                        )

    return stats


def _summarise(stats: Iterable[UserStats]) -> str:
    items = list(stats)
    impressions = sum(s.impressions for s in items)
    clicks = sum(s.clicks for s in items)
    purchases = sum(s.purchases for s in items)
    ctr = clicks / impressions if impressions else 0.0
    pr = purchases / clicks if clicks else 0.0
    per_user_clicks = [s.clicks for s in items]
    return (
        f"users={len(items)} impressions={impressions} clicks={clicks} "
        f"purchases={purchases} CTR={ctr:.3f} purchase|click={pr:.3f} "
        f"clicks/user p50={statistics.median(per_user_clicks):.1f} "
        f"max={max(per_user_clicks) if per_user_clicks else 0}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="DruSearch click simulator")
    p.add_argument("--users", type=int, default=int(os.getenv("SIM_USERS", "200")))
    p.add_argument("--queries-per-user", type=int, default=int(os.getenv("SIM_QPU", "50")))
    p.add_argument("--k", type=int, default=int(os.getenv("SIM_K", "10")))
    p.add_argument("--workers", type=int, default=int(os.getenv("SIM_WORKERS", "16")))
    p.add_argument("--seed", type=int, default=int(os.getenv("SIM_SEED", "42")))
    p.add_argument("--ranker", default=os.getenv("SIM_RANKER", "hybrid"))
    p.add_argument(
        "--base-url",
        default=os.getenv("SIM_BASE_URL", "http://api:8080"),
    )
    args = p.parse_args()

    cat = _load_catalog()
    if not cat.queries:
        log.error("no queries available; run pipelines.ingest.esci first")
        return 1

    log.info(
        "starting simulator users=%d qpu=%d k=%d workers=%d base=%s",
        args.users, args.queries_per_user, args.k, args.workers, args.base_url,
    )

    started = time.perf_counter()
    base_seed = args.seed
    stats: list[UserStats] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                _simulate_user,
                i,
                random.Random(base_seed + i),
                cat,
                args.base_url,
                args.queries_per_user,
                args.k,
                args.ranker,
            )
            for i in range(args.users)
        ]
        for n, f in enumerate(as_completed(futs), 1):
            stats.append(f.result())
            if n % 25 == 0 or n == args.users:
                log.info("progress users=%d/%d %s",
                         n, args.users,
                         _summarise(stats))
    log.info("done elapsed=%.1fs %s", time.perf_counter() - started, _summarise(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
