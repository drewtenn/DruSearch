"""Feature schema for the LTR ranker.

Phase 4 ships a Python-only feature implementation. The list below is the
single source of truth for *both* training-row construction and (eventually)
Go inference. When Phase 5 adds Go serving, the same names are codegen'd
into Go accessors so the feature vector indices line up at compile time.

Sources:
  RETRIEVAL       — from the impression's retrieval_scores blob
  STATIC_PRODUCT  — from products / OpenSearch _source
  PRODUCT_AGG     — from product_features (Beta-smoothed CTR, etc.)
  INTERACTION     — pure function of (query, product, ...); see transforms.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Source(str, Enum):
    RETRIEVAL = "retrieval"
    STATIC_PRODUCT = "static_product"
    PRODUCT_AGG = "product_agg"
    INTERACTION = "interaction"
    ONLINE_USER = "online_user"


@dataclass(frozen=True)
class FeatureDef:
    name: str
    source: Source
    description: str


# Ordered feature list. v5 intentionally replaces the previous schema; rebuild
# the API, training rows, and model before serving these indices.
FEATURES: tuple[FeatureDef, ...] = (
    FeatureDef("bm25_score", Source.RETRIEVAL, "BM25 raw score"),
    FeatureDef("bm25_rank", Source.RETRIEVAL, "BM25 rank within query; absent encoded as max observed rank + 1"),
    FeatureDef("title_bm25_score", Source.RETRIEVAL, "Named BM25 contribution from the title field"),
    FeatureDef("category_path_bm25_score", Source.RETRIEVAL, "Named BM25 contribution from the category path field"),
    FeatureDef("category_bm25_score", Source.RETRIEVAL, "Named BM25 contribution from the leaf category field"),
    FeatureDef("bullets_bm25_score", Source.RETRIEVAL, "Named BM25 contribution from the bullet text field"),
    FeatureDef("description_bm25_score", Source.RETRIEVAL, "Named BM25 contribution from the description field"),
    FeatureDef("brand_bm25_score", Source.RETRIEVAL, "Named fuzzy BM25 contribution from the brand field"),
    FeatureDef("knn_score", Source.RETRIEVAL, "k-NN cosine similarity"),
    FeatureDef("knn_rank", Source.RETRIEVAL, "k-NN rank within query; absent encoded as max observed rank + 1"),
    FeatureDef("rrf_score", Source.RETRIEVAL, "RRF fused retrieval score"),
    FeatureDef("popularity_prior", Source.STATIC_PRODUCT, "Catalog popularity prior in [0,1]"),
    FeatureDef("price_log_cents", Source.STATIC_PRODUCT, "log1p(price_cents)"),
    FeatureDef("title_length_tokens", Source.STATIC_PRODUCT, "Token count of title"),
    FeatureDef("query_length_tokens", Source.INTERACTION, "Token count of query"),
    FeatureDef("query_has_brand", Source.INTERACTION, "Query contains a known brand token"),
    FeatureDef("query_has_color", Source.INTERACTION, "Query contains a known color token"),
    FeatureDef("query_has_category_token", Source.INTERACTION, "Query contains a known category token"),
    FeatureDef("query_has_size_pattern", Source.INTERACTION, "Query contains size/unit pattern"),
    FeatureDef("query_gender_intent", Source.INTERACTION, "Requested gender: 0=none, 1=men, 2=women, 3=boys, 4=girls, 5=unisex"),
    FeatureDef("product_gender", Source.STATIC_PRODUCT, "Derived product gender from indexed gender, category path, or title"),
    FeatureDef("gender_intent_match", Source.INTERACTION, "Query gender match strength; unisex products partially match men and women"),
    FeatureDef("gender_intent_mismatch", Source.INTERACTION, "Known query gender differs from product gender"),
    FeatureDef("product_brand_match", Source.INTERACTION, "Query brand token matches product brand"),
    FeatureDef("product_brand_token_overlap", Source.INTERACTION, "Fraction of product brand tokens present in query"),
    FeatureDef("product_color_match", Source.INTERACTION, "Query color token matches product color"),
    FeatureDef("title_query_token_coverage", Source.INTERACTION, "Fraction of non-brand query tokens present in title"),
    FeatureDef("category_query_token_coverage", Source.INTERACTION, "Fraction of non-brand query tokens present in category path"),
    FeatureDef("product_category_token_overlap", Source.INTERACTION, "Fraction of product category tokens present in query"),
    FeatureDef("title_exact_query_match", Source.INTERACTION, "Normalized full query appears in normalized title"),
    FeatureDef("user_brand_affinity", Source.ONLINE_USER, "User brand click-share in [0,1]"),
    FeatureDef("query_affordability_intent", Source.INTERACTION, "Query asks for affordable, cheap, budget, value, or low-cost products"),
    FeatureDef("affordability_price_score", Source.INTERACTION, "Low-price score active only for affordability-intent queries; larger means cheaper"),
    FeatureDef("brand_family_match", Source.INTERACTION, "Query brand/subbrand intent matches product brand family without broadening to unrelated family products"),
    FeatureDef("subbrand_title_match", Source.INTERACTION, "Query brand/subbrand alias appears in product title"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
NUM_FEATURES: int = len(FEATURES)

# Drift guard: every import compares this hand-written list against the file
# generated by libs/schema/codegen.py from feature_schema.json (the canonical
# source of truth). If you see this assertion fail, run `make
# check-feature-parity` and reconcile.
from . import _generated as _gen  # noqa: E402

assert FEATURE_NAMES == _gen.FEATURE_NAMES, (
    f"feature schema drift: hand-written {FEATURE_NAMES} vs generated {_gen.FEATURE_NAMES}"
)
assert NUM_FEATURES == _gen.NUM_FEATURES
