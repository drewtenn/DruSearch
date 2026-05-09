"""Generated from libs/schema/feature_schema.json by libs/schema/codegen.py.
DO NOT EDIT BY HAND. Run `make check-feature-parity` after schema changes."""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "v8"

@dataclass(frozen=True)
class GeneratedFeature:
    index: int
    name: str
    kind: str
    source: str
    description: str

FEATURES: tuple[GeneratedFeature, ...] = (
    GeneratedFeature(index=0, name="bm25_score", kind="FLOAT", source="RETRIEVAL", description="BM25 raw score"),
    GeneratedFeature(index=1, name="bm25_rank", kind="INT", source="RETRIEVAL", description="BM25 rank within query; absent encoded as max observed rank + 1"),
    GeneratedFeature(index=2, name="title_bm25_score", kind="FLOAT", source="RETRIEVAL", description="Named BM25 contribution from the title field"),
    GeneratedFeature(index=3, name="category_path_bm25_score", kind="FLOAT", source="RETRIEVAL", description="Named BM25 contribution from the category path field"),
    GeneratedFeature(index=4, name="category_bm25_score", kind="FLOAT", source="RETRIEVAL", description="Named BM25 contribution from the leaf category field"),
    GeneratedFeature(index=5, name="bullets_bm25_score", kind="FLOAT", source="RETRIEVAL", description="Named BM25 contribution from the bullet text field"),
    GeneratedFeature(index=6, name="description_bm25_score", kind="FLOAT", source="RETRIEVAL", description="Named BM25 contribution from the description field"),
    GeneratedFeature(index=7, name="brand_bm25_score", kind="FLOAT", source="RETRIEVAL", description="Named fuzzy BM25 contribution from the brand field"),
    GeneratedFeature(index=8, name="knn_score", kind="FLOAT", source="RETRIEVAL", description="k-NN cosine similarity"),
    GeneratedFeature(index=9, name="knn_rank", kind="INT", source="RETRIEVAL", description="k-NN rank within query; absent encoded as max observed rank + 1"),
    GeneratedFeature(index=10, name="rrf_score", kind="FLOAT", source="RETRIEVAL", description="RRF fused retrieval score"),
    GeneratedFeature(index=11, name="popularity_prior", kind="FLOAT", source="STATIC_PRODUCT", description="Catalog popularity prior in [0,1]"),
    GeneratedFeature(index=12, name="price_log_cents", kind="FLOAT", source="STATIC_PRODUCT", description="log1p(price_cents)"),
    GeneratedFeature(index=13, name="title_length_tokens", kind="FLOAT", source="STATIC_PRODUCT", description="Token count of title"),
    GeneratedFeature(index=14, name="query_length_tokens", kind="FLOAT", source="INTERACTION", description="Token count of query"),
    GeneratedFeature(index=15, name="query_has_brand", kind="BOOL", source="INTERACTION", description="Query contains a known brand token"),
    GeneratedFeature(index=16, name="query_has_color", kind="BOOL", source="INTERACTION", description="Query contains a known color token"),
    GeneratedFeature(index=17, name="query_has_category_token", kind="BOOL", source="INTERACTION", description="Query contains a known category token"),
    GeneratedFeature(index=18, name="query_has_size_pattern", kind="BOOL", source="INTERACTION", description="Query contains size/unit pattern"),
    GeneratedFeature(index=19, name="query_gender_intent", kind="INT", source="INTERACTION", description="Requested gender: 0=none, 1=men, 2=women, 3=boys, 4=girls, 5=unisex"),
    GeneratedFeature(index=20, name="product_gender", kind="INT", source="STATIC_PRODUCT", description="Derived product gender from indexed gender, category path, or title"),
    GeneratedFeature(index=21, name="gender_intent_match", kind="FLOAT", source="INTERACTION", description="Query gender match strength; unisex products partially match men and women"),
    GeneratedFeature(index=22, name="gender_intent_mismatch", kind="BOOL", source="INTERACTION", description="Known query gender differs from product gender"),
    GeneratedFeature(index=23, name="product_brand_match", kind="BOOL", source="INTERACTION", description="Query brand token matches product brand"),
    GeneratedFeature(index=24, name="product_brand_token_overlap", kind="FLOAT", source="INTERACTION", description="Fraction of product brand tokens present in query"),
    GeneratedFeature(index=25, name="product_color_match", kind="BOOL", source="INTERACTION", description="Query color token matches product color"),
    GeneratedFeature(index=26, name="title_query_token_coverage", kind="FLOAT", source="INTERACTION", description="Fraction of non-brand query tokens present in title"),
    GeneratedFeature(index=27, name="category_query_token_coverage", kind="FLOAT", source="INTERACTION", description="Fraction of non-brand query tokens present in category path"),
    GeneratedFeature(index=28, name="product_category_token_overlap", kind="FLOAT", source="INTERACTION", description="Fraction of product category tokens present in query"),
    GeneratedFeature(index=29, name="title_exact_query_match", kind="BOOL", source="INTERACTION", description="Normalized full query appears in normalized title"),
    GeneratedFeature(index=30, name="user_brand_affinity", kind="FLOAT", source="ONLINE_USER", description="User brand click-share in [0,1]"),
    GeneratedFeature(index=31, name="query_affordability_intent", kind="BOOL", source="INTERACTION", description="Query asks for affordable, cheap, budget, value, or low-cost products"),
    GeneratedFeature(index=32, name="affordability_price_score", kind="FLOAT", source="INTERACTION", description="Low-price score active only for affordability-intent queries; larger means cheaper"),
    GeneratedFeature(index=33, name="brand_family_match", kind="BOOL", source="INTERACTION", description="Query brand/subbrand intent matches product brand family without broadening to unrelated family products"),
    GeneratedFeature(index=34, name="subbrand_title_match", kind="BOOL", source="INTERACTION", description="Query brand/subbrand alias appears in product title"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
NUM_FEATURES: int = len(FEATURES)

IDX_BM25_SCORE = 0
IDX_BM25_RANK = 1
IDX_TITLE_BM25_SCORE = 2
IDX_CATEGORY_PATH_BM25_SCORE = 3
IDX_CATEGORY_BM25_SCORE = 4
IDX_BULLETS_BM25_SCORE = 5
IDX_DESCRIPTION_BM25_SCORE = 6
IDX_BRAND_BM25_SCORE = 7
IDX_KNN_SCORE = 8
IDX_KNN_RANK = 9
IDX_RRF_SCORE = 10
IDX_POPULARITY_PRIOR = 11
IDX_PRICE_LOG_CENTS = 12
IDX_TITLE_LENGTH_TOKENS = 13
IDX_QUERY_LENGTH_TOKENS = 14
IDX_QUERY_HAS_BRAND = 15
IDX_QUERY_HAS_COLOR = 16
IDX_QUERY_HAS_CATEGORY_TOKEN = 17
IDX_QUERY_HAS_SIZE_PATTERN = 18
IDX_QUERY_GENDER_INTENT = 19
IDX_PRODUCT_GENDER = 20
IDX_GENDER_INTENT_MATCH = 21
IDX_GENDER_INTENT_MISMATCH = 22
IDX_PRODUCT_BRAND_MATCH = 23
IDX_PRODUCT_BRAND_TOKEN_OVERLAP = 24
IDX_PRODUCT_COLOR_MATCH = 25
IDX_TITLE_QUERY_TOKEN_COVERAGE = 26
IDX_CATEGORY_QUERY_TOKEN_COVERAGE = 27
IDX_PRODUCT_CATEGORY_TOKEN_OVERLAP = 28
IDX_TITLE_EXACT_QUERY_MATCH = 29
IDX_USER_BRAND_AFFINITY = 30
IDX_QUERY_AFFORDABILITY_INTENT = 31
IDX_AFFORDABILITY_PRICE_SCORE = 32
IDX_BRAND_FAMILY_MATCH = 33
IDX_SUBBRAND_TITLE_MATCH = 34
