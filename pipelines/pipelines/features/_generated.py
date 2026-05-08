"""Generated from libs/schema/feature_schema.json by libs/schema/codegen.py.
DO NOT EDIT BY HAND. Run `make check-feature-parity` after schema changes."""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "v2"

@dataclass(frozen=True)
class GeneratedFeature:
    index: int
    name: str
    kind: str
    source: str
    description: str

FEATURES: tuple[GeneratedFeature, ...] = (
    GeneratedFeature(index=0, name="bm25_score", kind="FLOAT", source="RETRIEVAL", description="BM25 raw score"),
    GeneratedFeature(index=1, name="bm25_rank", kind="INT", source="RETRIEVAL", description="BM25 rank within query (1-indexed); 0 if not in BM25 set"),
    GeneratedFeature(index=2, name="knn_score", kind="FLOAT", source="RETRIEVAL", description="k-NN cosine similarity"),
    GeneratedFeature(index=3, name="knn_rank", kind="INT", source="RETRIEVAL", description="k-NN rank within query (1-indexed); 0 if not in k-NN set"),
    GeneratedFeature(index=4, name="rrf_score", kind="FLOAT", source="RETRIEVAL", description="RRF fused score"),
    GeneratedFeature(index=5, name="popularity_prior", kind="FLOAT", source="STATIC_PRODUCT", description="Pre-existing popularity prior in [0,1]"),
    GeneratedFeature(index=6, name="price_log_cents", kind="FLOAT", source="STATIC_PRODUCT", description="log1p(price_cents)"),
    GeneratedFeature(index=7, name="title_length_tokens", kind="FLOAT", source="STATIC_PRODUCT", description="Token count of title"),
    GeneratedFeature(index=8, name="query_length_tokens", kind="FLOAT", source="INTERACTION", description="Token count of query"),
    GeneratedFeature(index=9, name="query_has_brand", kind="BOOL", source="INTERACTION", description="Query contains a known brand token (0/1)"),
    GeneratedFeature(index=10, name="query_has_color", kind="BOOL", source="INTERACTION", description="Query contains a known color token (0/1)"),
    GeneratedFeature(index=11, name="query_has_size_pattern", kind="BOOL", source="INTERACTION", description="Query matches `\\b\\d+(?:\\.\\d+)?\\s?(oz|ml|gb|tb|in|cm|mm|kg|lb|l|g)\\b` (0/1)"),
    GeneratedFeature(index=12, name="user_brand_affinity", kind="FLOAT", source="ONLINE_USER", description="User's click-share for the candidate's brand in [0,1]; 0 if unknown user"),
    GeneratedFeature(index=13, name="query_gender_intent", kind="INT", source="INTERACTION", description="Requested gender inferred from query: 0=none, 1=men, 2=women, 3=boys, 4=girls"),
    GeneratedFeature(index=14, name="product_gender", kind="INT", source="STATIC_PRODUCT", description="Product gender inferred from category path: 0=none, 1=men, 2=women, 3=boys, 4=girls"),
    GeneratedFeature(index=15, name="gender_intent_match", kind="BOOL", source="INTERACTION", description="1 when query_gender_intent is known and equals product_gender"),
    GeneratedFeature(index=16, name="gender_intent_mismatch", kind="BOOL", source="INTERACTION", description="1 when query_gender_intent and product_gender are known but differ"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
NUM_FEATURES: int = len(FEATURES)

IDX_BM25_SCORE = 0
IDX_BM25_RANK = 1
IDX_KNN_SCORE = 2
IDX_KNN_RANK = 3
IDX_RRF_SCORE = 4
IDX_POPULARITY_PRIOR = 5
IDX_PRICE_LOG_CENTS = 6
IDX_TITLE_LENGTH_TOKENS = 7
IDX_QUERY_LENGTH_TOKENS = 8
IDX_QUERY_HAS_BRAND = 9
IDX_QUERY_HAS_COLOR = 10
IDX_QUERY_HAS_SIZE_PATTERN = 11
IDX_USER_BRAND_AFFINITY = 12
IDX_QUERY_GENDER_INTENT = 13
IDX_PRODUCT_GENDER = 14
IDX_GENDER_INTENT_MATCH = 15
IDX_GENDER_INTENT_MISMATCH = 16
