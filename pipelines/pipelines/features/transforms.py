"""Reference Python implementations for INTERACTION features.

Specced in libs/schema/transforms.md. Phase 5 mirrors these in Go and a
parity test runs both impls on the same fixtures.
"""

from __future__ import annotations

import re
import math

# Tokenize on \w+ (Unicode word characters), lowercase. Matches Go regexp default.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Size-with-unit pattern: 12oz, 1.5L, 16gb, 7in, 9.5 mm, etc.
_SIZE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:oz|ml|gb|tb|in|cm|mm|kg|lb|l|g)\b",
    re.IGNORECASE,
)

GENDER_NONE = 0.0
GENDER_MEN = 1.0
GENDER_WOMEN = 2.0
GENDER_BOYS = 3.0
GENDER_GIRLS = 4.0

_QUERY_GENDER_TOKENS = {
    "men": GENDER_MEN,
    "mens": GENDER_MEN,
    "man": GENDER_MEN,
    "male": GENDER_MEN,
    "women": GENDER_WOMEN,
    "womens": GENDER_WOMEN,
    "woman": GENDER_WOMEN,
    "female": GENDER_WOMEN,
    "boys": GENDER_BOYS,
    "boy": GENDER_BOYS,
    "girls": GENDER_GIRLS,
    "girl": GENDER_GIRLS,
}

_CATEGORY_GENDER_VALUES = {
    "men": GENDER_MEN,
    "women": GENDER_WOMEN,
    "boys": GENDER_BOYS,
    "girls": GENDER_GIRLS,
}


def tokenize(s: str | None) -> list[str]:
    if not s:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(s)]


def query_length_tokens(query: str | None) -> float:
    return float(len(tokenize(query)))


def query_has_brand(query: str | None, known_brands_lower: frozenset[str]) -> float:
    return 1.0 if (set(tokenize(query)) & known_brands_lower) else 0.0


def query_has_color(query: str | None, known_colors_lower: frozenset[str]) -> float:
    return 1.0 if (set(tokenize(query)) & known_colors_lower) else 0.0


def query_has_category_token(query: str | None, known_category_tokens: frozenset[str]) -> float:
    return 1.0 if (set(tokenize(query)) & known_category_tokens) else 0.0


def query_has_size_pattern(query: str | None) -> float:
    return 1.0 if (query and _SIZE_RE.search(query)) else 0.0


_AFFORDABILITY_TOKENS = {
    "affordable",
    "affordability",
    "cheap",
    "cheaper",
    "cheapest",
    "budget",
    "inexpensive",
    "economical",
    "value",
    "price",
    "priced",
    "pricing",
    "cost",
    "costs",
}
_AFFORDABILITY_PREFIXES = {"low", "lower", "lowest"}
_AFFORDABILITY_NOUNS = {"cost", "price"}


def query_affordability_intent(query: str | None) -> float:
    tokens = tokenize(query)
    for i, token in enumerate(tokens):
        if token in _AFFORDABILITY_TOKENS:
            return 1.0
        if i and tokens[i - 1] in _AFFORDABILITY_PREFIXES and token in _AFFORDABILITY_NOUNS:
            return 1.0
    return 0.0


def affordability_price_score(query_affordability: float, price_cents: int | float | None) -> float:
    price = float(price_cents or 0)
    if not query_affordability or price <= 0:
        return 0.0
    return float(1 / math.log1p(price))


def query_gender_intent(query: str | None) -> float:
    found = {g for t in tokenize(query) if (g := _QUERY_GENDER_TOKENS.get(t))}
    return found.pop() if len(found) == 1 else GENDER_NONE


def product_gender(category_path: list[str] | tuple[str, ...] | None) -> float:
    if not category_path:
        return GENDER_NONE
    for part in category_path:
        gender = _CATEGORY_GENDER_VALUES.get(str(part).strip().lower())
        if gender:
            return gender
    return GENDER_NONE


def gender_intent_match(query_gender: float, product_gender_value: float) -> float:
    return 1.0 if query_gender and query_gender == product_gender_value else 0.0


def gender_intent_mismatch(query_gender: float, product_gender_value: float) -> float:
    return 1.0 if query_gender and product_gender_value and query_gender != product_gender_value else 0.0


def product_brand_match(query: str | None, brand: str | None) -> float:
    query_tokens = set(tokenize(query))
    brand_tokens = set(tokenize(brand))
    return 1.0 if query_tokens and brand_tokens and bool(query_tokens & brand_tokens) else 0.0


def product_brand_token_overlap(query: str | None, brand: str | None) -> float:
    query_tokens = set(tokenize(query))
    brand_tokens = set(tokenize(brand))
    if not brand_tokens:
        return 0.0
    return float(len(query_tokens & brand_tokens) / len(brand_tokens))


def product_color_match(query: str | None, color: str | None) -> float:
    return product_brand_match(query, color)


def query_token_coverage(
    query: str | None,
    text: str | None,
    ignored_tokens: frozenset[str] = frozenset(),
) -> float:
    query_tokens = tokenize(query)
    q = [t for t in query_tokens if t not in ignored_tokens]
    if not q:
        q = query_tokens
    if not q:
        return 0.0
    text_tokens = set(tokenize(text))
    return float(sum(1 for t in q if t in text_tokens) / len(q))


def exact_query_phrase_match(query: str | None, text: str | None) -> float:
    q = " ".join(tokenize(query))
    haystack = " ".join(tokenize(text))
    return 1.0 if q and q in haystack else 0.0


def token_overlap_fraction(query: str | None, text: str | None) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    return float(len(query_tokens & text_tokens) / len(text_tokens))


def product_category_token_overlap(query: str | None, category_text: str | None) -> float:
    return token_overlap_fraction(query, category_text)
