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
GENDER_UNISEX = 5.0

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
    "unisex": GENDER_UNISEX,
}

_CATEGORY_GENDER_VALUES = {
    "men": GENDER_MEN,
    "women": GENDER_WOMEN,
    "boys": GENDER_BOYS,
    "girls": GENDER_GIRLS,
    "unisex": GENDER_UNISEX,
}

_BRAND_STOP_TOKENS = {
    "accessories",
    "accessory",
    "active",
    "athletic",
    "basketball",
    "boy",
    "boys",
    "clothing",
    "fashion",
    "girl",
    "girls",
    "jewelry",
    "men",
    "mens",
    "running",
    "shoe",
    "shoes",
    "sneaker",
    "sneakers",
    "sports",
    "team",
    "unisex",
    "watch",
    "watches",
    "woman",
    "women",
    "womens",
}

_SUBBRAND_TO_PARENT_BRANDS = {
    "jordan": frozenset({"nike"}),
}

_SUBBRAND_TITLE_ALIASES = {
    "jordan": (("air", "jordan"), ("jordan",)),
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


def brand_tokens(text: str | None) -> set[str]:
    return {t for t in tokenize(text) if t not in _BRAND_STOP_TOKENS}


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


def product_gender_label(label: str | None) -> float:
    return _CATEGORY_GENDER_VALUES.get(str(label or "").strip().lower(), GENDER_NONE)


def product_gender(category_path: list[str] | tuple[str, ...] | None, title: str | None = None) -> float:
    if not category_path:
        return _title_gender(title)
    for part in category_path:
        gender = product_gender_label(part)
        if gender:
            return gender
    return _title_gender(title)


def gender_intent_match(query_gender: float, product_gender_value: float) -> float:
    if query_gender and query_gender == product_gender_value:
        return 1.0
    if query_gender in {GENDER_MEN, GENDER_WOMEN} and product_gender_value == GENDER_UNISEX:
        return 0.5
    return 0.0


def gender_intent_mismatch(query_gender: float, product_gender_value: float) -> float:
    if product_gender_value == GENDER_UNISEX:
        return 0.0
    return 1.0 if query_gender and product_gender_value and query_gender != product_gender_value else 0.0


def product_brand_match(query: str | None, brand: str | None) -> float:
    query_tokens = brand_tokens(query)
    product_tokens = brand_tokens(brand)
    return 1.0 if query_tokens and product_tokens and bool(query_tokens & product_tokens) else 0.0


def product_brand_token_overlap(query: str | None, brand: str | None) -> float:
    query_tokens = brand_tokens(query)
    product_tokens = brand_tokens(brand)
    if not product_tokens:
        return 0.0
    return float(len(query_tokens & product_tokens) / len(product_tokens))


def _contains_token_sequence(tokens: list[str], sequence: tuple[str, ...]) -> bool:
    if not sequence or len(sequence) > len(tokens):
        return False
    width = len(sequence)
    return any(tuple(tokens[i : i + width]) == sequence for i in range(len(tokens) - width + 1))


def subbrand_title_match(query: str | None, title: str | None) -> float:
    query_tokens = set(brand_tokens(query))
    if not query_tokens:
        return 0.0
    title_tokens = tokenize(title)
    for subbrand, aliases in _SUBBRAND_TITLE_ALIASES.items():
        if subbrand not in query_tokens:
            continue
        if any(_contains_token_sequence(title_tokens, alias) for alias in aliases):
            return 1.0
    return 0.0


def brand_family_match(query: str | None, brand: str | None, title: str | None) -> float:
    query_tokens = set(brand_tokens(query))
    product_brand_tokens = set(brand_tokens(brand))
    if not query_tokens or not product_brand_tokens:
        return 0.0
    if query_tokens & product_brand_tokens:
        return 1.0
    for subbrand in query_tokens:
        parents = _SUBBRAND_TO_PARENT_BRANDS.get(subbrand)
        if parents and product_brand_tokens & parents and subbrand_title_match(query, title):
            return 1.0
    return 0.0


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


def _title_gender(title: str | None) -> float:
    found = GENDER_NONE
    for token in tokenize(title):
        gender = _QUERY_GENDER_TOKENS.get(token, GENDER_NONE)
        if not gender:
            continue
        if gender == GENDER_UNISEX:
            return GENDER_UNISEX
        if found and found != gender:
            if {found, gender} == {GENDER_MEN, GENDER_WOMEN}:
                return GENDER_UNISEX
            return GENDER_NONE
        found = gender
    return found
