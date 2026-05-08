"""Reference Python implementations for INTERACTION features.

Specced in libs/schema/transforms.md. Phase 5 mirrors these in Go and a
parity test runs both impls on the same fixtures.
"""

from __future__ import annotations

import re

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


def query_has_size_pattern(query: str | None) -> float:
    return 1.0 if (query and _SIZE_RE.search(query)) else 0.0


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
