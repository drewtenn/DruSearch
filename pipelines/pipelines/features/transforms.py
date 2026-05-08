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
