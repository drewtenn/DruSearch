"""Cross-language interaction-feature parity tests (Python side).

Reads libs/schema/fixtures/interaction_fixtures.json and runs the Python
reference transforms against it. The Go side
(services/api-go/internal/features/parity_test.go) reads the same file and
runs the same assertions; together they form the cross-language byte-equal
guarantee the architecture doc promises.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.features.transforms import (
    gender_intent_match,
    gender_intent_mismatch,
    product_gender,
    query_has_brand,
    query_has_color,
    query_gender_intent,
    query_has_size_pattern,
    query_length_tokens,
    tokenize,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "libs" / "schema" / "fixtures" / "interaction_fixtures.json"


def _load():
    with FIXTURES.open() as f:
        return json.load(f)


def test_fixtures_loadable():
    data = _load()
    assert data["cases"], "expected at least one fixture case"
    assert data["vocab"]["brands"]
    assert data["vocab"]["colors"]


def _vocab():
    data = _load()
    return (
        frozenset(b.lower() for b in data["vocab"]["brands"]),
        frozenset(c.lower() for c in data["vocab"]["colors"]),
    )


def test_interaction_parity_fixtures():
    data = _load()
    brands, colors = _vocab()
    failures: list[str] = []
    for case in data["cases"]:
        q = case["query"]
        exp = case["expected"]

        got_tok = tokenize(q)
        if got_tok != exp["tokenize"]:
            failures.append(f"{case['name']}: tokenize({q!r}) = {got_tok!r}, want {exp['tokenize']!r}")

        got_qlt = query_length_tokens(q)
        if got_qlt != exp["query_length_tokens"]:
            failures.append(f"{case['name']}: query_length_tokens({q!r}) = {got_qlt}, want {exp['query_length_tokens']}")

        got_qhb = query_has_brand(q, brands)
        if got_qhb != exp["query_has_brand"]:
            failures.append(f"{case['name']}: query_has_brand({q!r}) = {got_qhb}, want {exp['query_has_brand']}")

        got_qhc = query_has_color(q, colors)
        if got_qhc != exp["query_has_color"]:
            failures.append(f"{case['name']}: query_has_color({q!r}) = {got_qhc}, want {exp['query_has_color']}")

        got_qhs = query_has_size_pattern(q)
        if got_qhs != exp["query_has_size_pattern"]:
            failures.append(f"{case['name']}: query_has_size_pattern({q!r}) = {got_qhs}, want {exp['query_has_size_pattern']}")

        category_path = case.get("category_path", [])
        got_qgi = query_gender_intent(q)
        if got_qgi != exp["query_gender_intent"]:
            failures.append(f"{case['name']}: query_gender_intent({q!r}) = {got_qgi}, want {exp['query_gender_intent']}")

        got_pg = product_gender(category_path)
        if got_pg != exp["product_gender"]:
            failures.append(f"{case['name']}: product_gender({category_path!r}) = {got_pg}, want {exp['product_gender']}")

        got_gim = gender_intent_match(got_qgi, got_pg)
        if got_gim != exp["gender_intent_match"]:
            failures.append(f"{case['name']}: gender_intent_match({got_qgi}, {got_pg}) = {got_gim}, want {exp['gender_intent_match']}")

        got_gix = gender_intent_mismatch(got_qgi, got_pg)
        if got_gix != exp["gender_intent_mismatch"]:
            failures.append(f"{case['name']}: gender_intent_mismatch({got_qgi}, {got_pg}) = {got_gix}, want {exp['gender_intent_mismatch']}")

    assert not failures, "interaction-feature parity drift:\n  " + "\n  ".join(failures)


def test_schema_matches_generated():
    """Hand-written FEATURES tuple must match the JSON-driven generator output."""
    from pipelines.features import FEATURE_NAMES, NUM_FEATURES
    from pipelines.features import _generated as gen

    assert FEATURE_NAMES == gen.FEATURE_NAMES, (
        f"FEATURE_NAMES drift: {FEATURE_NAMES} vs {gen.FEATURE_NAMES}"
    )
    assert NUM_FEATURES == gen.NUM_FEATURES
    assert gen.SCHEMA_VERSION == "v2"
