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
    brand_family_match,
    exact_query_phrase_match,
    gender_intent_match,
    gender_intent_mismatch,
    product_brand_match,
    product_brand_token_overlap,
    product_color_match,
    product_category_token_overlap,
    product_gender,
    query_affordability_intent,
    query_token_coverage,
    query_has_brand,
    query_has_category_token,
    query_has_color,
    query_gender_intent,
    query_has_size_pattern,
    query_length_tokens,
    subbrand_title_match,
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
    category_tokens = frozenset(
        t
        for case in data["cases"]
        for part in case.get("category_path", [])
        for t in tokenize(part)
    )
    return (
        frozenset(b.lower() for b in data["vocab"]["brands"]),
        frozenset(c.lower() for c in data["vocab"]["colors"]),
        category_tokens,
    )


def test_interaction_parity_fixtures():
    data = _load()
    brands, colors, categories = _vocab()
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

        got_qhcat = query_has_category_token(q, categories)
        if got_qhcat != exp["query_has_category_token"]:
            failures.append(f"{case['name']}: query_has_category_token({q!r}) = {got_qhcat}, want {exp['query_has_category_token']}")

        got_qhs = query_has_size_pattern(q)
        if got_qhs != exp["query_has_size_pattern"]:
            failures.append(f"{case['name']}: query_has_size_pattern({q!r}) = {got_qhs}, want {exp['query_has_size_pattern']}")

        got_qai = query_affordability_intent(q)
        want_qai = exp.get("query_affordability_intent", 0.0)
        if got_qai != want_qai:
            failures.append(f"{case['name']}: query_affordability_intent({q!r}) = {got_qai}, want {want_qai}")

        category_path = case.get("category_path", [])
        got_qgi = query_gender_intent(q)
        if got_qgi != exp["query_gender_intent"]:
            failures.append(f"{case['name']}: query_gender_intent({q!r}) = {got_qgi}, want {exp['query_gender_intent']}")

        product_title = case.get("product_title", "")
        got_pg = product_gender(category_path, product_title)
        if got_pg != exp["product_gender"]:
            failures.append(f"{case['name']}: product_gender({category_path!r}, {product_title!r}) = {got_pg}, want {exp['product_gender']}")

        got_gim = gender_intent_match(got_qgi, got_pg)
        if got_gim != exp["gender_intent_match"]:
            failures.append(f"{case['name']}: gender_intent_match({got_qgi}, {got_pg}) = {got_gim}, want {exp['gender_intent_match']}")

        got_gix = gender_intent_mismatch(got_qgi, got_pg)
        if got_gix != exp["gender_intent_mismatch"]:
            failures.append(f"{case['name']}: gender_intent_mismatch({got_qgi}, {got_pg}) = {got_gix}, want {exp['gender_intent_mismatch']}")

        product_brand = case.get("product_brand", "")
        got_pbm = product_brand_match(q, product_brand)
        if got_pbm != exp["product_brand_match"]:
            failures.append(f"{case['name']}: product_brand_match({q!r}, {product_brand!r}) = {got_pbm}, want {exp['product_brand_match']}")

        got_pbto = product_brand_token_overlap(q, product_brand)
        if got_pbto != exp["product_brand_token_overlap"]:
            failures.append(f"{case['name']}: product_brand_token_overlap({q!r}, {product_brand!r}) = {got_pbto}, want {exp['product_brand_token_overlap']}")

        product_color = case.get("product_color", "")
        got_pcm = product_color_match(q, product_color)
        if got_pcm != exp["product_color_match"]:
            failures.append(f"{case['name']}: product_color_match({q!r}, {product_color!r}) = {got_pcm}, want {exp['product_color_match']}")

        got_tqtc = query_token_coverage(q, product_title, brands)
        if got_tqtc != exp["title_query_token_coverage"]:
            failures.append(f"{case['name']}: title_query_token_coverage({q!r}, {product_title!r}) = {got_tqtc}, want {exp['title_query_token_coverage']}")

        category_text = " ".join(category_path or [])
        got_cqtc = query_token_coverage(q, category_text, brands)
        if got_cqtc != exp["category_query_token_coverage"]:
            failures.append(f"{case['name']}: category_query_token_coverage({q!r}, {category_text!r}) = {got_cqtc}, want {exp['category_query_token_coverage']}")

        got_pcto = product_category_token_overlap(q, category_text)
        if got_pcto != exp["product_category_token_overlap"]:
            failures.append(f"{case['name']}: product_category_token_overlap({q!r}, {category_text!r}) = {got_pcto}, want {exp['product_category_token_overlap']}")

        got_teqm = exact_query_phrase_match(q, product_title)
        if got_teqm != exp["title_exact_query_match"]:
            failures.append(f"{case['name']}: title_exact_query_match({q!r}, {product_title!r}) = {got_teqm}, want {exp['title_exact_query_match']}")

        if "brand_family_match" in exp:
            product_brand = case.get("product_brand", "")
            got_bfm = brand_family_match(q, product_brand, product_title)
            if got_bfm != exp["brand_family_match"]:
                failures.append(f"{case['name']}: brand_family_match({q!r}, {product_brand!r}, {product_title!r}) = {got_bfm}, want {exp['brand_family_match']}")

        if "subbrand_title_match" in exp:
            got_stm = subbrand_title_match(q, product_title)
            if got_stm != exp["subbrand_title_match"]:
                failures.append(f"{case['name']}: subbrand_title_match({q!r}, {product_title!r}) = {got_stm}, want {exp['subbrand_title_match']}")

    assert not failures, "interaction-feature parity drift:\n  " + "\n  ".join(failures)


def test_schema_matches_generated():
    """Hand-written FEATURES tuple must match the JSON-driven generator output."""
    from pipelines.features import FEATURE_NAMES, NUM_FEATURES
    from pipelines.features import _generated as gen

    assert FEATURE_NAMES == gen.FEATURE_NAMES, (
        f"FEATURE_NAMES drift: {FEATURE_NAMES} vs {gen.FEATURE_NAMES}"
    )
    assert NUM_FEATURES == gen.NUM_FEATURES
    assert gen.SCHEMA_VERSION == "v7"


def test_query_token_coverage_keeps_brand_tokens_when_query_is_only_brand():
    brands = frozenset({"jordan"})

    assert query_token_coverage("jordan", "Air Jordan Future", brands) == 1.0
    assert query_token_coverage("jordan", "Anti Crease Shoe Guard", brands) == 0.0


def test_generic_brand_tokens_do_not_count_as_brand_matches():
    assert product_brand_match("mens jordan basketball", "Altra Running Mens") == 0.0
    assert product_brand_token_overlap("mens jordan basketball", "Altra Running Mens") == 0.0
    assert product_brand_match("mens jordan basketball", "Jordan") == 1.0


def test_brand_family_match_requires_subbrand_evidence_for_parent_brand():
    assert brand_family_match("mens jordan basketball", "Nike", "Nike Men's Air Jordan 1 Mid Shoes") == 1.0
    assert brand_family_match("mens jordan basketball", "Nike", "Nike Mens PG 5 Basketball Shoe") == 0.0
    assert brand_family_match("mens jordan basketball", "Jordan", "Air Jordan Future") == 1.0

    assert subbrand_title_match("mens jordan basketball", "Nike Men's Air Jordan 1 Mid Shoes") == 1.0
    assert subbrand_title_match("mens jordan basketball", "Nike Mens PG 5 Basketball Shoe") == 0.0
