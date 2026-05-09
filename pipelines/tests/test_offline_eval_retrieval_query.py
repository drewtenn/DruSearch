from pipelines.evaluate.offline_eval import _bm25_query


def test_bm25_query_uses_derived_gender_for_gendered_queries():
    query = _bm25_query("nike mens shoes")

    boosting = query["boosting"]
    should = boosting["positive"]["bool"]["should"]

    assert len(should) == 2
    assert boosting["negative_boost"] == 0.25
    assert boosting["negative"]["terms"]["derived_gender"] == ["women", "boys", "girls"]


def test_bm25_query_keeps_plain_query_without_gender_intent():
    query = _bm25_query("nike running shoes")

    assert "multi_match" in query
