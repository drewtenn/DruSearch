from __future__ import annotations

from pipelines.simulate import click_simulator


def test_search_params_pin_simulator_to_requested_ranker():
    params = click_simulator._search_params(
        query="running shoes",
        user_id="u1",
        session_id="s1",
        k=10,
        ranker="hybrid",
    )

    assert params == {
        "q": "running shoes",
        "user_id": "u1",
        "session_id": "s1",
        "k": 10,
        "ranker": "hybrid",
    }
