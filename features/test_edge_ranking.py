"""Smoke tests for cross-edge ranking (posted-line pitcher props only)."""

from features.edge_logic import (
    _ranking_sort_key,
    _season_count_for_ip,
    compute_all_edges,
    er_edges_from_pitchers,
    k_edges_from_pitchers,
    select_top_edges,
)


def _k_pitcher(pid: int, lam: float, ip: float = 5.5, name: str = "Ace", market_k: float | None = 5.5) -> dict:
    p = {
        "pitcher_id": pid,
        "pitcher_name": name,
        "game_id": 1,
        "lambda_k": lam,
        "expected_ip": ip,
        "team_name": "Test",
    }
    if market_k is not None:
        p["_market_lines"] = {"k": market_k}
    return p


def _er_pitcher(pid: int, lam: float, ip: float, name: str = "SP", market_er: float | None = 2.5) -> dict:
    p = {
        "pitcher_id": pid,
        "pitcher_name": name,
        "game_id": pid,
        "lambda_er": lam,
        "expected_ip": ip,
        "team_name": "Test",
    }
    if market_er is not None:
        p["_market_lines"] = {"er": market_er}
    return p


def test_k_edges_require_posted_line_and_show_over_format():
    pitchers = [_k_pitcher(1, 7.0, ip=5.5, market_k=5.5)]
    edges = k_edges_from_pitchers(pitchers, {1: 9.0})
    assert len(edges) == 1
    e = edges[0]
    assert e["pick_description"] == "Ace — Over 5.5 K"
    assert e["detail_line"].startswith("Model:")
    assert e["market_line"] == 5.5
    assert "more than usual" not in e["pick_description"]
    assert "season average" not in (e.get("detail_line") or "")


def test_k_edges_skipped_without_posted_line():
    pitchers = [_k_pitcher(1, 7.0, market_k=None)]
    assert k_edges_from_pitchers(pitchers, {1: 9.0}) == []


def test_er_edges_require_posted_line():
    pitchers = [_er_pitcher(1, 2.4, 6.0, market_er=None)]
    assert er_edges_from_pitchers(pitchers, {1: 3.0}) == []


def test_er_skips_low_posted_market_line():
    pitchers = [{
        "pitcher_id": 1,
        "pitcher_name": "Ace",
        "game_id": 1,
        "lambda_er": 2.5,
        "expected_ip": 6.0,
        "_market_lines": {"er": 1.5},
    }]
    assert er_edges_from_pitchers(pitchers, {1: 3.0}) == []


def test_k_tier_below_batter_prop():
    games = [{
        "game_id": 1,
        "away_team": "Away",
        "home_team": "Home",
        "edge_home": 0.12,
        "edge_away": 0.02,
        "p_win_home": 0.62,
        "p_home_market": 0.50,
        "ou_edge_over": 1.2,
        "ou_line": 8.5,
        "total_runs_pred": 9.8,
    }]
    batters = [
        {"batter_id": 10, "batter_name": "Slugger", "game_id": 1, "p_hr": 0.28, "team_name": "Test"},
        {"batter_id": 11, "batter_name": "Contact", "game_id": 1, "p_hit": 0.72, "team_name": "Test"},
        {"batter_id": 12, "batter_name": "Walker", "game_id": 1, "p_walk": 0.42, "team_name": "Test"},
    ]
    pitchers = [_k_pitcher(i, 7.0 + i * 0.2, market_k=4.5) for i in range(1, 8)]
    top = compute_all_edges(games, batters, pitchers, {}, {"k": {i: 9.0 for i in range(1, 8)}})
    top6_types = [e["edge_type"] for e in top[:6]]
    assert top6_types[0] in ("ml", "total", "prop")
    assert top6_types.count("k") <= 1


def test_ml_and_totals_rank_above_weak_pitcher_props():
    games = [
        {
            "game_id": 1,
            "away_team": "TOR",
            "home_team": "ATL",
            "edge_home": 0.102,
            "edge_away": -0.102,
            "p_win_home": 0.62,
            "p_home_market": 0.52,
            "ou_edge_over": 1.78,
            "ou_line": 8.5,
            "total_runs_pred": 10.3,
        },
    ]
    pitchers = [_er_pitcher(i, 2.3, 6.0, f"SP{i}", market_er=2.5) for i in range(1, 12)]
    top = compute_all_edges(games, [], pitchers, {}, {"er": {i: 1.5 for i in range(1, 12)}})
    top5_types = [e["edge_type"] for e in top[:5]]
    assert top5_types[0] in ("ml", "total")


def test_ranking_sort_key_k_with_market_is_tier_one_when_strong():
    item = {
        "edge_type": "k",
        "market_line": 5.5,
        "model_prob_pct": 62.0,
        "sort_score": 50.0,
    }
    assert _ranking_sort_key(item)[0] == 1


def test_ranking_sort_key_k_weak_prob_is_tier_two():
    item = {
        "edge_type": "k",
        "market_line": 5.5,
        "model_prob_pct": 52.0,
        "sort_score": 10.0,
    }
    assert _ranking_sort_key(item)[0] == 2


def test_season_count_for_ip_helper():
    assert abs(_season_count_for_ip(3.6, 5.5) - (3.6 * 5.5 / 9.0)) < 1e-9


def test_nan_game_edges_do_not_produce_null_edge_magnitude():
    games = [{
        "game_id": 1,
        "away_team": "Away",
        "home_team": "Home",
        "edge_home": float("nan"),
        "edge_away": 0.02,
        "p_win_home": 0.62,
        "p_home_market": 0.50,
        "ou_edge_over": float("nan"),
        "ou_edge_under": 1.0,
        "ou_line": 8.5,
        "total_runs_pred": 9.8,
    }]
    top = compute_all_edges(games, [], [], {})
    for e in top:
        mag = e.get("edge_magnitude")
        assert mag is not None, e
        assert mag == mag, e


def test_no_pitcher_prop_uses_norm_phrasing():
    games = [{
        "game_id": 1,
        "away_team": "A",
        "home_team": "B",
        "ou_edge_over": 1.0,
        "ou_line": 8.0,
        "total_runs_pred": 9.0,
    }]
    pitchers = [
        _k_pitcher(1, 7.0, market_k=5.5),
        {
            "pitcher_id": 2,
            "pitcher_name": "SP2",
            "game_id": 1,
            "lambda_hits": 5.5,
            "expected_ip": 6.0,
            "_market_lines": {"hits": 4.5},
        },
    ]
    top = compute_all_edges(games, [], pitchers, {}, {"k": {1: 9.0}, "hits": {2: 8.0}})
    for e in top:
        if e["edge_type"] not in ("k", "walks", "hits", "er"):
            continue
        blob = " ".join(
            str(e.get(k) or "")
            for k in ("pick_description", "detail_line", "rate_detail_line")
        ).lower()
        assert "more than usual" not in blob
        assert "fewer than usual" not in blob
        assert "season average" not in blob
        assert e.get("market_line") is not None


if __name__ == "__main__":
    test_k_edges_require_posted_line_and_show_over_format()
    test_k_edges_skipped_without_posted_line()
    test_er_edges_require_posted_line()
    test_er_skips_low_posted_market_line()
    test_k_tier_below_batter_prop()
    test_ml_and_totals_rank_above_weak_pitcher_props()
    test_ranking_sort_key_k_with_market_is_tier_one_when_strong()
    test_ranking_sort_key_k_weak_prob_is_tier_two()
    test_season_count_for_ip_helper()
    test_nan_game_edges_do_not_produce_null_edge_magnitude()
    test_no_pitcher_prop_uses_norm_phrasing()
    print("ok")
