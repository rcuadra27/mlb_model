"""Unit tests for player name matching and prop lookup helpers."""

from main import (
    _name_match_score,
    _normalize_name_for_match,
    _normalize_prop_type,
    _parse_prop_detail_baseline,
    _resolve_batter_matches,
    tool_get_player_prop,
)


def test_normalize_accent():
    assert _normalize_name_for_match("José Ramírez") == "jose ramirez"
    assert _name_match_score("Jose Ramirez", "José Ramírez") == 100


def test_name_match_ryan_ward():
    assert _name_match_score("Ryan Ward", "Ryan Ward") == 100
    assert _name_match_score("ward", "Ryan Ward") == 95
    assert _name_match_score("ryan ward", "Ryan Ward") == 100


def test_resolve_batter_partial():
    batters = [
        {"batter_name": "Ryan Ward", "batter_id": 1},
        {"batter_name": "Taylor Ward", "batter_id": 2},
    ]
    m = _resolve_batter_matches("Ryan Ward", batters)
    assert len(m) == 1
    assert m[0]["batter_name"] == "Ryan Ward"


def test_parse_detail_baseline():
    pct, kind = _parse_prop_detail_baseline("Model 48.7% vs league avg 30.0%")
    assert pct == 30.0
    assert kind == "league"


def test_walk_prop_type():
    assert _normalize_prop_type("walk") == "walk"
    assert _normalize_prop_type("1+ Walk") == "1plus_walk"  # may not map - tool uses walk


if __name__ == "__main__":
    test_normalize_accent()
    test_name_match_ryan_ward()
    test_resolve_batter_partial()
    test_parse_detail_baseline()
    test_walk_prop_type()
    print("local ok")
    out = tool_get_player_prop("Ryan Ward", "walk", "2026-06-02")
    assert out.get("found") is True, out
    assert out.get("focused_prop", {}).get("model_probability_pct") is not None
    assert out.get("top_edge", {}).get("rank") == 1
    print("bq ok", out.get("reasoning_summary"))
