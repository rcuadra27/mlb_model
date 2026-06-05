"""Tests for strength-based total → home/away run split."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.run_split import split_total


def test_road_favorite_dodgers_at_arizona():
    total = 9.16
    home, away = split_total(
        total,
        away_runs_scored=5.3,
        home_runs_scored=4.5,
        away_runs_allowed=3.2,
        home_runs_allowed=4.4,
        league_avg_runs=4.5,
    )
    assert away > home, f"away={away} home={home}"
    assert abs((home + away) - total) < 0.01
    assert away >= 5.0


def test_equal_teams_near_league_hfa():
    total = 9.0
    home, away = split_total(
        total,
        away_runs_scored=4.5,
        home_runs_scored=4.5,
        away_runs_allowed=4.5,
        home_runs_allowed=4.5,
        league_avg_runs=4.5,
    )
    assert home > away
    assert 4.4 <= home <= 4.7
    assert 4.3 <= away <= 4.6


def test_home_favorite_projects_more_runs():
    total = 8.5
    home, away = split_total(
        total,
        away_runs_scored=3.8,
        home_runs_scored=5.2,
        away_runs_allowed=5.0,
        home_runs_allowed=3.5,
        league_avg_runs=4.5,
    )
    assert home > away


def test_sums_to_total():
    home, away = split_total(
        10.0,
        away_runs_scored=5.0,
        home_runs_scored=4.0,
        away_runs_allowed=4.0,
        home_runs_allowed=5.0,
    )
    assert abs(home + away - 10.0) < 0.001


if __name__ == "__main__":
    test_road_favorite_dodgers_at_arizona()
    test_equal_teams_near_league_hfa()
    test_home_favorite_projects_more_runs()
    test_sums_to_total()
    print("ok")
