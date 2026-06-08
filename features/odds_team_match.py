"""Shared team-name matching between Postgres team_name and Odds API labels."""


def team_names_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    x, y = a.strip().lower(), b.strip().lower()
    if x == y or x in y or y in x:
        return True
    x_last, y_last = x.split()[-1], y.split()[-1]
    return x_last == y_last and len(x_last) > 2


def match_game_to_odds_row(
    home_team: str,
    away_team: str,
    df_odds,
) -> dict | None:
    """Return the first odds row whose home/away teams match our DB names."""
    for _, row in df_odds.iterrows():
        kh = row.get("api_home_team") or row.get("home_team")
        ka = row.get("api_away_team") or row.get("away_team")
        if team_names_match(home_team, kh) and team_names_match(away_team, ka):
            return row.to_dict()
        if team_names_match(home_team, ka) and team_names_match(away_team, kh):
            return row.to_dict()
    return None
