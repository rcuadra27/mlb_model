import argparse, os, json
from datetime import date
import psycopg2
from odds.the_odds_api import TheOddsAPI


def try_map_game_id(cur, game_date, home_name, away_name):
    """
    Best-effort mapping to games.game_id.
    First try exact name match on games.home_team_name/away_team_name.
    If you standardize names later, this gets better.
    """
    cur.execute(
        """
        SELECT game_id
        FROM games
        WHERE game_date = %s
          AND home_team_name = %s
          AND away_team_name = %s
        LIMIT 1;
        """,
        (game_date, home_name, away_name),
    )
    row = cur.fetchone()
    return row[0] if row else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"))
    ap.add_argument("--provider", default="theoddsapi")
    ap.add_argument("--sportbook_filter", default="", help="comma-separated, optional")
    ap.add_argument("--game_date", default=str(date.today()), help="YYYY-MM-DD")
    args = ap.parse_args()

    if not args.pg_dsn:
        raise RuntimeError("Missing PG_DSN")

    game_date = args.game_date

    client = TheOddsAPI()

    data = client.get_moneylines()  # we’ll filter locally by date using commence_time

    print("Events returned:", len(data))
    if data:
        print("First event keys:", list(data[0].keys()))
        print("First commence_time:", data[0].get("commence_time"))
        print("First home/away:", data[0].get("home_team"), "vs", data[0].get("away_team"))


    sb_allow = set(s.strip().lower() for s in args.sportbook_filter.split(",") if s.strip())

    rows = []
    for ev in data:
        commence = ev.get("commence_time")  # ISO timestamp string
        if not commence or not commence.startswith(game_date):
            continue

        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        for book in ev.get("bookmakers", []) or []:
            sb = (book.get("title") or book.get("key") or "").strip()
            if not sb:
                continue
            if sb_allow and sb.lower() not in sb_allow:
                continue

            # find h2h market
            markets = book.get("markets", []) or []
            h2h = next((m for m in markets if m.get("key") == "h2h"), None)
            if not h2h:
                continue

            outcomes = h2h.get("outcomes", []) or []
            # outcomes typically have name + price
            home_price = None
            away_price = None
            for o in outcomes:
                name = o.get("name")
                price = o.get("price")
                if name == home:
                    home_price = price
                elif name == away:
                    away_price = price

            rows.append({
                "game_date": game_date,
                "provider": args.provider,
                "sportsbook": sb,
                "market": "h2h",
                "home_team": home,
                "away_team": away,
                "home_price": home_price,
                "away_price": away_price,
                "commence_time": commence,
                "raw": ev,
            })

    if not rows:
        print("No odds rows found for date:", game_date)
        return

    conn = psycopg2.connect(args.pg_dsn)
    with conn.cursor() as cur:
        for r in rows:
            game_id = try_map_game_id(cur, r["game_date"], r["home_team"], r["away_team"])
            cur.execute(
                """
                INSERT INTO odds_ml (
                  game_date, game_id, provider, sportsbook, market,
                  home_team, away_team, home_price, away_price, commence_time, raw
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb);
                """,
                (
                    r["game_date"],
                    game_id,
                    r["provider"],
                    r["sportsbook"],
                    r["market"],
                    r["home_team"],
                    r["away_team"],
                    r["home_price"],
                    r["away_price"],
                    r["commence_time"],
                    json.dumps(r["raw"]),
                ),
            )
    conn.commit()
    conn.close()
    print(f"Inserted {len(rows)} odds rows for {game_date}")

if __name__ == "__main__":
    main()
