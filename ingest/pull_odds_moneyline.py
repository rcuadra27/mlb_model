import argparse, os, json
import time
from datetime import date, datetime, timedelta, timezone
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

def process_date(client, game_date, args, conn):
    """Process odds for a single date."""
    print(f"\nProcessing date: {game_date}")
    
    # Format date as ISO datetime string for the API
    # Use snapshot hour if provided, otherwise use start of day (00:00 UTC)
    snapshot_hour = args.snapshot_hour_utc if args.snapshot_hour_utc is not None else 0
    date_dt = datetime.combine(game_date, datetime.min.time().replace(hour=snapshot_hour))
    date_dt = date_dt.replace(tzinfo=timezone.utc)
    # Format as ISO string with 'Z' suffix for UTC
    date_iso = date_dt.isoformat().replace('+00:00', 'Z')
    
    print(f"  Fetching odds with commenceTimeFrom: {date_iso}")
    
    try:
        data = client.get_moneylines(date_iso=date_iso)
    except Exception as e:
        print(f"Error fetching odds for {game_date}: {e}")
        return 0

    print(f"Events returned from API: {len(data)}")
    if data:
        print(f"First event keys: {list(data[0].keys())}")
        print(f"First commence_time: {data[0].get('commence_time')}")
        print(f"First home/away: {data[0].get('home_team')} vs {data[0].get('away_team')}")

    sb_allow = set(s.strip().lower() for s in args.bookmakers.split(",") if s.strip()) if args.bookmakers else set()

    rows = []
    for ev in data:
        commence = ev.get("commence_time")  # ISO timestamp string
        if not commence:
            continue
        
        # Parse commence_time to check date - filter to only games on our target date
        try:
            # Handle both 'Z' suffix and timezone-aware formats
            if commence.endswith('Z'):
                commence_dt = datetime.fromisoformat(commence.replace('Z', '+00:00'))
            else:
                commence_dt = datetime.fromisoformat(commence)
            
            # Convert to UTC date for comparison
            if commence_dt.tzinfo:
                commence_date = commence_dt.astimezone(timezone.utc).date()
            else:
                commence_date = commence_dt.date()
                
            if commence_date != game_date:
                continue
        except (ValueError, AttributeError) as e:
            # Fallback to string matching
            if not commence.startswith(str(game_date)):
                continue
            print(f"  Warning: Could not parse commence_time '{commence}': {e}")

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
                "game_date": str(game_date),
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
        print(f"No odds rows found for date: {game_date}")
        return 0

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
    print(f"Inserted {len(rows)} odds rows for {game_date}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"))
    ap.add_argument("--provider", default="theoddsapi")
    ap.add_argument("--sportbook_filter", default="", help="comma-separated, optional (deprecated, use --bookmakers)")
    ap.add_argument("--bookmakers", default="", help="comma-separated list of bookmakers to filter")
    ap.add_argument("--game_date", help="YYYY-MM-DD (single date, ignored if --start/--end provided)")
    ap.add_argument("--start", help="YYYY-MM-DD (start of date range)")
    ap.add_argument("--end", help="YYYY-MM-DD (end of date range)")
    ap.add_argument("--snapshot_hour_utc", type=int, help="Hour in UTC to take snapshot (0-23)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls")
    args = ap.parse_args()

    if not args.pg_dsn:
        raise RuntimeError("Missing PG_DSN")

    # Handle bookmakers argument (support both old and new names)
    if not args.bookmakers and args.sportbook_filter:
        args.bookmakers = args.sportbook_filter

    # Determine date range
    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
    elif args.game_date:
        dates = [date.fromisoformat(args.game_date)]
    else:
        dates = [date.today()]

    print(f"Processing {len(dates)} date(s) from {dates[0]} to {dates[-1]}")

    client = TheOddsAPI()
    conn = psycopg2.connect(args.pg_dsn)

    total_rows = 0
    processed_dates = 0
    today = date.today()
    
    for i, game_date in enumerate(dates):
        # Handle snapshot hour wait logic for future dates only
        if args.snapshot_hour_utc is not None and game_date >= today:
            # Calculate when we should take the snapshot (in UTC)
            snapshot_time = datetime.combine(game_date, datetime.min.time().replace(hour=args.snapshot_hour_utc))
            snapshot_time = snapshot_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            
            if now < snapshot_time:
                wait_seconds = (snapshot_time - now).total_seconds()
                print(f"Waiting {wait_seconds:.1f} seconds until snapshot time {snapshot_time} UTC")
                time.sleep(wait_seconds)
            elif now > snapshot_time + timedelta(hours=1):
                print(f"Warning: Snapshot time {snapshot_time} UTC has passed, proceeding anyway")

        rows = process_date(client, game_date, args, conn)
        processed_dates += 1
        total_rows += rows

        # Sleep between dates (except after the last one)
        if args.sleep > 0 and i < len(dates) - 1:
            print(f"Sleeping {args.sleep} seconds...")
            time.sleep(args.sleep)

    conn.close()
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total dates processed: {processed_dates}")
    print(f"  Total odds rows inserted: {total_rows}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
