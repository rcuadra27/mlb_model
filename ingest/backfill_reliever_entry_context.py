import argparse, os, time, math
import requests
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed

LIVE_URL = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"

def is_high_leverage_proxy(inning, diff):
    if inning is None or diff is None:
        return None
    return (inning >= 7) and (abs(diff) <= 2)

def fetch_live(game_id, session):
    r = session.get(LIVE_URL.format(gamePk=game_id), timeout=40)
    r.raise_for_status()
    return r.json()

def score_at_start_of_half(linescore, inning, half):
    """
    Approx score at start of the half-inning using linescore innings.
    inning is 1-based. half is 'top' or 'bottom'.
    Returns (home_score, away_score) at start of that half.
    """
    innings = (linescore or {}).get("innings", []) or []

    # sum completed innings strictly before this inning
    home = 0
    away = 0
    for i in range(0, max(0, inning - 1)):
        inn = innings[i] if i < len(innings) else {}
        home += ((inn.get("home") or {}).get("runs") or 0)
        away += ((inn.get("away") or {}).get("runs") or 0)

    # if bottom of current inning, add away runs from top of this inning (already happened)
    if half == "bottom":
        idx = inning - 1
        if idx < len(innings):
            inn = innings[idx] or {}
            away += ((inn.get("away") or {}).get("runs") or 0)

    return home, away

def extract_context_from_play(play):
    """
    Returns (inning:int|None, half:str|None, outs:int|None, pitcher_id:int|None)
    using robust fallbacks for older MLB feeds.
    """
    about = play.get("about") or {}
    matchup = play.get("matchup") or {}
    pitcher = (matchup.get("pitcher") or {})
    pitcher_id = pitcher.get("id")

    # First try play["about"]
    inning = about.get("inning")
    half = about.get("halfInning")
    outs = about.get("outs")

    # Fallback: playEvents (older feeds often have inning/outs here)
    if inning is None or half is None or outs is None:
        events = play.get("playEvents") or []
        if events:
            ev = events[-1]  # last event usually has latest/valid context
            # inning
            inning = inning if inning is not None else ev.get("inning")
            # half: some feeds use boolean isTopInning
            if half is None:
                if ev.get("isTopInning") is True:
                    half = "top"
                elif ev.get("isTopInning") is False:
                    half = "bottom"
                else:
                    half = ev.get("halfInning")  # rare
            # outs: sometimes ev["outs"], sometimes ev["count"]["outs"]
            if outs is None:
                outs = ev.get("outs")
                if outs is None:
                    outs = (ev.get("count") or {}).get("outs")

    # normalize half if present
    if half is not None:
        half = str(half).lower()

    return inning, half, outs, pitcher_id



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="debug limit")
    args = ap.parse_args()

    conn = psycopg2.connect(args.pg_dsn)

    # games that have relief appearances missing context
    with conn.cursor() as cur:
        cur.execute("""
          SELECT DISTINCT pa.game_id
          FROM pitcher_appearances pa
          JOIN games g ON g.game_id = pa.game_id
          WHERE g.game_type='R'
            AND pa.is_starter = FALSE
            AND (pa.inning_entered IS NULL OR pa.score_diff_on_entry IS NULL OR pa.is_high_leverage IS NULL)
          ORDER BY pa.game_id;
        """)
        game_ids = [r[0] for r in cur.fetchall()]

    if args.limit and args.limit > 0:
        game_ids = game_ids[:args.limit]

    print(f"Need relief context for {len(game_ids)} games")

    session = requests.Session()

    updates = []  # (inning, outs, diff, is_hlev, game_id, pitcher_id)

    def process_game(game_id):
        data = fetch_live(game_id, session)

        # team IDs
        home_id = data["gameData"]["teams"]["home"]["id"]
        away_id = data["gameData"]["teams"]["away"]["id"]
        linescore = data.get("liveData", {}).get("linescore", {})


        # We'll track first-seen pitcher per side
        first_seen = set()

        plays = data["liveData"]["plays"]["allPlays"]
        local_updates = []

        cnt_total = 0
        cnt_skip_about = 0
        cnt_skip_pitcher = 0
        cnt_skip_half = 0
        cnt_skip_seen = 0

        for play in plays:
            cnt_total += 1

            inning, half_norm, outs, pitcher_id = extract_context_from_play(play)

            if inning is None or half_norm is None or outs is None:
                cnt_skip_about += 1
                continue

            if pitcher_id is None:
                cnt_skip_pitcher += 1
                continue

            if half_norm not in ("top", "bottom"):
                cnt_skip_half += 1
                continue

            pitching_team_id = home_id if half_norm == "top" else away_id

            home_score, away_score = score_at_start_of_half(linescore, inning, half_norm)

            if pitching_team_id == home_id:
                diff = home_score - away_score
            else:
                diff = away_score - home_score

            key = (game_id, pitching_team_id, pitcher_id)
            if key in first_seen:
                cnt_skip_seen += 1
                continue
            first_seen.add(key)

            hlev = is_high_leverage_proxy(inning, diff)
            local_updates.append((inning, outs, diff, hlev, game_id, pitcher_id))


        if len(local_updates) == 0:
                print(
                    f"DEBUG game {game_id}: plays={cnt_total} "
                    f"skip_about={cnt_skip_about} skip_pitcher={cnt_skip_pitcher} "
                    f"skip_half={cnt_skip_half} skip_seen={cnt_skip_seen}"
                )

        return local_updates

    # parallel fetch
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_game, gid): gid for gid in game_ids}
        done = 0
        for fut in as_completed(futs):
            gid = futs[fut]
            try:
                updates.extend(fut.result())
            except Exception as e:
                print(f"FAIL game {gid}: {e}")
            done += 1
            if done % 200 == 0:
                print(f"Processed {done}/{len(game_ids)} games")
            time.sleep(0.02)  # mild politeness

    print(f"Writing {len(updates)} pitcher entry updates")

    sql = """
    UPDATE pitcher_appearances
    SET
      inning_entered = %s,
      outs_on_entry = %s,
      score_diff_on_entry = %s,
      is_high_leverage = %s
    WHERE game_id = %s AND pitcher_id = %s AND is_starter = FALSE;
    """

    with conn.cursor() as cur:
        cur.executemany(sql, updates)
    conn.commit()
    conn.close()
    print("DONE")

if __name__ == "__main__":
    main()
