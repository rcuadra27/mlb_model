import json
import os
import urllib.error
import urllib.parse
import urllib.request

import functions_framework
from google.cloud import bigquery
from psycopg2.extras import RealDictCursor
import psycopg2


def _normalize_pg_dsn(dsn: str) -> str:
    """Strip SQLAlchemy +psycopg2 for raw psycopg2."""
    if dsn.startswith("postgresql+psycopg2://"):
        return "postgresql://" + dsn[len("postgresql+psycopg2://") :]
    if dsn.startswith("postgres+psycopg2://"):
        return "postgres://" + dsn[len("postgres+psycopg2://") :]
    return dsn


def _fetch_graded_games_from_pg(dsn: str, min_date_exclusive: str) -> list:
    """
    Same row shape as _fetch_graded_games_from_bq, from Cloud SQL
    (updates as soon as scores land in public.games).
    """
    dsn = _normalize_pg_dsn(dsn.strip())
    q = """
        WITH latest AS (
            SELECT DISTINCT ON (p.game_id)
                p.game_id,
                p.game_date::text AS game_date,
                p.home_team,
                p.away_team,
                p.p_home_win_poisson::float AS p_home,
                p.p_away_win_poisson::float AS p_away,
                p.total_runs_pred::float AS total_runs_pred,
                COALESCE(fg.closing_ou_line, fg.morning_ou_line, p.ou_line)::float AS ou_line,
                COALESCE(fg.closing_home_price, fg.morning_home_price)::int AS home_odds,
                COALESCE(fg.closing_away_price, fg.morning_away_price)::int AS away_odds,
                g.home_runs,
                g.away_runs,
                g.status
            FROM public.inference_game_predictions p
            LEFT JOIN public.features_game fg ON fg.game_id = p.game_id
            LEFT JOIN public.games g ON g.game_id = p.game_id AND g.game_date = p.game_date
            WHERE p.game_date > %s::date
            ORDER BY p.game_id, p.as_of_ts DESC NULLS LAST
        )
        SELECT * FROM latest
        WHERE home_runs IS NOT NULL
          AND away_runs IS NOT NULL
          AND (
              LOWER(COALESCE(status, '')) LIKE 'final%%'
              OR LOWER(COALESCE(status, '')) = 'game over'
              OR LOWER(COALESCE(status, '')) LIKE 'completed%%'
          )
        ORDER BY game_date, game_id
    """
    conn = psycopg2.connect(dsn, connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (min_date_exclusive,))
            return [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()

_ODDS_MLB = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
_ODDS_EVENTS = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"


def _norm_team(s):
    return (s or "").lower().replace(".", "").strip()


def _team_match(a, b):
    na, nb = _norm_team(a), _norm_team(b)
    if na == nb:
        return True
    la = na.split()[-1] if na else ""
    lb = nb.split()[-1] if nb else ""
    return len(la) > 2 and len(lb) > 2 and la == lb


def _find_event(events, away, home):
    if not isinstance(events, list):
        return None
    for ev in events:
        ea = ev.get("away_team") or ""
        eh = ev.get("home_team") or ""
        if _team_match(away, ea) and _team_match(home, eh):
            return ev
        if _team_match(away, eh) and _team_match(home, ea):
            return ev
    return None


def _h2h_price(outcomes, team):
    if not outcomes:
        return None
    for o in outcomes:
        name = o.get("name") or ""
        if _team_match(team, name):
            p = o.get("price")
            if p is None:
                continue
            try:
                return int(round(float(p)))
            except (TypeError, ValueError):
                return None
    return None


def _book_sort_key(book):
    pref = [
        "draftkings", "fanduel", "betmgm", "caesars", "pointsbetus", "betrivers",
        "wynnbet", "barstool", "superbook", "unibet_us", "foxbet", "twinspires",
    ]
    k = book.get("key") or ""
    try:
        i = pref.index(k)
    except ValueError:
        i = 99
    return (i, (book.get("title") or book.get("key") or "").lower())


def _build_moneyline_matrix(ev, away, home):
    books = sorted(ev.get("bookmakers") or [], key=_book_sort_key)
    columns = []
    for book in books:
        market = None
        for m in book.get("markets") or []:
            if m.get("key") == "h2h":
                market = m
                break
        if not market:
            continue
        outs = market.get("outcomes") or []
        av = _h2h_price(outs, away)
        hv = _h2h_price(outs, home)
        if av is None and hv is None:
            continue
        columns.append(
            {
                "key": book.get("key") or "",
                "title": book.get("title") or book.get("key") or "",
                "away": av,
                "home": hv,
            }
        )
    return {"columns": columns}


def _fetch_json_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-model-get-daily-predictions/1"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode())


def _fetch_odds_event_payload(api_key, away, home):
    q = urllib.parse.urlencode(
        {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        }
    )
    try:
        odds_list = _fetch_json_url(f"{_ODDS_MLB}?{q}")
        ev = _find_event(odds_list, away, home)
        if ev and ev.get("bookmakers"):
            return ev
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        pass

    try:
        q2 = urllib.parse.urlencode({"apiKey": api_key})
        events = _fetch_json_url(f"{_ODDS_EVENTS}?{q2}")
        ev = _find_event(events, away, home)
        eid = ev.get("id") if ev else None
        if not eid:
            return None
        q3 = urllib.parse.urlencode(
            {
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
            }
        )
        enc = urllib.parse.quote(str(eid), safe="")
        detail = _fetch_json_url(
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{enc}/odds?{q3}"
        )
        return detail
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def _odds_board_view(request):
    headers = {"Access-Control-Allow-Origin": "*"}
    api_key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not api_key:
        return ({"columns": [], "error": "no_odds_api_key"}, 200, headers)

    away = (request.args.get("away_team") or "").strip()
    home = (request.args.get("home_team") or "").strip()
    if not away or not home:
        return ({"columns": [], "error": "missing_teams"}, 400, headers)

    ev = _fetch_odds_event_payload(api_key, away, home)
    if not ev:
        return ({"columns": [], "error": "no_event"}, 200, headers)

    return (_build_moneyline_matrix(ev, away, home), 200, headers)


def _row_to_dict(row):
    out = dict(row)
    for k, v in out.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif v is None:
            out[k] = None
    return out


def _run_daily_games_query(client, date):
    date_filter = f"AND game_date = '{date}'" if date else ""
    query = f"""
        WITH latest AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY as_of_ts DESC
                ) AS rn
            FROM `mlb-model-491223.mlb_model_logs.daily_games`
            WHERE TRUE {date_filter}
        )
        SELECT
            CAST(game_date AS STRING)                           AS game_date,
            game_id,
            away_team,
            home_team,
            away_sp_name,
            home_sp_name,
            ROUND(CAST(away_runs_pred AS FLOAT64), 2)          AS away_runs_pred,
            ROUND(CAST(home_runs_pred AS FLOAT64), 2)          AS home_runs_pred,
            ROUND(CAST(total_runs_pred AS FLOAT64), 2)         AS total_runs_pred,
            ROUND(CAST(p_win_away AS FLOAT64) * 100, 1)        AS p_win_away,
            ROUND(CAST(p_win_home AS FLOAT64) * 100, 1)        AS p_win_home,
            COALESCE(CAST(ou_line AS FLOAT64), CAST(morning_ou_line AS FLOAT64)) AS ou_line,
            ou_recommendation,
            ROUND(CAST(ou_edge_over AS FLOAT64) * 100, 1)      AS ou_edge_over,
            ROUND(CAST(ou_edge_under AS FLOAT64) * 100, 1)     AS ou_edge_under,
            ROUND(CAST(edge_away AS FLOAT64) * 100, 1)         AS edge_away,
            ROUND(CAST(edge_home AS FLOAT64) * 100, 1)         AS edge_home,
            is_value_ml_away,
            is_value_ml_home,
            is_value_ou_over,
            is_value_ou_under,
            ROUND(CAST(p_home_market_median AS FLOAT64) * 100, 1)  AS market_p_home,
            ROUND(CAST(p_away_market_median AS FLOAT64) * 100, 1)  AS market_p_away,
            CAST(morning_p_home AS FLOAT64)                    AS morning_p_home,
            CAST(closing_p_home AS FLOAT64)                    AS closing_p_home,
            CAST(morning_ou_line AS FLOAT64)                   AS morning_ou_line,
            CAST(closing_ou_line AS FLOAT64)                   AS closing_ou_line,
            CAST(total_line_move AS FLOAT64)                   AS total_line_move,
            CAST(home_line_move AS FLOAT64)                    AS home_line_move,
            CAST(sharp_action_home AS INT64)                   AS sharp_action_home,
            CAST(morning_home_price AS INT64)                  AS morning_home_price,
            CAST(morning_away_price AS INT64)                  AS morning_away_price,
            CAST(closing_home_price AS INT64)                  AS closing_home_price,
            CAST(closing_away_price AS INT64)                  AS closing_away_price,
            CAST(n_books_ml AS INT64)                          AS n_books_ml,
            CAST(n_books_ou AS INT64)                          AS n_books_ou,
            first_pitch_utc,
            CAST(away_runs AS INT64)                           AS away_runs,
            CAST(home_runs AS INT64)                           AS home_runs,
            status
        FROM latest
        WHERE rn = 1
        ORDER BY first_pitch_utc ASC NULLS LAST
    """
    return [_row_to_dict(r) for r in client.query(query).result()]


def _fetch_graded_games_from_bq(client, min_date_exclusive):
    """
    Reads graded games from the BigQuery mirror of Postgres
    (`mlb_model_logs.daily_games`) written by the daily pipeline.

    One row per game_id, using the most-recent prediction snapshot, filtered to:
      - game_date > @min_date_exclusive
      - both final-run columns populated
    """
    query = f"""
        WITH latest AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY as_of_ts DESC
                ) AS rn
            FROM `mlb-model-491223.mlb_model_logs.daily_games`
            WHERE game_date > DATE('{min_date_exclusive}')
        )
        SELECT
            CAST(game_date AS STRING)                                      AS game_date,
            game_id,
            home_team,
            away_team,
            CAST(p_win_home AS FLOAT64)                                    AS p_home,
            CAST(p_win_away AS FLOAT64)                                    AS p_away,
            CAST(total_runs_pred AS FLOAT64)                               AS total_runs_pred,
            COALESCE(
                CAST(closing_ou_line AS FLOAT64),
                CAST(morning_ou_line AS FLOAT64),
                CAST(ou_line AS FLOAT64)
            )                                                              AS ou_line,
            COALESCE(
                CAST(closing_home_price AS INT64),
                CAST(morning_home_price AS INT64)
            )                                                              AS home_odds,
            COALESCE(
                CAST(closing_away_price AS INT64),
                CAST(morning_away_price AS INT64)
            )                                                              AS away_odds,
            CAST(home_runs AS INT64)                                       AS home_runs,
            CAST(away_runs AS INT64)                                       AS away_runs,
            status
        FROM latest
        WHERE rn = 1
          AND home_runs IS NOT NULL
          AND away_runs IS NOT NULL
          AND (
              LOWER(IFNULL(status, '')) LIKE 'final%'
              OR LOWER(IFNULL(status, '')) = 'game over'
              OR LOWER(IFNULL(status, '')) LIKE 'completed%'
          )
        ORDER BY game_date ASC, game_id ASC
    """
    return [_row_to_dict(r) for r in client.query(query).result()]


def _ml_pnl(pick_won, pick_odds):
    """American-odds P&L on a $10 stake."""
    if pick_odds is None:
        return 0.0
    if not pick_won:
        return -10.0
    odds = float(pick_odds)
    if odds > 0:
        return 10.0 * (odds / 100.0)
    if odds < 0:
        return 10.0 * (100.0 / abs(odds))
    return 0.0


def _grade_games(raw_rows):
    """
    Produces a flat list of graded bet rows (one per {ml|ou} pick).
    Each row: {game_date, game_id, kind, is_hit, pnl_dollars, conf, conf_bucket, ou_pick, ou_line, total_actual, ou_result}
    """
    graded = []
    for r in raw_rows:
        game_date = r.get("game_date")
        game_id = r.get("game_id")
        p_home = r.get("p_home")
        p_away = r.get("p_away")
        home_runs = r.get("home_runs")
        away_runs = r.get("away_runs")
        home_odds = r.get("home_odds")
        away_odds = r.get("away_odds")
        total_pred = r.get("total_runs_pred")
        ou_line = r.get("ou_line")

        if home_runs is None or away_runs is None:
            continue
        home_runs = int(home_runs)
        away_runs = int(away_runs)
        total_actual = home_runs + away_runs

        # --- Moneyline (skip ties and missing predictions)
        if p_home is not None and p_away is not None and home_runs != away_runs:
            p_home_f = float(p_home)
            p_away_f = float(p_away)
            conf = max(p_home_f, p_away_f)
            pick_home = p_home_f >= p_away_f
            pick_won = (pick_home and home_runs > away_runs) or ((not pick_home) and away_runs > home_runs)
            pick_odds = home_odds if pick_home else away_odds
            pnl = _ml_pnl(pick_won, pick_odds)
            if conf < 0.55:
                bucket = "50-55%"
            elif conf < 0.60:
                bucket = "55-60%"
            elif conf < 0.65:
                bucket = "60-65%"
            else:
                bucket = "65%+"
            graded.append({
                "game_date": game_date,
                "game_id": game_id,
                "kind": "ml",
                "is_hit": 1 if pick_won else 0,
                "pnl_dollars": float(pnl),
                "conf": conf,
                "conf_bucket": bucket,
                "ou_pick": None,
                "ou_line": None,
                "total_actual": total_actual,
                "ou_result": None,
            })

        # --- Over/Under (grade whenever we have a model prediction and market line)
        if total_pred is not None and ou_line is not None:
            tp = float(total_pred)
            line = float(ou_line)
            if abs(tp - line) < 0.10:
                ou_pick = "push"
            elif tp > line:
                ou_pick = "over"
            else:
                ou_pick = "under"

            # Determine market-line push (integer-valued lines) vs hit/miss
            half_line = (abs(line * 10 - int(round(line * 10))) < 1e-6) and (int(round(line * 10)) % 10 != 0)
            if ou_pick == "push":
                ou_result = None  # model said no-bet
            elif (not half_line) and total_actual == int(round(line)):
                ou_result = "push"
            elif ou_pick == "over" and total_actual > line:
                ou_result = "hit"
            elif ou_pick == "under" and total_actual < line:
                ou_result = "hit"
            else:
                ou_result = "miss"

            if ou_result in ("hit", "miss"):
                if ou_result == "hit":
                    pnl = 10.0 * (100.0 / 110.0)
                else:
                    pnl = -10.0
                graded.append({
                    "game_date": game_date,
                    "game_id": game_id,
                    "kind": "ou",
                    "is_hit": 1 if ou_result == "hit" else 0,
                    "pnl_dollars": float(pnl),
                    "conf": None,
                    "conf_bucket": None,
                    "ou_pick": ou_pick,
                    "ou_line": line,
                    "total_actual": total_actual,
                    "ou_result": ou_result,
                })
    return graded


def _game_date_key(d) -> str:
    if d is None:
        return ""
    s = d if isinstance(d, str) else str(d)
    return s[:10] if len(s) >= 10 else s


# Model Accuracy tab (O/U only): omit graded O/U picks on these slate dates. Moneyline unchanged.
OU_ACCURACY_EXCLUDED_SLATE_DATES = frozenset(
    {
        "2026-04-23",
        "2026-04-24",
        "2026-04-25",
        "2026-04-26",
    }
)


def _drop_ou_bets_on_excluded_slates(graded_rows: list) -> list:
    if not OU_ACCURACY_EXCLUDED_SLATE_DATES:
        return graded_rows
    out = []
    for r in graded_rows:
        if (r.get("kind") or "ml") != "ou":
            out.append(r)
            continue
        if _game_date_key(r.get("game_date")) in OU_ACCURACY_EXCLUDED_SLATE_DATES:
            continue
        out.append(r)
    return out


def _run_accuracy_query(client=None):
    """
    Prefers Postgres (Cloud SQL) when PG_DSN is set so accuracy updates as soon
    as scores land; otherwise uses the BigQuery mirror. v9 cutoff: games after 2026-04-13.
    """
    min_date_exclusive = "2026-04-13"
    source_meta = "BigQuery mlb-model-491223.mlb_model_logs.daily_games (mirror of Postgres)"
    raw_rows = None
    pg_dsn = (os.environ.get("PG_DSN") or "").strip()
    if pg_dsn:
        try:
            raw_rows = _fetch_graded_games_from_pg(pg_dsn, min_date_exclusive)
            source_meta = "Postgres Cloud SQL (inference + games; updates when scores are ingested)"
        except Exception:
            raw_rows = None
    if raw_rows is None:
        if client is None:
            client = bigquery.Client()
        raw_rows = _fetch_graded_games_from_bq(client, min_date_exclusive)
        source_meta = "BigQuery mlb-model-491223.mlb_model_logs.daily_games (mirror of Postgres)"
    rows = _drop_ou_bets_on_excluded_slates(_grade_games(raw_rows))

    def _new_overall():
        return {"bets": 0, "wins": 0, "win_pct": None, "net_dollars": 0.0, "roi_pct": None}

    def _new_buckets():
        return {
            "50-55%": {"bucket": "50-55%", "bets": 0, "wins": 0, "net_dollars": 0.0},
            "55-60%": {"bucket": "55-60%", "bets": 0, "wins": 0, "net_dollars": 0.0},
            "60-65%": {"bucket": "60-65%", "bets": 0, "wins": 0, "net_dollars": 0.0},
            "65%+":   {"bucket": "65%+",   "bets": 0, "wins": 0, "net_dollars": 0.0},
        }

    ml_overall = _new_overall()
    ou_overall = _new_overall()
    ml_buckets = _new_buckets()
    ml_daily = {}
    ou_daily = {}
    combined_daily = {}
    ou_picks = {"over": 0, "under": 0}

    for r in rows:
        kind = r.get("kind") or "ml"
        hit = int(r["is_hit"])
        pnl = float(r["pnl_dollars"] or 0.0)
        d = r["game_date"]
        target = ml_overall if kind == "ml" else ou_overall
        target["bets"] += 1
        target["wins"] += hit
        target["net_dollars"] += pnl

        if kind == "ml":
            b = r["conf_bucket"]
            if b in ml_buckets:
                ml_buckets[b]["bets"] += 1
                ml_buckets[b]["wins"] += hit
                ml_buckets[b]["net_dollars"] += pnl

            if d not in ml_daily:
                ml_daily[d] = {"date": d, "bets": 0, "wins": 0, "net_dollars": 0.0}
            ml_daily[d]["bets"] += 1
            ml_daily[d]["wins"] += hit
            ml_daily[d]["net_dollars"] += pnl
        elif kind == "ou":
            pick = r.get("ou_pick")
            if pick in ou_picks:
                ou_picks[pick] += 1
            if d not in ou_daily:
                ou_daily[d] = {"date": d, "bets": 0, "wins": 0, "net_dollars": 0.0}
            ou_daily[d]["bets"] += 1
            ou_daily[d]["wins"] += hit
            ou_daily[d]["net_dollars"] += pnl

        if d not in combined_daily:
            combined_daily[d] = {"date": d, "bets": 0, "wins": 0, "net_dollars": 0.0}
        combined_daily[d]["bets"] += 1
        combined_daily[d]["wins"] += hit
        combined_daily[d]["net_dollars"] += pnl

    def _finalize_overall(o):
        if o["bets"] > 0:
            o["win_pct"] = round(100.0 * o["wins"] / o["bets"], 1)
            o["roi_pct"] = round(100.0 * o["net_dollars"] / (o["bets"] * 10.0), 1)
        o["net_dollars"] = round(o["net_dollars"], 2)

    _finalize_overall(ml_overall)
    _finalize_overall(ou_overall)

    bucket_rows = []
    for key in ["50-55%", "55-60%", "60-65%", "65%+"]:
        br = ml_buckets[key]
        if br["bets"] > 0:
            br["win_pct"] = round(100.0 * br["wins"] / br["bets"], 1)
            br["roi_pct"] = round(100.0 * br["net_dollars"] / (br["bets"] * 10.0), 1)
        else:
            br["win_pct"] = None
            br["roi_pct"] = None
        br["net_dollars"] = round(br["net_dollars"], 2)
        bucket_rows.append(br)

    def _finalize_daily(daily_dict):
        out = []
        cum = 0.0
        cum_rows = []
        for day in sorted(daily_dict.keys()):
            dr = daily_dict[day]
            if dr["bets"] > 0:
                dr["win_pct"] = round(100.0 * dr["wins"] / dr["bets"], 1)
                dr["roi_pct"] = round(100.0 * dr["net_dollars"] / (dr["bets"] * 10.0), 1)
            else:
                dr["win_pct"] = None
                dr["roi_pct"] = None
            dr["net_dollars"] = round(dr["net_dollars"], 2)
            cum += dr["net_dollars"]
            dr["cumulative_dollars"] = round(cum, 2)
            out.append(dr)
            cum_rows.append({"date": dr["date"], "cumulative_dollars": dr["cumulative_dollars"]})
        return out, cum_rows

    ml_daily_rows, ml_cum_rows = _finalize_daily(ml_daily)
    ou_daily_rows, ou_cum_rows = _finalize_daily(ou_daily)
    combined_daily_rows, combined_cum_rows = _finalize_daily(combined_daily)

    return {
        "overall": ml_overall,
        "buckets": bucket_rows,
        "daily": ml_daily_rows,
        "daily_cumulative": ml_cum_rows,
        "ou_overall": ou_overall,
        "ou_daily": ou_daily_rows,
        "ou_daily_cumulative": ou_cum_rows,
        "ou_pick_counts": ou_picks,
        "combined_daily": combined_daily_rows,
        "combined_daily_cumulative": combined_cum_rows,
        "meta": {
            "source": source_meta,
            "version": "v9",
            "min_game_date_exclusive": min_date_exclusive,
            "ou_pricing_assumption": "standard -110 (10/11 on hit)",
            "ou_slate_dates_excluded_from_ou_stats": sorted(OU_ACCURACY_EXCLUDED_SLATE_DATES),
            "graded_games": len(raw_rows),
            "graded_bets": len(rows),
        },
    }

@functions_framework.http
def get_daily_predictions(request):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return ("", 204, headers)

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json; charset=utf-8",
    }
    view = (request.args.get("view", "") or "").strip().lower()
    if view == "odds_board":
        return _odds_board_view(request)
    if view == "accuracy":
        try:
            bq = None
            if not (os.environ.get("PG_DSN") or "").strip():
                bq = bigquery.Client()
            payload = _run_accuracy_query(bq)
            return (payload, 200, headers)
        except Exception as exc:
            err_body = {"error": "accuracy_query_failed", "message": str(exc)}
            return (err_body, 500, headers)

    client = bigquery.Client()
    date = request.args.get("date", None)
    games = _run_daily_games_query(client, date)
    return ({"games": games}, 200, headers)