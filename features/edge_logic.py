"""Shared edge computation for daily_edges pre-compute pipeline."""

from __future__ import annotations

import math

ML_EDGE_MIN = 0.05
TOTAL_EDGE_MIN = 0.8
K_RATE_EDGE_MIN = 1.2
WALKS_EDGE_MIN = 0.35
HITS_EDGE_MIN = 0.75
ER_EDGE_MIN = 0.45
# Posted ER lines at 1.5 are a low bar — exclude from Top Edges (not actionable).
MIN_ER_MARKET_LINE = 2.0

DEFAULT_EXPECTED_IP = 5.5
MIN_COUNTING_STAT_IP = 5.0

PITCHER_RATE_EDGE_TYPES = frozenset({"k", "walks", "hits", "er"})
COUNTING_STAT_EDGE_TYPES = PITCHER_RATE_EDGE_TYPES

RATE_EDGE_CONFIG = {
    "k": {"lambda_col": "lambda_k", "label": "Ks", "stat_word": "K", "rate_label": "K/9", "baseline_label": "season K/9"},
    "walks": {"lambda_col": "lambda_walks", "label": "Walks", "stat_word": "BB", "rate_label": "BB/9", "baseline_label": "season BB/9"},
    "hits": {"lambda_col": "lambda_hits", "label": "Hits", "stat_word": "hits", "rate_label": "H/9", "baseline_label": "season H/9"},
    "er": {"lambda_col": "lambda_er", "label": "ER", "stat_word": "ER", "rate_label": "ERA", "baseline_label": "season ERA"},
}

RATE_EDGE_MIN = {
    "k": K_RATE_EDGE_MIN,
    "walks": WALKS_EDGE_MIN,
    "hits": HITS_EDGE_MIN,
    "er": ER_EDGE_MIN,
}

LEAGUE_BASE_RATES = {
    "p_hit": 0.608,
    "p_2plus_hits": 0.220,
    "p_hr": 0.117,
    "p_k": 0.611,
    "p_2plus_bases": 0.343,
    "p_walk": 0.300,
}

EDGE_THRESHOLDS = {
    "p_hit": 0.10,
    "p_2plus_hits": 0.10,
    "p_hr": 0.06,
    "p_k": 0.10,
    "p_2plus_bases": 0.10,
    "p_walk": 0.08,
}

PROP_LABELS = {
    "p_hit": "1+ Hit",
    "p_hr": "HR",
    "p_2plus_hits": "2+ Hits",
    "p_k": "1+ K",
    "p_2plus_bases": "2+ TB",
    "p_walk": "1+ Walk",
}

PROP_SUBTYPES = {
    "p_hit": "HIT",
    "p_hr": "HR",
    "p_2plus_hits": "2+ H",
    "p_k": "K",
    "p_2plus_bases": "TB",
    "p_walk": "WALK",
}

BATTER_PROP_KEYS = (
    "p_hr",
    "p_2plus_bases",
    "p_2plus_hits",
    "p_hit",
    "p_k",
    "p_walk",
)

MAX_PER_PROP_SUBTYPE = 3
TOP_N = 15
# When tier-2 pitcher stat edges fill the list, cap so ER/walks cannot monopolize Top Edges.
TIER2_PITCHER_STAT_MAX = 6
TIER2_PITCHER_TYPE_MAX = 3

# Minimum model prob edge (pp from 50%) to surface a posted-line pitcher prop.
COUNTING_STAT_MARKET_PROB_EDGE_PP = 8.0
# Weak posted-line edges rank below ML/totals/batter props; strong prob edges join tier-1.
PRIMARY_EDGE_TYPES = frozenset({"ml", "total", "prop"})


def _safe_float(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _finite_edge_magnitude(value) -> float | None:
    """Coerce edge magnitude for DB NOT NULL; return None if non-finite."""
    x = _safe_float(value)
    return x


def _fmt_american(odds) -> str | None:
    if odds is None:
        return None
    try:
        o = int(round(float(odds)))
    except (TypeError, ValueError):
        return None
    return f"+{o}" if o > 0 else str(o)


def normalized_edge_score(edge_type: str, raw_magnitude: float) -> float:
    if edge_type == "ml":
        return min(100.0, max(0.0, (raw_magnitude - 5.0) / 15.0 * 100.0))
    if edge_type == "total":
        return min(100.0, max(0.0, (raw_magnitude - TOTAL_EDGE_MIN) / 1.7 * 100.0))
    if edge_type in PITCHER_RATE_EDGE_TYPES:
        base = RATE_EDGE_MIN[edge_type]
        return min(100.0, max(0.0, (raw_magnitude - base) / (base * 3.0) * 100.0))
    if edge_type == "prop":
        return min(100.0, max(0.0, raw_magnitude / 20.0 * 100.0))
    return raw_magnitude


def comparison_rate(prop_key: str, personal: dict | None) -> tuple[float, str]:
    """Return (comparison_rate, baseline_label) where label is 'league' or 'blend'."""
    league = LEAGUE_BASE_RATES[prop_key]
    if not personal:
        return league, "league"
    pa = personal.get("sd_pa") or 0
    personal_rate = personal.get(f"personal_{prop_key}")
    if pa >= 50 and personal_rate is not None:
        blended = (personal_rate * pa + league * 100) / (pa + 100)
        return blended, "blend"
    return league, "league"


def ml_edges_from_games(games: list) -> list:
    out = []
    for g in games:
        eh = _safe_float(g.get("edge_home"))
        ea = _safe_float(g.get("edge_away"))
        if eh is None and ea is None:
            continue
        eh = eh if eh is not None else 0.0
        ea = ea if ea is not None else 0.0
        if eh >= ea:
            if eh < ML_EDGE_MIN:
                continue
            pick_home = True
            edge = eh
        else:
            if ea < ML_EDGE_MIN:
                continue
            pick_home = False
            edge = ea

        away = g.get("away_team") or "Away"
        home = g.get("home_team") or "Home"
        pick_team = home if pick_home else away
        p_model = _safe_float(g.get("p_win_home") if pick_home else g.get("p_win_away"))
        p_mkt = _safe_float(g.get("p_home_market") if pick_home else g.get("p_away_market"))
        odds = (
            g.get("morning_home_price") if pick_home else g.get("morning_away_price")
        ) or (g.get("closing_home_price") if pick_home else g.get("closing_away_price"))
        edge_pp = edge * 100.0
        detail_bits = []
        if p_model is not None:
            detail_bits.append(f"Model {p_model * 100:.1f}%")
        if p_mkt is not None:
            detail_bits.append(f"market {p_mkt * 100:.1f}%")
        odds_fmt = _fmt_american(odds)
        if odds_fmt:
            detail_bits.append(odds_fmt)

        out.append({
            "edge_type": "ml",
            "prop_subtype": None,
            "direction": "over",
            "game_id": g.get("game_id"),
            "player_id": None,
            "pick_description": f"{away} @ {home} — {pick_team}",
            "detail_line": " vs ".join(detail_bits[:2]) + (f" · {detail_bits[2]}" if len(detail_bits) > 2 else ""),
            "model_value": (p_model * 100.0) if p_model is not None else None,
            "comparison_value": (p_mkt * 100.0) if p_mkt is not None else None,
            "edge_magnitude": edge_pp,
            "sort_score": normalized_edge_score("ml", edge_pp),
        })
    return out


def total_edges_from_games(games: list) -> list:
    out = []
    for g in games:
        oe = _safe_float(g.get("ou_edge_over"))
        ue = _safe_float(g.get("ou_edge_under"))
        if oe is None and ue is None:
            continue
        oe = oe if oe is not None else 0.0
        ue = ue if ue is not None else 0.0
        if oe >= ue:
            if oe < TOTAL_EDGE_MIN:
                continue
            side, edge, direction = "Over", oe, "over"
        else:
            if ue < TOTAL_EDGE_MIN:
                continue
            side, edge, direction = "Under", ue, "under"

        away = g.get("away_team") or "Away"
        home = g.get("home_team") or "Home"
        line = _safe_float(g.get("ou_line"))
        total_pred = _safe_float(g.get("total_runs_pred"))
        line_str = f"{line:.1f}" if line is not None else "—"
        pred_str = f"{total_pred:.1f}" if total_pred is not None else "—"
        out.append({
            "edge_type": "total",
            "prop_subtype": None,
            "direction": direction,
            "game_id": g.get("game_id"),
            "player_id": None,
            "pick_description": f"{away} @ {home} — {side} {line_str}",
            "detail_line": f"Model {pred_str} vs O/U line {line_str}",
            "model_value": total_pred,
            "comparison_value": line,
            "edge_magnitude": edge,
            "sort_score": normalized_edge_score("total", edge),
        })
    return out


def k_edges_from_pitchers(pitchers: list, k_rates: dict) -> list:
    return _pitcher_rate_poisson_edges(pitchers, k_rates, "k", K_RATE_EDGE_MIN)


COUNTING_STAT_ABBREV = {
    "k": "K",
    "walks": "BB",
    "hits": "H",
    "er": "ER",
}


def _normalize_pitcher_name(name: str | None) -> str:
    return (name or "").strip().lower()


def _pitcher_market_lines(pitcher: dict) -> dict[str, float]:
    raw = pitcher.get("_market_lines")
    return raw if isinstance(raw, dict) else {}


def _poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    total = 0.0
    term = math.exp(-lam)
    total += term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(1.0, max(0.0, total))


def _poisson_over_prob(lam: float, line: float) -> float:
    k_floor = int(float(line) + 0.5)
    return 1.0 - _poisson_cdf(k_floor - 1, lam)


def _poisson_under_prob(lam: float, line: float) -> float:
    return 1.0 - _poisson_over_prob(lam, line)


def _season_count_for_ip(season_rate: float, ip: float) -> float:
    return season_rate * ip / 9.0


def _pitcher_poisson_pick_description(
    name: str,
    over: bool,
    market_line: float,
    edge_type: str,
) -> str:
    side = "Over" if over else "Under"
    abbr = COUNTING_STAT_ABBREV[edge_type]
    return f"{name} — {side} {market_line:.1f} {abbr}"


def _pitcher_counting_stat_detail_line(model_prob: float) -> str:
    return f"Model: {model_prob * 100:.1f}%"


def _pitcher_expected_ip(pitcher: dict) -> float:
    ip = _safe_float(pitcher.get("expected_ip"))
    if ip is None or ip <= 0:
        return DEFAULT_EXPECTED_IP
    return ip


def _pitcher_rate_detail_line(lam: float, market_line: float, stat_word: str) -> str:
    return f"Model projects {lam:.1f} {stat_word} vs posted {market_line:.1f}"


def _pitcher_rate_poisson_edges(
    pitchers: list,
    season_rates: dict,
    edge_type: str,
    min_gap: float,
) -> list:
    """Pitcher K/walks/hits/ER edges: require posted line; rank on model prob vs line."""
    del season_rates, min_gap  # kept for call-site compatibility
    cfg = RATE_EDGE_CONFIG[edge_type]
    lambda_col = cfg["lambda_col"]
    label = cfg["label"]
    min_prob_pp = COUNTING_STAT_MARKET_PROB_EDGE_PP
    out = []
    for p in pitchers:
        pid = p.get("pitcher_id")
        lam = _safe_float(p.get(lambda_col))
        if pid is None or lam is None:
            continue
        ip = _pitcher_expected_ip(p)
        if ip < MIN_COUNTING_STAT_IP:
            continue
        market_line = _safe_float(_pitcher_market_lines(p).get(edge_type))
        if market_line is None:
            continue
        if edge_type == "er" and market_line < MIN_ER_MARKET_LINE:
            continue
        p_over = _poisson_over_prob(lam, market_line)
        p_under = _poisson_under_prob(lam, market_line)
        over = p_over >= p_under
        model_prob = p_over if over else p_under
        edge_pp = abs(model_prob - 0.5) * 100.0
        if edge_pp < min_prob_pp:
            continue
        mag = _finite_edge_magnitude(edge_pp)
        if mag is None:
            continue
        name = p.get("pitcher_name") or "SP"
        out.append({
            "edge_type": edge_type,
            "prop_subtype": label.upper(),
            "direction": "over" if over else "under",
            "game_id": p.get("game_id"),
            "player_id": pid,
            "team_id": p.get("team_id"),
            "team_abbr": p.get("team_abbr"),
            "team_name": p.get("team_name"),
            "pick_description": _pitcher_poisson_pick_description(
                name, over, market_line, edge_type,
            ),
            "detail_line": _pitcher_counting_stat_detail_line(model_prob),
            "rate_detail_line": _pitcher_rate_detail_line(lam, market_line, cfg["stat_word"]),
            "market_line": market_line,
            "model_prob_pct": model_prob * 100.0,
            "expected_ip": ip,
            "model_value": lam,
            "comparison_value": market_line,
            "edge_magnitude": mag,
            "sort_score": normalized_edge_score("prop", edge_pp),
        })
    return out


def walks_edges_from_pitchers(pitchers: list, avgs: dict) -> list:
    return _pitcher_rate_poisson_edges(pitchers, avgs, "walks", WALKS_EDGE_MIN)


def hits_edges_from_pitchers(pitchers: list, avgs: dict) -> list:
    return _pitcher_rate_poisson_edges(pitchers, avgs, "hits", HITS_EDGE_MIN)


def er_edges_from_pitchers(pitchers: list, avgs: dict) -> list:
    return _pitcher_rate_poisson_edges(pitchers, avgs, "er", ER_EDGE_MIN)


def batter_prop_edges(batters: list, personal_rates: dict) -> list:
    out = []
    for b in batters:
        bid = b.get("batter_id")
        if bid is None:
            continue
        personal = personal_rates.get(int(bid)) or personal_rates.get(bid)
        name = b.get("batter_name") or "Batter"
        best = None
        for prop_key in BATTER_PROP_KEYS:
            p_model = _safe_float(b.get(prop_key))
            if p_model is None:
                continue
            comp, baseline_kind = comparison_rate(prop_key, personal)
            dev = p_model - comp
            threshold = EDGE_THRESHOLDS[prop_key]
            if dev < threshold:
                continue
            score = normalized_edge_score("prop", dev * 100.0)
            if best is None or score > best["score"]:
                best = {
                    "prop_key": prop_key,
                    "p_model": p_model,
                    "comp": comp,
                    "dev": dev,
                    "baseline_kind": baseline_kind,
                    "score": score,
                }
        if not best:
            continue

        prop_key = best["prop_key"]
        label = PROP_LABELS[prop_key]
        subtype = PROP_SUBTYPES[prop_key]
        if prop_key == "p_hr":
            title = f"{name} — HR more likely"
        elif prop_key in ("p_2plus_bases", "p_hit"):
            short = "2+ TB" if prop_key == "p_2plus_bases" else "Hit"
            title = f"{name} — {short}"
        else:
            title = f"{name} — {label} more likely"

        if best["baseline_kind"] == "league":
            detail = f"Model {best['p_model'] * 100:.1f}% vs league avg {best['comp'] * 100:.1f}%"
            edge_label_kind = "league"
        else:
            detail = f"Model {best['p_model'] * 100:.1f}% vs avg {best['comp'] * 100:.1f}%"
            edge_label_kind = "blend"

        out.append({
            "edge_type": "prop",
            "prop_subtype": subtype,
            "prop_key": prop_key,
            "direction": "over",
            "game_id": b.get("game_id"),
            "player_id": bid,
            "team_id": b.get("team_id"),
            "team_abbr": b.get("team_abbr"),
            "team_name": b.get("team_name"),
            "pick_description": title,
            "detail_line": detail,
            "model_value": best["p_model"] * 100.0,
            "comparison_value": best["comp"] * 100.0,
            "edge_magnitude": best["dev"] * 100.0,
            "edge_label_kind": edge_label_kind,
            "sort_score": best["score"],
        })
    return out


def _counting_stat_market_prob_edge_pp(item: dict) -> float | None:
    """Signed distance from 50% for the model's pick side (percentage points)."""
    prob_pct = _safe_float(item.get("model_prob_pct"))
    if prob_pct is None or item.get("market_line") is None:
        return None
    return abs(prob_pct - 50.0)


def _counting_stat_has_strong_market_edge(item: dict) -> bool:
    edge_pp = _counting_stat_market_prob_edge_pp(item)
    return edge_pp is not None and edge_pp >= COUNTING_STAT_MARKET_PROB_EDGE_PP


def _ranking_sort_key(item: dict) -> tuple[int, float]:
    """
    Cross-edge ranking key: (tier, score), lower tier always wins.

    Tier 0 — ML, totals, batter props (comparable %/pt baselines).
    Tier 1 — pitcher K / walks / hits / ER with strong model prob vs posted line.
    Tier 2 — same props with weaker prob edge (still posted-line only at generation).
    """
    edge_type = item.get("edge_type")
    if edge_type in PRIMARY_EDGE_TYPES:
        return 0, item.get("sort_score") or 0.0
    if edge_type in PITCHER_RATE_EDGE_TYPES:
        if _counting_stat_has_strong_market_edge(item):
            edge_pp = _counting_stat_market_prob_edge_pp(item) or 0.0
            return 1, normalized_edge_score("prop", edge_pp)
        return 2, item.get("sort_score") or 0.0
    return 0, item.get("sort_score") or 0.0


def select_top_edges(combined: list, limit: int = TOP_N, max_per_prop_subtype: int = MAX_PER_PROP_SUBTYPE) -> list:
    ranked = sorted(
        combined,
        key=lambda x: (_ranking_sort_key(x)[0], -_ranking_sort_key(x)[1]),
    )
    selected = []
    prop_subtype_counts: dict[str, int] = {}
    pitcher_stat_seen: set[int] = set()
    tier2_pitcher_stat_count = 0
    tier2_pitcher_type_counts: dict[str, int] = {}
    for item in ranked:
        edge_type = item.get("edge_type")
        tier, _ = _ranking_sort_key(item)
        if edge_type == "prop":
            sub = item.get("prop_key") or item.get("prop_subtype") or "prop"
            if prop_subtype_counts.get(sub, 0) >= max_per_prop_subtype:
                continue
            prop_subtype_counts[sub] = prop_subtype_counts.get(sub, 0) + 1
        if edge_type in PITCHER_RATE_EDGE_TYPES:
            pid = item.get("player_id")
            if pid is not None:
                pid_key = int(pid)
                if pid_key in pitcher_stat_seen:
                    continue
                pitcher_stat_seen.add(pid_key)
            if tier == 2:
                if tier2_pitcher_stat_count >= TIER2_PITCHER_STAT_MAX:
                    continue
                if tier2_pitcher_type_counts.get(edge_type, 0) >= TIER2_PITCHER_TYPE_MAX:
                    continue
        selected.append(item)
        if edge_type in PITCHER_RATE_EDGE_TYPES and tier == 2:
            tier2_pitcher_stat_count += 1
            tier2_pitcher_type_counts[edge_type] = tier2_pitcher_type_counts.get(edge_type, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def compute_all_edges(
    games: list,
    batters: list,
    pitchers: list,
    personal_rates: dict,
    sp_avgs: dict | None = None,
) -> list:
    sp_avgs = sp_avgs or {}
    combined = (
        ml_edges_from_games(games)
        + total_edges_from_games(games)
        + k_edges_from_pitchers(pitchers, sp_avgs.get("k", {}))
        + walks_edges_from_pitchers(pitchers, sp_avgs.get("walks", {}))
        + hits_edges_from_pitchers(pitchers, sp_avgs.get("hits", {}))
        + er_edges_from_pitchers(pitchers, sp_avgs.get("er", {}))
        + batter_prop_edges(batters, personal_rates)
    )
    top = select_top_edges(combined)
    cleaned = []
    for item in top:
        mag = _finite_edge_magnitude(item.get("edge_magnitude"))
        if mag is None:
            continue
        item["edge_magnitude"] = mag
        cleaned.append(item)
    for i, item in enumerate(cleaned, start=1):
        item["rank"] = i
    return cleaned
