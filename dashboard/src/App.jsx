import { useEffect, useState, useCallback, useMemo } from "react";

const API = "https://us-central1-mlb-model-491223.cloudfunctions.net/get-daily-predictions";
const MLB_API = "https://statsapi.mlb.com/api/v1.1/game";

function useLiveScore(gameId, status) {
  const [live, setLive] = useState(null);
  const active = status === 'In Progress' || status === 'Warmup' || status === 'Pre-Game';
  useEffect(() => {
    if (!active || !gameId) return;
    const fetch_ = () => {
      fetch(`${MLB_API}/${gameId}/feed/live`)
        .then(r => r.json())
        .then(d => {
          const ls = d?.liveData?.linescore ?? {};
          const st = d?.gameData?.status?.detailedState ?? '';
          setLive({
            awayRuns: ls?.teams?.away?.runs ?? null,
            homeRuns: ls?.teams?.home?.runs ?? null,
            inning: ls?.currentInning ?? null,
            inningHalf: ls?.inningHalf ?? null,
            outs: ls?.outs ?? null,
            status: st,
          });
        })
        .catch(() => {});
    };
    fetch_();
    const interval = setInterval(fetch_, 30000);
    return () => clearInterval(interval);
  }, [gameId, active]);
  return live;
}
const MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule";
const MLB_BOX = "https://statsapi.mlb.com/api/v1/game";
const MLB_PEOPLE = "https://statsapi.mlb.com/api/v1/people";
const today = new Date().toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });
const REFRESH_INTERVAL = 45000;
const LIVESCORE_POLL = 30000;

/** UI palette: light base, model=blue, market=purple, semantic green/red */
const COL = {
  bg: "#F1F5F9",
  card: "#FFFFFF",
  cardInner: "#F8FAFC",
  border: "#E2E8F0",
  model: "#2563EB",
  modelTint: "rgba(37,99,235,0.1)",
  market: "#7C3AED",
  positive: "#16A34A",
  negative: "#DC2626",
  text: "#0F172A",
  textSecondary: "#475569",
  textMuted: "#64748B",
  controlBg: "#FFFFFF",
  controlBorder: "#CBD5E1",
  logoBg: "#E2E8F0",
};

function isGameHours() {
  const now = new Date();
  const pt = new Date(now.toLocaleString("en-US", { timeZone: "America/Los_Angeles" }));
  const h = pt.getHours();
  return h >= 9 && h < 23;
}

const TEAM_IDS = {
  "Arizona Diamondbacks": 109, "Atlanta Braves": 144, "Baltimore Orioles": 110,
  "Boston Red Sox": 111, "Chicago Cubs": 112, "Chicago White Sox": 145,
  "Cincinnati Reds": 113, "Cleveland Guardians": 114, "Colorado Rockies": 115,
  "Detroit Tigers": 116, "Houston Astros": 117, "Kansas City Royals": 118,
  "Los Angeles Angels": 108, "Los Angeles Dodgers": 119, "Miami Marlins": 146,
  "Milwaukee Brewers": 158, "Minnesota Twins": 142, "New York Mets": 121,
  "New York Yankees": 147, "Oakland Athletics": 133, "Philadelphia Phillies": 143,
  "Pittsburgh Pirates": 134, "San Diego Padres": 135, "San Francisco Giants": 137,
  "Seattle Mariners": 136, "St. Louis Cardinals": 138, "Tampa Bay Rays": 139,
  "Texas Rangers": 140, "Toronto Blue Jays": 141, "Washington Nationals": 120,
  "Athletics": 133,
};

/** Standard 3-letter abbreviations for micro charts */
const TEAM_ABBR = {
  "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
  "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
  "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
  "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
  "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
  "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
  "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
  "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
  "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
  "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
  "Athletics": "OAK",
};

function teamAbbr(full) {
  if (!full) return "—";
  return TEAM_ABBR[full] || full.split(/\s+/).pop()?.slice(0, 3).toUpperCase() || "—";
}

function logoUrl(t) {
  const id = TEAM_IDS[t];
  return id ? `https://www.mlbstatic.com/team-logos/${id}.svg` : null;
}

function toAmerican(p) {
  if (!p || p <= 0 || p >= 1) return null;
  return p >= 0.5 ? `-${Math.round(p / (1 - p) * 100)}` : `+${Math.round((1 - p) / p * 100)}`;
}

function fmt(price) {
  if (price === null || price === undefined) return "—";
  const n = parseInt(price, 10);
  return Number.isNaN(n) ? "—" : n > 0 ? `+${n}` : `${n}`;
}

/** Raw implied prob from American odds (no devig) */
function americanToRawProb(am) {
  if (am == null || am === undefined) return null;
  const n = Number(am);
  if (!Number.isFinite(n)) return null;
  if (n > 0) return 100 / (n + 100);
  return (-n) / ((-n) + 100);
}

/** Devigged market % for home / away */
function deviggedMarketPct(homeAm, awayAm) {
  const ph = americanToRawProb(homeAm);
  const pa = americanToRawProb(awayAm);
  if (ph == null || pa == null) return { home: null, away: null };
  const s = ph + pa;
  if (s <= 0) return { home: null, away: null };
  return { home: (ph / s) * 100, away: (pa / s) * 100 };
}

/**
 * O/U display pick from model total vs market line.
 * Do Not Bet (push) only when |line − predicted total| < 0.10; else Over if predicted > line, else Under.
 */
function computeOU(totalPred, line) {
  if (line == null || totalPred == null) return null;
  const L = Number(line);
  const T = Number(totalPred);
  if (!Number.isFinite(L) || !Number.isFinite(T)) return null;
  if (Math.abs(L - T) < 0.10) return "push";
  return T > L ? "over" : "under";
}

function formatFirstPitch(utcStr) {
  if (!utcStr) return null;
  const d = new Date(utcStr);
  const et = d.toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit", hour12: true });
  const pt = d.toLocaleTimeString("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit", hour12: true });
  return `${et} ET / ${pt} PT`;
}

function formatTime(date) {
  if (!date) return null;
  return date.toLocaleTimeString("en-US", {
    timeZone: "America/Los_Angeles",
    hour: "numeric", minute: "2-digit", hour12: true,
  }) + " PT";
}

function ordinal(n) {
  if (n == null) return "";
  const v = n % 100;
  if (v >= 11 && v <= 13) return `${n}th`;
  switch (n % 10) {
    case 1: return `${n}st`;
    case 2: return `${n}nd`;
    case 3: return `${n}rd`;
    default: return `${n}th`;
  }
}

function isLiveStatus(status) {
  const s = (status || "").toLowerCase();
  if (s.includes("final") || s.includes("game over")) return false;
  if (s.includes("completed early")) return false;
  return s.includes("progress") || s.includes("warmup") || s.includes("delayed") || s === "live";
}

/** True when the game is over (includes weather-shortened "Completed Early" and abstract Final). */
function isMlbGameFinished(detailedState, abstractGameState, codedGameState) {
  const abs = (abstractGameState || "").trim().toLowerCase();
  if (abs === "final") return true;
  const code = (codedGameState || "").trim().toUpperCase();
  if (code === "F") return true;
  const d = (detailedState || "").trim().toLowerCase();
  if (d === "final" || d === "game over") return true;
  if (d.includes("completed early")) return true;
  if (d.includes("game over")) return true;
  return false;
}

function isPostponedOrCancelled(detailedState, abstractGameState) {
  const abs = (abstractGameState || "").trim().toLowerCase();
  if (abs.includes("postponed")) return true;
  const d = (detailedState || "").trim().toLowerCase();
  return d.includes("postponed") || d.includes("cancelled") || d.includes("canceled");
}

/** Sort: live/upcoming first, then finals, postponed last. */
function dashboardSortKey(g, liveMap) {
  const lr = liveMap[g.game_id];
  const detailed = lr?.status ?? g.status ?? "";
  const abstract = lr?.abstractGameState ?? null;
  const coded = lr?.codedGameState ?? null;
  if (isPostponedOrCancelled(detailed, abstract)) return 2;
  if (isMlbGameFinished(detailed, abstract, coded)) return 1;
  return 0;
}

/**
 * For finished games, prefer MLB schedule/linescore runs (official final).
 * BigQuery `away_runs` / `home_runs` may be stale snapshots from during the game.
 */
function pickFinishedGameRuns(mlbLinescoreRuns, dbRuns) {
  if (mlbLinescoreRuns != null && mlbLinescoreRuns !== "") {
    const n = Number(mlbLinescoreRuns);
    if (Number.isFinite(n)) return n;
  }
  if (dbRuns != null && dbRuns !== "") {
    const n = Number(dbRuns);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function personName(personLike) {
  if (!personLike || typeof personLike !== "object") return null;
  return personLike.fullName || null;
}

/** balls-strikes, e.g. 2-1 */
function formatPitchCount(balls, strikes) {
  if (balls == null && strikes == null) return "—";
  return `${balls ?? 0}-${strikes ?? 0}`;
}

function mergeLinescoreDetail(target, ls) {
  if (!ls || !target) return;
  const offense = ls.offense || {};
  const defense = ls.defense || {};
  const runnerOn = (base) => base != null && typeof base === "object" && base.id != null;
  const bn = personName(offense.batter);
  const pn = personName(defense.pitcher);
  target.batterName = bn ?? target.batterName ?? null;
  target.pitcherName = pn ?? target.pitcherName ?? null;
  if (ls.balls != null) target.balls = Number(ls.balls);
  if (ls.strikes != null) target.strikes = Number(ls.strikes);
  if (ls.outs != null) target.outs = Number(ls.outs);
  target.onFirst = runnerOn(offense.first);
  target.onSecond = runnerOn(offense.second);
  target.onThird = runnerOn(offense.third);
}

/** Pregame / warmup → season AVG; in progress or final → game scorecard line */
function shouldShowGameBattingLine(status) {
  const s = (status || "").toLowerCase();
  if (s.includes("scheduled") || s.includes("pre-game") || s.includes("warmup")) return false;
  if (s.includes("postponed") || s.includes("cancelled") || s.includes("suspended")) return false;
  return true;
}

/** Build a scorecard string from MLB boxscore stats.batting (game stats). */
function formatGameBattingLine(bat) {
  if (!bat || typeof bat !== "object") return null;
  const summary = bat.summary;
  if (summary != null && String(summary).trim()) return String(summary).trim();

  const ab = Number(bat.atBats ?? bat.atBat ?? 0);
  const h = Number(bat.hits ?? 0);
  const hr = Number(bat.homeRuns ?? 0);
  const t = Number(bat.triples ?? 0);
  const d = Number(bat.doubles ?? 0);
  const bb = Number(bat.baseOnBalls ?? 0);
  const k = Number(bat.strikeOuts ?? 0);
  const rbi = Number(bat.rbi ?? 0);
  const r = Number(bat.runs ?? 0);

  if (ab === 0 && h === 0 && hr === 0 && d === 0 && t === 0) return null;

  let line = `${h}-${ab}`;
  if (hr >= 1) line += hr > 1 ? `, ${hr} HR` : ", HR";
  else if (t >= 1) line += t > 1 ? `, ${t} 3B` : ", 3B";
  else if (d === 1) line += ", 2B";
  else if (d > 1) line += `, ${d}×2B`;

  const tail = [];
  if (rbi > 0) tail.push(`${rbi} RBI`);
  if (r > 0) tail.push(`${r} R`);
  if (bb > 0) tail.push(`${bb} BB`);
  if (k > 0) tail.push(`${k} K`);
  if (tail.length) line += `, ${tail.join(", ")}`;
  return line;
}

function Logo({ team, size = 44 }) {
  const url = logoUrl(team);
  const ini = team ? team.split(" ").map(w => w[0]).slice(-2).join("") : "?";
  if (!url) return (
    <div style={{ width: size, height: size, borderRadius: "50%", background: COL.logoBg, display: "flex", alignItems: "center", justifyContent: "center", fontSize: size * 0.28, fontWeight: 700, color: COL.textMuted, flexShrink: 0 }}>{ini}</div>
  );
  return <img src={url} alt={team} style={{ width: size, height: size, objectFit: "contain", flexShrink: 0 }} onError={e => { e.target.style.display = "none"; }} />;
}

function Pill({ children, color }) {
  const c = {
    green: { bg: "rgba(34,197,94,0.15)", text: COL.positive, bd: "rgba(34,197,94,0.35)" },
    red: { bg: "rgba(239,68,68,0.15)", text: COL.negative, bd: "rgba(239,68,68,0.35)" },
    blue: { bg: COL.modelTint, text: COL.model, bd: "rgba(59,130,246,0.35)" },
    gray: { bg: "rgba(107,114,128,0.12)", text: COL.textMuted, bd: "rgba(107,114,128,0.25)" },
  }[color] || { bg: "rgba(107,114,128,0.12)", text: COL.textMuted, bd: "rgba(107,114,128,0.25)" };
  return (
    <span style={{ background: c.bg, color: c.text, border: `1px solid ${c.bd}`, padding: "3px 10px", borderRadius: 100, fontSize: 11, fontWeight: 700, whiteSpace: "nowrap" }}>
      {children}
    </span>
  );
}

/** Arrow for line movement: up = green, down = red (total line) */
function LineMoveArrow({ before, after }) {
  if (before == null || after == null || Math.abs(after - before) < 0.02) return null;
  const up = after > before;
  return (
    <span style={{ fontSize: 12, marginLeft: 4, color: up ? COL.positive : COL.negative, fontWeight: 800 }}>
      {up ? "▲" : "▼"}
    </span>
  );
}

/** ML implied move: positive delta = home more favored */
function MlOddsArrow({ deltaProb }) {
  if (deltaProb == null || Math.abs(deltaProb) < 0.005) return null;
  const home = deltaProb > 0;
  return (
    <span style={{ fontSize: 11, marginLeft: 4, color: home ? COL.positive : COL.negative, fontWeight: 800 }}>
      {home ? "▲" : "▼"}
    </span>
  );
}

/** Runners on 1st / 2nd / 3rd from MLB linescore.offense (schedule linescore hydrate). */
function BaseDiamond({ onFirst, onSecond, onThird }) {
  const baseDot = (on) => ({
    width: 11,
    height: 11,
    borderRadius: 2,
    transform: "rotate(45deg)",
    background: on ? COL.model : COL.card,
    border: `2px solid ${on ? COL.model : COL.border}`,
    boxSizing: "border-box",
  });
  const occupied = [onFirst && "1st", onSecond && "2nd", onThird && "3rd"].filter(Boolean);
  const label = occupied.length ? `Runners: ${occupied.join(", ")}` : "Bases empty";
  return (
    <div
      role="img"
      aria-label={label}
      style={{ position: "relative", width: 52, height: 48, flexShrink: 0 }}
    >
      <div style={{ position: "absolute", left: "50%", top: 0, transform: "translateX(-50%)", width: 11, height: 11 }}>
        <div style={baseDot(onSecond)} />
      </div>
      <div style={{ position: "absolute", left: 0, top: "50%", transform: "translateY(-50%)", width: 11, height: 11 }}>
        <div style={baseDot(onThird)} />
      </div>
      <div style={{ position: "absolute", right: 0, top: "50%", transform: "translateY(-50%)", width: 11, height: 11 }}>
        <div style={baseDot(onFirst)} />
      </div>
    </div>
  );
}

/** Model (blue) vs market (purple pattern) on one track; percentages 0–100 */
function ProbabilityBar({ modelPct, marketPct }) {
  const m = modelPct != null && Number.isFinite(Number(modelPct)) ? Math.min(100, Math.max(0, Number(modelPct))) : null;
  const mk = marketPct != null && Number.isFinite(Number(marketPct)) ? Math.min(100, Math.max(0, Number(marketPct))) : null;
  if (m == null && mk == null) return null;
  const marketStripe = `repeating-linear-gradient(135deg, ${COL.market} 0px, ${COL.market} 2px, rgba(139,92,246,0.25) 2px, rgba(139,92,246,0.25) 4px)`;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8, padding: "6px 8px", borderRadius: 8, background: COL.cardInner, border: `1px solid ${COL.border}` }}>
      <div style={{ flex: 1, minWidth: 0, height: 9, borderRadius: 5, background: COL.cardInner, position: "relative", overflow: "hidden", border: `1px solid ${COL.border}` }}>
        {m != null && (
          <div style={{
            position: "absolute", left: 0, top: 0, bottom: 0, width: `${m}%`,
            background: COL.model, borderRadius: 4,
          }} />
        )}
        {mk != null && (
          <div style={{
            position: "absolute", left: 0, top: 0, bottom: 0, width: `${mk}%`,
            background: marketStripe, opacity: 0.88, borderRadius: 4,
          }} />
        )}
      </div>
      <div style={{ fontSize: 11, fontWeight: 700, color: COL.text, fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap", lineHeight: 1.2 }}>
        {m != null ? <span style={{ color: COL.model }}>{m.toFixed(1)}%</span> : <span style={{ color: COL.textMuted }}>—</span>}
        {mk != null && (
          <span style={{ color: COL.textMuted, fontWeight: 600 }}>
            {" · "}<span style={{ color: COL.market }}>{mk.toFixed(1)}%</span>
          </span>
        )}
      </div>
    </div>
  );
}

function RunsBarRow({ abbr, value, maxVal }) {
  const v = value != null && Number.isFinite(Number(value)) ? Number(value) : null;
  const max = maxVal > 0 ? maxVal : 1;
  const pct = v != null ? Math.min(100, (v / max) * 100) : 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontSize: 11, fontWeight: 800, color: COL.textSecondary, width: 34, flexShrink: 0, letterSpacing: "0.02em" }}>{abbr}</span>
      <div style={{ flex: 1, minWidth: 0, height: 7, borderRadius: 4, background: COL.border, overflow: "hidden" }}>
        {v != null && (
          <div style={{ width: `${pct}%`, height: "100%", background: COL.model, borderRadius: 4, minWidth: v > 0 ? 2 : 0 }} />
        )}
      </div>
      <span style={{ fontSize: 14, fontWeight: 700, color: COL.text, fontVariantNumeric: "tabular-nums", width: 40, textAlign: "right" }}>
        {v != null ? v.toFixed(1) : "—"}
      </span>
    </div>
  );
}

function useLiveScores(dateStr) {
  const [map, setMap] = useState({});

  const load = useCallback(async () => {
    const url = `${MLB_SCHEDULE}?sportId=1&date=${dateStr}&gameTypes=R&hydrate=linescore`;
    try {
      const res = await fetch(url);
      const data = await res.json();
      const out = {};
      const livePks = [];
      const runnerOn = (base) => base != null && typeof base === "object" && base.id != null;
      for (const d of data.dates || []) {
        for (const g of d.games || []) {
          const pk = g.gamePk;
          const ls = g.linescore;
          const st = g.status?.detailedState || "";
          const teams = ls?.teams;
          const offense = ls?.offense || {};
          const defense = ls?.defense || {};
          out[pk] = {
            status: st,
            abstractGameState: g.status?.abstractGameState ?? null,
            codedGameState: g.status?.codedGameState ?? null,
            awayRuns: teams?.away?.runs,
            homeRuns: teams?.home?.runs,
            currentInning: ls?.currentInning,
            inningState: ls?.inningState,
            venueName: g.venue?.name || g.venue?.default || null,
            outs: ls?.outs != null ? Number(ls.outs) : null,
            onFirst: runnerOn(offense.first),
            onSecond: runnerOn(offense.second),
            onThird: runnerOn(offense.third),
            batterName: personName(offense.batter),
            pitcherName: personName(defense.pitcher),
            balls: ls?.balls != null ? Number(ls.balls) : null,
            strikes: ls?.strikes != null ? Number(ls.strikes) : null,
          };
          if (isLiveStatus(st)) livePks.push(pk);
        }
      }
      await Promise.all(
        livePks.map(async (pk) => {
          try {
            const r = await fetch(`${MLB_BOX}/${pk}/linescore`);
            if (!r.ok) return;
            const j = await r.json();
            const detailLs = j.linescore ?? j;
            if (out[pk]) mergeLinescoreDetail(out[pk], detailLs);
          } catch {
            /* ignore */
          }
        }),
      );
      setMap(out);
    } catch {
      /* ignore */
    }
  }, [dateStr]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    const id = setInterval(() => {
      if (isGameHours()) load();
    }, LIVESCORE_POLL);
    return () => clearInterval(id);
  }, [load]);

  return map;
}

const MLB_ODDS_LIST = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds";
const MLB_EVENTS_LIST = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events";

function mlbEventOddsUrl(eventId, key) {
  return `https://api.the-odds-api.com/v4/sports/baseball_mlb/events/${encodeURIComponent(eventId)}/odds?apiKey=${encodeURIComponent(key)}&regions=us&markets=h2h&oddsFormat=american`;
}

function normTeamName(s) {
  return (s || "").toLowerCase().replace(/\./g, "").trim();
}

function teamNamesMatchOdds(a, b) {
  const na = normTeamName(a);
  const nb = normTeamName(b);
  if (na === nb) return true;
  const la = na.split(/\s+/).pop() || "";
  const lb = nb.split(/\s+/).pop() || "";
  return la.length > 2 && lb.length > 2 && la === lb;
}

function findEventForGame(events, awayTeam, homeTeam) {
  if (!Array.isArray(events)) return null;
  return events.find((ev) => {
    const ea = ev.away_team;
    const eh = ev.home_team;
    return (teamNamesMatchOdds(awayTeam, ea) && teamNamesMatchOdds(homeTeam, eh))
      || (teamNamesMatchOdds(awayTeam, eh) && teamNamesMatchOdds(homeTeam, ea));
  }) ?? null;
}

function h2hPriceForTeam(outcomes, team) {
  if (!Array.isArray(outcomes)) return null;
  const o = outcomes.find((x) => teamNamesMatchOdds(team, x.name));
  const p = o?.price;
  return p != null && Number.isFinite(Number(p)) ? Number(p) : null;
}

function bookmakerSortKey(b) {
  const pref = ["draftkings", "fanduel", "betmgm", "caesars"];
  const i = pref.indexOf(b?.key);
  return i === -1 ? 99 : i;
}

/** Pull h2h prices from any bookmaker on an event payload (list or single-event odds). */
function extractMoneylinesFromEvent(ev, awayTeam, homeTeam) {
  const books = [...(ev.bookmakers || [])].sort((a, b) => bookmakerSortKey(a) - bookmakerSortKey(b));
  for (const book of books) {
    const market = book.markets?.find(m => m.key === "h2h");
    const outs = market?.outcomes || [];
    const away = h2hPriceForTeam(outs, awayTeam);
    const home = h2hPriceForTeam(outs, homeTeam);
    if (away != null || home != null) {
      return { away, home };
    }
  }
  return null;
}

/**
 * Live moneylines via The Odds API. Bulk /odds often drops in-progress games or empty books;
 * we fall back to /events + /events/{id}/odds. Requires VITE_ODDS_API_KEY.
 */
function useLiveMoneylines(awayTeam, homeTeam, enabled) {
  const key = import.meta.env.VITE_ODDS_API_KEY;
  const [lines, setLines] = useState(null);

  useEffect(() => {
    if (!enabled || !key || !awayTeam || !homeTeam) {
      setLines(null);
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const oddsUrl = `${MLB_ODDS_LIST}?apiKey=${encodeURIComponent(key)}&regions=us&markets=h2h&oddsFormat=american`;
        const r1 = await fetch(oddsUrl);
        if (!r1.ok) throw new Error("odds list");
        const oddsList = await r1.json();
        let ev = findEventForGame(oddsList, awayTeam, homeTeam);
        let extracted = ev ? extractMoneylinesFromEvent(ev, awayTeam, homeTeam) : null;
        if (extracted && (extracted.away != null || extracted.home != null)) {
          if (!cancelled) setLines(extracted);
          return;
        }

        const r2 = await fetch(`${MLB_EVENTS_LIST}?apiKey=${encodeURIComponent(key)}`);
        if (!r2.ok) throw new Error("events");
        const eventRows = await r2.json();
        ev = findEventForGame(eventRows, awayTeam, homeTeam);
        if (!ev?.id) {
          if (!cancelled) setLines(null);
          return;
        }

        const r3 = await fetch(mlbEventOddsUrl(ev.id, key));
        if (!r3.ok) throw new Error("event odds");
        const detail = await r3.json();
        extracted = extractMoneylinesFromEvent(detail, awayTeam, homeTeam);
        if (!cancelled) setLines(extracted ?? null);
      } catch {
        if (!cancelled) setLines(null);
      }
    };
    load();
    const id = setInterval(load, 60000);
    return () => { cancelled = true; clearInterval(id); };
  }, [enabled, key, awayTeam, homeTeam]);

  return lines;
}

function homePlateUmpireFromBox(box) {
  const lists = [box?.officials, box?.gameData?.officials].filter(Boolean);
  for (const officials of lists) {
    if (!Array.isArray(officials)) continue;
    for (const o of officials) {
      const typ = o?.officialType ?? o?.official?.officialType;
      const name = o?.official?.fullName ?? o?.official?.name ?? o?.name;
      if (name && (typ === "Home Plate" || typ === "Home plate")) return String(name);
    }
  }
  return null;
}

function useGameEnrichment(gameIds, seasonYear, refreshKey = 0) {
  const [data, setData] = useState({});

  useEffect(() => {
    if (!gameIds.length) return;
    let cancelled = false;

    async function fetchBox(pk) {
      const r = await fetch(`${MLB_BOX}/${pk}/boxscore?hydrate=officials`);
      if (!r.ok) return null;
      return r.json();
    }

    function venueNameFromBox(box) {
      if (!box) return null;
      return (
        box.venues?.[0]?.name
        || box.gameData?.venue?.name
        || box.gameData?.venues?.[0]?.name
        || null
      );
    }

    /** MLB battingOrder: e.g. 100=slot1, 301=slot3 1st sub (same slot stacked in UI) */
    function slotSeqFromBattingOrder(bo) {
      if (bo == null || bo === "") return null;
      const n = parseInt(String(bo), 10);
      if (!Number.isFinite(n)) return null;
      const slot = Math.floor(n / 100);
      const seq = n % 100;
      if (slot < 1 || slot > 9) return null;
      return { slot, seq };
    }

    function parseLineup(side, box) {
      const t = box?.teams?.[side];
      if (!t) return { lineup: [], spId: null, spName: null };
      const batters = t.batters || [];
      const players = t.players || {};
      const rows = [];
      for (const id of batters) {
        const key = `ID${id}`;
        const pl = players[key];
        if (!pl) continue;
        const ss = slotSeqFromBattingOrder(pl.battingOrder);
        if (!ss) continue;
        const bat = pl?.stats?.batting;
        const gameLine = formatGameBattingLine(bat);
        rows.push({
          id,
          name: pl?.person?.fullName || `#${id}`,
          gameLine,
          battingOrder: pl.battingOrder,
          slot: ss.slot,
          seq: ss.seq,
        });
      }
      const bySlot = Array.from({ length: 9 }, () => []);
      for (const r of rows) {
        bySlot[r.slot - 1].push(r);
      }
      for (const arr of bySlot) {
        arr.sort((a, b) => a.seq - b.seq);
      }
      const lineup = bySlot.map((arr, i) => ({
        order: i + 1,
        entries: arr.map((r) => ({
          id: r.id,
          name: r.name,
          gameLine: r.gameLine,
          isSub: r.seq > 0,
        })),
      }));
      const pitchers = t.pitchers || [];
      const spId = pitchers[0] ?? null;
      let spName = null;
      if (spId) {
        const pk = `ID${spId}`;
        spName = players[pk]?.person?.fullName || null;
      }
      return { lineup, spId, spName };
    }

    async function fetchHitterAvgs(personIds) {
      const ids = [...new Set(personIds.filter(Boolean))];
      if (!ids.length) return {};
      const out = {};
      for (let i = 0; i < ids.length; i += 40) {
        const chunk = ids.slice(i, i + 40);
        const url = `${MLB_PEOPLE}?personIds=${chunk.join(",")}&hydrate=stats(group=[hitting],type=[season],season=[${seasonYear}],sportId=1)`;
        const r = await fetch(url);
        if (!r.ok) continue;
        const j = await r.json();
        for (const p of j.people || []) {
          const splits = p.stats?.[0]?.splits || [];
          const st = splits[0]?.stat;
          if (st?.avg != null) {
            const a = Number(st.avg);
            out[p.id] = Number.isFinite(a) ? a.toFixed(3).replace(/^0/, "") : "—";
          }
        }
      }
      return out;
    }

    async function fetchPitcherLine(personIds) {
      const ids = [...new Set(personIds.filter(Boolean))];
      if (!ids.length) return {};
      const url = `${MLB_PEOPLE}?personIds=${ids.join(",")}&hydrate=stats(group=[pitching],type=[season],season=[${seasonYear}],sportId=1)`;
      const r = await fetch(url);
      if (!r.ok) return {};
      const j = await r.json();
      const out = {};
      for (const p of j.people || []) {
        const pid = p.id;
        const splits = p.stats?.[0]?.splits || [];
        const st = splits[0]?.stat;
        if (st) {
          const ip = Number(st.inningsPitched) || 0;
          const k = Number(st.strikeOuts) || 0;
          const k9 = ip > 0 ? ((k / ip) * 9).toFixed(1) : "—";
          out[pid] = {
            era: st.era != null ? Number(st.era).toFixed(2) : "—",
            whip: st.whip != null ? Number(st.whip).toFixed(2) : "—",
            k9,
            gs: st.gamesStarted ?? st.gamesPlayed ?? "—",
          };
        }
      }
      return out;
    }

    (async () => {
      const chunks = [];
      for (let i = 0; i < gameIds.length; i += 4) {
        chunks.push(gameIds.slice(i, i + 4));
      }
      const boxByPk = {};
      for (const chunk of chunks) {
        await Promise.all(
          chunk.map(async pk => {
            const box = await fetchBox(pk);
            if (box && !cancelled) boxByPk[pk] = box;
          })
        );
      }
      if (cancelled) return;
      const spIds = [];
      const batterIds = [];
      const parsed = {};
      for (const pk of gameIds) {
        const box = boxByPk[pk];
        if (!box) continue;
        const away = parseLineup("away", box);
        const home = parseLineup("home", box);
        if (away.spId) spIds.push(away.spId);
        if (home.spId) spIds.push(home.spId);
        for (const slot of away.lineup) {
          for (const e of slot.entries || []) batterIds.push(e.id);
        }
        for (const slot of home.lineup) {
          for (const e of slot.entries || []) batterIds.push(e.id);
        }
        parsed[pk] = {
          away,
          home,
          venueName: venueNameFromBox(box),
          umpireName: homePlateUmpireFromBox(box),
        };
      }
      const [pStats, avgs] = await Promise.all([
        fetchPitcherLine(spIds),
        fetchHitterAvgs(batterIds),
      ]);
      if (cancelled) return;
      for (const pk of Object.keys(parsed)) {
        const a = parsed[pk].away;
        const h = parsed[pk].home;
        const vnu = parsed[pk].venueName;
        const ump = parsed[pk].umpireName;
        parsed[pk] = {
          venueName: vnu,
          umpireName: ump,
          away: {
            ...a,
            stats: a.spId ? pStats[a.spId] : null,
            lineup: a.lineup.map(slot => ({
              ...slot,
              entries: slot.entries.map(en => ({ ...en, avg: avgs[en.id] ?? null })),
            })),
          },
          home: {
            ...h,
            stats: h.spId ? pStats[h.spId] : null,
            lineup: h.lineup.map(slot => ({
              ...slot,
              entries: slot.entries.map(en => ({ ...en, avg: avgs[en.id] ?? null })),
            })),
          },
        };
      }
      setData(parsed);
    })();

    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps -- stable when id list or season changes
  }, [gameIds.join(","), seasonYear, refreshKey]);

  return data;
}

/** Actual total equals betting line (integer lines; .5 lines rarely push). */
function isTotalsPush(totalRuns, line) {
  const t = Number(totalRuns);
  const L = Number(line);
  if (!Number.isFinite(t) || !Number.isFinite(L)) return false;
  return Math.abs(t - L) < 0.001;
}

/** Finished game: 'hit' | 'miss' | 'push' (show dash) | null */
function gradeOuResult(ouRec, totalRuns, line) {
  if (ouRec == null || line == null || totalRuns == null) return null;
  const t = Number(totalRuns);
  const L = Number(line);
  if (!Number.isFinite(t) || !Number.isFinite(L)) return null;
  const r = String(ouRec).toLowerCase();
  const push = isTotalsPush(t, L);
  if (r === "over") {
    if (push) return "push";
    return t > L ? "hit" : "miss";
  }
  if (r === "under") {
    if (push) return "push";
    return t < L ? "hit" : "miss";
  }
  if (r === "push") {
    return push ? "hit" : "miss";
  }
  return null;
}

const LINEUP_CELL_FS = 12;

function LineupTableHalf({ slots, teamTitle, showScorecardBatting }) {
  return (
    <table
      style={{
        width: "100%",
        borderCollapse: "collapse",
        tableLayout: "fixed",
        fontSize: LINEUP_CELL_FS,
      }}
    >
      <colgroup>
        <col style={{ width: 22 }} />
        <col />
        <col style={{ width: "40%" }} />
      </colgroup>
      <thead>
        <tr>
          <th
            colSpan={3}
            style={{
              textAlign: "left",
              fontSize: 11,
              fontWeight: 700,
              color: COL.textSecondary,
              padding: "0 0 8px",
              borderBottom: `1px solid ${COL.border}`,
            }}
          >
            {teamTitle}
          </th>
        </tr>
      </thead>
      <tbody>
        {slots.flatMap((slot) => {
          const entries = slot.entries || [];
          if (!entries.length) {
            return [(
              <tr key={`empty-${slot.order}`}>
                <td style={{ color: COL.textMuted, fontVariantNumeric: "tabular-nums", verticalAlign: "top", padding: "5px 6px 5px 0", borderBottom: `1px solid ${COL.border}`, fontSize: LINEUP_CELL_FS }}>{slot.order}</td>
                <td style={{ color: COL.textMuted, verticalAlign: "top", padding: "5px 8px 5px 0", borderBottom: `1px solid ${COL.border}`, fontSize: LINEUP_CELL_FS }}>—</td>
                <td style={{ color: COL.textMuted, borderBottom: `1px solid ${COL.border}`, fontSize: LINEUP_CELL_FS }}>—</td>
              </tr>
            )];
          }
          const rs = entries.length;
          return entries.map((row, ei) => {
            const cell = showScorecardBatting && row.gameLine
              ? row.gameLine
              : (row.avg != null ? row.avg : "—");
            const sub = row.isSub;
            return (
              <tr key={`${slot.order}-${row.id}-${ei}`}>
                {ei === 0 && (
                  <td
                    rowSpan={rs}
                    style={{
                      color: COL.textMuted,
                      fontVariantNumeric: "tabular-nums",
                      verticalAlign: "top",
                      padding: "5px 6px 5px 0",
                      borderBottom: `1px solid ${COL.border}`,
                      fontSize: LINEUP_CELL_FS,
                    }}
                  >
                    {slot.order}
                  </td>
                )}
                <td
                  style={{
                    color: COL.text,
                    fontWeight: sub ? 500 : 600,
                    verticalAlign: "top",
                    padding: sub ? "4px 8px 4px 10px" : "5px 8px 5px 0",
                    borderBottom: `1px solid ${COL.border}`,
                    fontSize: LINEUP_CELL_FS,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {sub ? <span style={{ color: COL.textMuted, marginRight: 5, fontWeight: 600 }}>↳</span> : null}
                  {row.name}
                </td>
                <td
                  style={{
                    color: showScorecardBatting && row.gameLine ? COL.model : COL.textSecondary,
                    fontVariantNumeric: "tabular-nums",
                    textAlign: "right",
                    verticalAlign: "top",
                    padding: sub ? "4px 0" : "5px 0",
                    borderBottom: `1px solid ${COL.border}`,
                    lineHeight: 1.35,
                    whiteSpace: "normal",
                    wordBreak: "break-word",
                    fontSize: LINEUP_CELL_FS,
                  }}
                >
                  {cell}
                </td>
              </tr>
            );
          });
        })}
      </tbody>
    </table>
  );
}

function RecResultMark({ ok }) {
  if (ok === null || ok === undefined) return null;
  return (
    <span
      aria-label={ok ? "Recommendation hit" : "Recommendation missed"}
      style={{ fontSize: 14, fontWeight: 800, color: ok ? COL.positive : COL.negative, lineHeight: 1, marginLeft: 2 }}
    >
      {ok ? "✓" : "✗"}
    </span>
  );
}

/** O/U: push on actual total = line → muted dash, not ✗ */
function OuRecommendationMark({ result }) {
  if (result == null) return null;
  if (result === "push") {
    return (
      <span aria-label="Push" style={{ fontSize: 14, fontWeight: 600, color: COL.textMuted, lineHeight: 1, marginLeft: 2 }}>
        —
      </span>
    );
  }
  return <RecResultMark ok={result === "hit"} />;
}

function GameCard({ g, live, enrich }) {
  const liveRow = live[g.game_id];
  const detailed = liveRow?.status ?? g.status ?? "";
  const abstract = liveRow?.abstractGameState ?? null;
  const coded = liveRow?.codedGameState ?? null;
  const mlbStatus = detailed;
  const gameFinished = isMlbGameFinished(detailed, abstract, coded);
  const gamePostponed = isPostponedOrCancelled(detailed, abstract);
  const gameLive = !gameFinished && !gamePostponed && (isLiveStatus(mlbStatus) || isLiveStatus(g.status));
  const liveMoneylines = useLiveMoneylines(g.away_team, g.home_team, gameLive);

  const morningH = g.morning_home_price ?? null;
  const morningA = g.morning_away_price ?? null;
  const closingH = g.closing_home_price ?? g.morning_home_price ?? null;
  const closingA = g.closing_away_price ?? g.morning_away_price ?? null;

  const mph = g.closing_p_home ?? g.morning_p_home;
  const rawH = closingH ?? morningH;
  const rawA = closingA ?? morningA;

  const homeML = rawH !== null && rawH !== undefined ? fmt(rawH) : (toAmerican(mph) ?? "—");
  const awayML = rawA !== null && rawA !== undefined ? fmt(rawA) : (toAmerican(mph ? 1 - mph : null) ?? "—");

  const mpFromApi = {
    home: g.market_p_home != null ? Number(g.market_p_home) : null,
    away: g.market_p_away != null ? Number(g.market_p_away) : null,
  };
  const mpDev = deviggedMarketPct(rawH ?? morningH, rawA ?? morningA);
  const marketPHome = mpFromApi.home != null ? mpFromApi.home : mpDev.home;
  const marketPAway = mpFromApi.away != null ? mpFromApi.away : mpDev.away;

  const mornPh = g.morning_p_home != null ? Number(g.morning_p_home) : null;
  const closPh = g.closing_p_home != null ? Number(g.closing_p_home) : null;
  const deltaHomeProb = mornPh != null && closPh != null ? closPh - mornPh : null;
  const deltaAwayProb = deltaHomeProb != null ? -deltaHomeProb : null;

  const ouLineMorning = g.morning_ou_line;
  const ouLineClosing = g.closing_ou_line ?? g.morning_ou_line;
  const ouDisplay = ouLineClosing;

  const homeWins = g.p_win_home > g.p_win_away;
  const awayRunsLive = gameLive
    ? (liveRow?.awayRuns ?? 0)
    : (gameFinished
      ? pickFinishedGameRuns(liveRow?.awayRuns, g.away_runs)
      : null);
  const homeRunsLive = gameLive
    ? (liveRow?.homeRuns ?? 0)
    : (gameFinished
      ? pickFinishedGameRuns(liveRow?.homeRuns, g.home_runs)
      : null);

  let inningLabel = null;
  if (gameLive && liveRow?.currentInning && liveRow?.inningState) {
    inningLabel = `${liveRow.inningState} ${ordinal(liveRow.currentInning)}`;
  }

  const ouRec = computeOU(g.total_runs_pred, ouDisplay ?? ouLineMorning);

  const arFinal = pickFinishedGameRuns(liveRow?.awayRuns, g.away_runs);
  const hrFinal = pickFinishedGameRuns(liveRow?.homeRuns, g.home_runs);
  const totalRunsActual =
    gameFinished && arFinal != null && hrFinal != null && Number.isFinite(arFinal) && Number.isFinite(hrFinal)
      ? arFinal + hrFinal
      : null;
  const ouLineForGrade = ouLineClosing ?? ouLineMorning ?? null;
  const ouResult = gameFinished ? gradeOuResult(ouRec, totalRunsActual, ouLineForGrade) : null;

  const e = enrich[g.game_id];
  const showScorecardBatting = shouldShowGameBattingLine(mlbStatus);

  const teams = [
    {
      team: g.away_team,
      sp: g.away_sp_name,
      spEnrich: e?.away,
      pct: g.p_win_away,
      ml: awayML,
      marketP: marketPAway,
      wins: !homeWins,
      score: awayRunsLive,
      isHome: false,
      mlDelta: deltaAwayProb,
      liveMl: liveMoneylines?.away ?? null,
    },
    {
      team: g.home_team,
      sp: g.home_sp_name,
      spEnrich: e?.home,
      pct: g.p_win_home,
      ml: homeML,
      marketP: marketPHome,
      wins: homeWins,
      score: homeRunsLive,
      isHome: true,
      mlDelta: deltaHomeProb,
      liveMl: liveMoneylines?.home ?? null,
    },
  ];

  const firstPitch = formatFirstPitch(g.first_pitch_utc);
  const venueLabel = liveRow?.venueName || e?.venueName || null;

  return (
    <div style={{
      background: COL.card,
      border: gamePostponed
        ? "2px solid #D97706"
        : gameFinished
          ? "2px solid #000000"
          : `1px solid ${COL.border}`,
      borderRadius: 16,
      marginBottom: 12,
      overflow: "hidden",
      boxShadow: gameFinished || gamePostponed
        ? "0 2px 16px rgba(0,0,0,0.12)"
        : "0 2px 16px rgba(15,23,42,0.08)",
    }}>
      {gamePostponed && (
        <div style={{
          padding: "8px 16px",
          background: "#F59E0B",
          textAlign: "center",
          borderBottom: "1px solid #D97706",
        }}>
          <span style={{
            fontSize: 11,
            fontWeight: 800,
            color: "#FFFFFF",
            letterSpacing: "0.16em",
          }}>
            POSTPONED
          </span>
        </div>
      )}
      {gameFinished && !gamePostponed && (
        <div style={{
          padding: "8px 16px",
          background: "#000000",
          textAlign: "center",
          borderBottom: "1px solid #000000",
        }}>
          <span style={{
            fontSize: 11,
            fontWeight: 800,
            color: "#FFFFFF",
            letterSpacing: "0.16em",
          }}>
            FINAL
          </span>
          {mlbStatus && String(mlbStatus).toLowerCase().includes("completed early") && (
            <div style={{ fontSize: 10, fontWeight: 600, color: "rgba(255,255,255,0.85)", marginTop: 4, letterSpacing: "0.04em" }}>
              {mlbStatus}
            </div>
          )}
        </div>
      )}
      {gameLive && (
        <div style={{ padding: "8px 16px", background: "rgba(220,38,38,0.08)", borderBottom: "1px solid rgba(220,38,38,0.18)" }}>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, fontWeight: 800, color: COL.negative, letterSpacing: "0.06em", opacity: 0.95, flexShrink: 0, paddingTop: 2 }}>LIVE</span>
            <div style={{ flex: 1, minWidth: 0, display: "flex", justifyContent: "center", alignItems: "center", padding: "0 4px" }}>
              <span style={{ fontSize: 22, fontWeight: 800, color: COL.text, textAlign: "center" }}>
                {g.away_team?.split(" ").pop()} {awayRunsLive ?? "—"} — {homeRunsLive ?? "—"} {g.home_team?.split(" ").pop()}
              </span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", flexShrink: 0, textAlign: "right", maxWidth: "42%" }}>
              {inningLabel && (
                <span style={{ fontSize: 12, fontWeight: 600, color: COL.textSecondary }}>{inningLabel}</span>
              )}
              {venueLabel && (
                <span style={{ fontSize: 11, color: COL.textSecondary, fontWeight: 600, marginTop: inningLabel ? 5 : 0 }}>
                  📍 {venueLabel}
                </span>
              )}
            </div>
          </div>
          {liveRow && (
            <div style={{
              marginTop: 10,
              paddingTop: 10,
              borderTop: "1px solid rgba(220,38,38,0.15)",
            }}
            >
              <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 16,
                flexWrap: "wrap",
              }}
              >
                <span style={{ fontSize: 12, fontWeight: 700, color: COL.text, fontVariantNumeric: "tabular-nums" }}>
                  {liveRow.outs != null ? `${liveRow.outs} out${liveRow.outs === 1 ? "" : "s"}` : "—"}
                </span>
                <BaseDiamond
                  onFirst={!!liveRow.onFirst}
                  onSecond={!!liveRow.onSecond}
                  onThird={!!liveRow.onThird}
                />
              </div>
              <div style={{
                marginTop: 10,
                textAlign: "center",
                fontSize: 11,
                color: COL.text,
                lineHeight: 1.55,
                paddingLeft: 8,
                paddingRight: 8,
              }}
              >
                <div style={{ marginBottom: 4 }}>
                  <span style={{ color: COL.textMuted, fontWeight: 700 }}>At bat </span>
                  <span style={{ fontWeight: 600 }}>{liveRow.batterName ?? "—"}</span>
                  <span style={{ color: COL.textMuted, margin: "0 10px" }}>·</span>
                  <span style={{ color: COL.textMuted, fontWeight: 700 }}>Pitching </span>
                  <span style={{ fontWeight: 600 }}>{liveRow.pitcherName ?? "—"}</span>
                </div>
                <div>
                  <span style={{ color: COL.textMuted, fontWeight: 700 }}>Count </span>
                  <span style={{ fontWeight: 700, fontVariantNumeric: "tabular-nums", color: COL.textSecondary }}>
                    {formatPitchCount(liveRow.balls, liveRow.strikes)}
                  </span>
                  <span style={{ color: COL.textMuted, fontSize: 10, marginLeft: 6 }}>balls-strikes</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {firstPitch && !gameFinished && !gameLive && !gamePostponed && (
        <div style={{ padding: "6px 16px 0", display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <span style={{ fontSize: 10, color: COL.textMuted, fontWeight: 600, letterSpacing: "0.05em" }}>⚾ FIRST PITCH</span>
          <span style={{ fontSize: 11, color: COL.model, fontWeight: 600 }}>{firstPitch}</span>
          {venueLabel && (
            <>
              <span style={{ fontSize: 10, color: COL.textMuted }}>·</span>
              <span style={{ fontSize: 11, color: COL.textSecondary, fontWeight: 600 }}>📍 {venueLabel}</span>
            </>
          )}
        </div>
      )}

      {gameLive && (
        <div style={{ padding: "6px 16px 0", fontSize: 10, color: COL.textMuted, lineHeight: 1.45 }}>
          Model P and market P are pre–first pitch. Odds column is pregame (closing). Live column shows current book moneylines.
        </div>
      )}

      <div style={{ padding: "16px 16px 12px" }}>
        {teams.map((r, i) => (
          <div key={i} style={{ marginBottom: i === 0 ? 14 : 0 }}>
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 0 }}>
              <Logo team={r.team} size={38} />
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 14, color: COL.text, fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.team}</div>
                <div style={{ fontSize: 11, color: COL.textMuted, marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {r.spEnrich?.spName || r.sp || "SP TBD"}
                  {r.spEnrich?.stats && (
                    <span style={{ color: COL.textSecondary, marginLeft: 6 }}>
                      ERA {r.spEnrich.stats.era} · WHIP {r.spEnrich.stats.whip} · K/9 {r.spEnrich.stats.k9}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 10, color: COL.textMuted, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 3 }}>Model p</div>
                <span style={{
                  fontSize: 15, fontWeight: 700,
                  color: r.wins ? COL.model : COL.textMuted,
                  background: r.wins ? COL.modelTint : "transparent",
                  padding: r.wins ? "2px 8px" : "0",
                  borderRadius: 100,
                }}>{r.pct != null ? `${Number(r.pct).toFixed(1)}%` : "—"}</span>
              </div>
              <div style={{ textAlign: "right", minWidth: 52 }}>
                <div style={{ fontSize: 10, color: COL.textMuted, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 3 }}>Market p</div>
                <span style={{ fontSize: 14, color: COL.market, fontWeight: 600 }}>
                  {r.marketP != null ? `${Number(r.marketP).toFixed(1)}%` : "—"}
                </span>
              </div>
              <div style={{ textAlign: "right", minWidth: 52 }}>
                <div style={{ fontSize: 10, color: COL.textMuted, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 3 }}>Odds</div>
                <span style={{ fontSize: 15, color: COL.textSecondary, fontWeight: 600, display: "inline-flex", alignItems: "center" }}>
                  {r.ml}
                  <MlOddsArrow deltaProb={r.mlDelta} />
                </span>
              </div>
              {gameLive && (
                <div style={{ textAlign: "right", minWidth: 52 }}>
                  <div style={{ fontSize: 10, color: COL.textMuted, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 3 }}>Live</div>
                  <span style={{ fontSize: 15, color: COL.market, fontWeight: 600 }}>
                    {r.liveMl != null ? fmt(r.liveMl) : "—"}
                  </span>
                </div>
              )}
              {(gameFinished || gameLive) && (
                <div style={{ textAlign: "right", minWidth: 36 }}>
                  <div style={{ fontSize: 10, color: COL.textMuted, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 3 }}>{gameFinished ? "Final" : "R"}</div>
                  <span style={{ fontSize: 15, fontWeight: 700, color: r.score != null && Number(r.score) > Number(i === 0 ? homeRunsLive : awayRunsLive) ? COL.positive : COL.text }}>
                    {r.score != null && r.score !== "" ? Math.round(Number(r.score)) : ""}
                  </span>
                </div>
              )}
            </div>
          </div>
          <ProbabilityBar modelPct={r.pct} marketPct={r.marketP} />
          </div>
        ))}

        {e && (e.away?.lineup?.some(s => s.entries?.length > 0) || e.home?.lineup?.some(s => s.entries?.length > 0)) && (
          <div style={{ marginTop: 14, paddingTop: 12, borderTop: `1px solid ${COL.border}` }}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, marginBottom: 10, flexWrap: "wrap" }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: COL.textMuted, letterSpacing: "0.08em" }}>LINEUPS</div>
              <div style={{ textAlign: "right", lineHeight: 1.4, display: "flex", alignItems: "baseline", justifyContent: "flex-end", gap: 8, flexWrap: "wrap" }}>
                <span style={{
                  fontSize: 9,
                  fontWeight: 700,
                  color: COL.textMuted,
                  letterSpacing: "0.07em",
                  textTransform: "uppercase",
                  flexShrink: 0,
                }}>
                  HP Umpire
                </span>
                <span style={{ fontSize: 11, fontWeight: 500, color: COL.text }}>
                  {e.umpireName ?? "—"}
                </span>
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, alignItems: "start" }}>
              {["away", "home"].map(side => {
                const slots = e[side]?.lineup || [];
                const title = side === "away" ? g.away_team : g.home_team;
                const teamTitle = title?.split(" ").slice(-1)[0] || side;
                return (
                  <div key={side} style={{ minWidth: 0 }}>
                    <LineupTableHalf
                      teamTitle={teamTitle}
                      slots={slots}
                      showScorecardBatting={showScorecardBatting}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <div style={{ background: COL.cardInner, borderTop: `1px solid ${COL.border}`, borderBottom: `1px solid ${COL.border}`, padding: "12px 16px" }}>
        <div style={{ fontSize: 9, color: COL.textMuted, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 8 }}>Runs prediction</div>
        {(() => {
          const a = g.away_runs_pred != null && Number.isFinite(Number(g.away_runs_pred)) ? Number(g.away_runs_pred) : null;
          const h = g.home_runs_pred != null && Number.isFinite(Number(g.home_runs_pred)) ? Number(g.home_runs_pred) : null;
          const maxVal = Math.max(a ?? 0, h ?? 0, 1) * 1.12;
          return (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <RunsBarRow abbr={teamAbbr(g.away_team)} value={g.away_runs_pred} maxVal={maxVal} />
              <RunsBarRow abbr={teamAbbr(g.home_team)} value={g.home_runs_pred} maxVal={maxVal} />
            </div>
          );
        })()}
        <div style={{
          marginTop: 10,
          paddingTop: 10,
          borderTop: `1px solid ${COL.border}`,
          background: COL.modelTint,
          border: "1px solid rgba(37,99,235,0.22)",
          borderRadius: 10,
          padding: "10px 12px",
        }}>
          <div style={{ display: "flex", flexDirection: "row", alignItems: "stretch", gap: 0 }}>
            <div style={{ flex: 1, minWidth: 0, textAlign: "center", paddingRight: 10 }}>
              <div style={{ fontSize: 9, color: COL.textMuted, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 6 }}>Total</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: COL.model, lineHeight: 1.2 }}>{g.total_runs_pred ?? "—"}</div>
              <div style={{ fontSize: 10, color: COL.textMuted, marginTop: 6 }}>
                {ouDisplay != null ? (
                  <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 2, flexWrap: "wrap" }}>
                    line {typeof ouDisplay === "number" ? ouDisplay.toFixed(1) : ouDisplay}
                    {ouLineMorning != null && ouLineClosing != null
                      && Math.abs(Number(ouLineClosing) - Number(ouLineMorning)) >= 0.02 && (
                      <LineMoveArrow before={Number(ouLineMorning)} after={Number(ouLineClosing)} />
                    )}
                  </span>
                ) : "no line"}
              </div>
            </div>
            <div style={{ width: 1, flexShrink: 0, background: COL.border, alignSelf: "stretch", margin: "0 4px" }} aria-hidden />
            <div style={{ flex: 1, minWidth: 0, textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-start", paddingLeft: 10 }}>
              <div style={{ fontSize: 9, color: COL.textMuted, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 12, lineHeight: 1.2 }}>
                O/U recommendation
              </div>
              <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 4, flexWrap: "wrap" }}>
                {ouRec === "over" ? <Pill color="green">Over</Pill>
                  : ouRec === "under" ? <Pill color="red">Under</Pill>
                    : ouRec === "push" ? <Pill color="gray">Do Not Bet</Pill>
                      : <Pill color="gray">—</Pill>}
                {gameFinished && ouResult != null && ouRec !== "push" && <OuRecommendationMark result={ouResult} />}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [date, setDate] = useState(today);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const seasonYear = useMemo(() => parseInt(date.slice(0, 4), 10) || new Date().getFullYear(), [date]);

  const gameIds = useMemo(() => games.map(g => g.game_id).filter(Boolean), [games]);
  const live = useLiveScores(date);

  const needsBoxscorePoll = useMemo(() => games.some(g => {
    const lr = live[g.game_id];
    const detailed = lr?.status ?? g.status ?? "";
    const abstract = lr?.abstractGameState ?? null;
    const coded = lr?.codedGameState ?? null;
    if (isPostponedOrCancelled(detailed, abstract)) return false;
    if (isMlbGameFinished(detailed, abstract, coded)) return false;
    const st = detailed.toLowerCase();
    return st.includes("progress") || st === "live" || st.includes("in progress");
  }), [games, live]);

  const sortedGames = useMemo(() => {
    if (!games.length) return [];
    return [...games].sort((a, b) => {
      const ka = dashboardSortKey(a, live);
      const kb = dashboardSortKey(b, live);
      if (ka !== kb) return ka - kb;
      const ta = a.first_pitch_utc ? new Date(a.first_pitch_utc).getTime() : 0;
      const tb = b.first_pitch_utc ? new Date(b.first_pitch_utc).getTime() : 0;
      return ta - tb;
    });
  }, [games, live]);

  const [boxRefresh, setBoxRefresh] = useState(0);
  useEffect(() => {
    if (!needsBoxscorePoll) return;
    const id = setInterval(() => setBoxRefresh(x => x + 1), 30000);
    return () => clearInterval(id);
  }, [needsBoxscorePoll]);

  const enrich = useGameEnrichment(gameIds, seasonYear, boxRefresh);

  const fetchGames = useCallback((showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true);
    fetch(`${API}?date=${date}`)
      .then(r => r.json())
      .then(d => {
        setGames(d.games || []);
        setLastUpdated(new Date());
        setLoading(false);
        setRefreshing(false);
      })
      .catch(() => { setLoading(false); setRefreshing(false); });
  }, [date]);

  useEffect(() => { setLoading(true); fetchGames(); }, [date, fetchGames]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (isGameHours()) fetchGames(true);
    }, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchGames]);

  return (
    <div style={{ background: COL.bg, minHeight: "100vh", width: "100%", fontFamily: "'SF Pro Display',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" }}>
      <style>{`
        * { box-sizing: border-box; }
        input[type="date"]::-webkit-calendar-picker-indicator { filter: invert(0.35); opacity: 0.7; }
        ::-webkit-scrollbar { height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 2px; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto", padding: "0 12px 32px" }}>

        <div style={{ textAlign: "center", padding: "28px 0 20px" }}>
          <h1 style={{ fontSize: "clamp(24px, 6vw, 34px)", fontWeight: 800, color: COL.text, margin: "0 0 20px", letterSpacing: "-0.03em" }}>
            MLB Game Predictor
          </h1>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, flexWrap: "wrap" }}>
            <input type="date" value={date} onChange={e => setDate(e.target.value)}
              style={{ background: COL.controlBg, border: `1px solid ${COL.controlBorder}`, borderRadius: 10, color: COL.text, padding: "8px 12px", fontSize: 13, outline: "none", fontFamily: "inherit" }} />
            <button onClick={() => fetchGames(true)} style={{
              fontSize: 12, fontWeight: 600, padding: "8px 14px", borderRadius: 10, cursor: "pointer",
              border: `1px solid ${COL.controlBorder}`, background: COL.controlBg, color: COL.textSecondary,
              opacity: refreshing ? 0.5 : 1, transition: "opacity 0.15s",
            }}>
              {refreshing ? "..." : "↺"}
            </button>
          </div>
          {lastUpdated && (
            <p style={{ fontSize: 11, color: COL.textMuted, margin: "10px 0 0" }}>
              Updated {formatTime(lastUpdated)}{isGameHours() ? " · auto-refreshing" : ""}
            </p>
          )}
        </div>

        {loading
          ? <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "3rem" }}>Loading games...</p>
          : games.length === 0
          ? <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "3rem" }}>No games found for {date}</p>
          : sortedGames.map(g => <GameCard key={g.game_id} g={g} live={live} enrich={enrich} />)
        }

        <div style={{ textAlign: "center", padding: "24px 16px 8px", marginTop: 8 }}>
          <p style={{ fontSize: 11, color: COL.textMuted, margin: 0, lineHeight: 1.6 }}>
            ⚠️ For informational and entertainment purposes only. This tool does not constitute
            financial or betting advice. Past model performance does not guarantee future results.
            Please gamble responsibly. Live scores and lineups load from MLB Stats API.
          </p>
        </div>

      </div>
    </div>
  );
}
