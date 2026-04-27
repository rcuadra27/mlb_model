import { useEffect, useState, useCallback, useMemo, useRef, Fragment } from "react";

const API = "https://us-central1-mlb-model-491223.cloudfunctions.net/get-daily-predictions";
const CHAT_API = "https://us-central1-mlb-model-491223.cloudfunctions.net/mlb-agent-chat";
const MLB_API = "https://statsapi.mlb.com/api/v1.1/game";

const MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule";
const MLB_BOX = "https://statsapi.mlb.com/api/v1/game";
const MLB_PEOPLE = "https://statsapi.mlb.com/api/v1/people";
const today = new Date().toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });
const REFRESH_INTERVAL = 45000;

function addDaysToYmd(ymd, delta) {
  const [y, m, d] = ymd.split("-").map((n) => parseInt(n, 10));
  if (!Number.isFinite(y) || !Number.isFinite(m) || !Number.isFinite(d)) return ymd;
  const t = new Date(y, m - 1, d + delta);
  const pad = (n) => String(n).padStart(2, "0");
  return `${t.getFullYear()}-${pad(t.getMonth() + 1)}-${pad(t.getDate())}`;
}

function shortDateLabel(ymd) {
  const [Y, M, D] = ymd.split("-").map((n) => parseInt(n, 10));
  if (!Y) return ymd;
  const dt = new Date(Y, (M || 1) - 1, D || 1);
  return dt.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/Los_Angeles" });
}

function weekdayShort(ymd) {
  const [Y, M, D] = ymd.split("-").map((n) => parseInt(n, 10));
  if (!Y) return "";
  return new Date(Y, (M || 1) - 1, D || 1).toLocaleDateString("en-US", { weekday: "short", timeZone: "America/Los_Angeles" });
}

function getInitialDateFromUrl() {
  try {
    const p = new URLSearchParams(window.location.search).get("date");
    if (p && /^\d{4}-\d{2}-\d{2}$/.test(p)) return p;
  } catch {
    /* ignore */
  }
  return today;
}

function readHashGameId() {
  const m = String(window.location.hash || "").match(/game=(\d+)/);
  return m ? m[1] : null;
}

function formatLeagueRecord(lr) {
  if (!lr || lr.wins == null || lr.losses == null) return null;
  return `${lr.wins}-${lr.losses}`;
}
const LIVESCORE_POLL = 30000;

/** UI palette: light base, model=blue, market=purple, semantic green/red */
const COL = {
  bg: "#F4F7FB",
  pageBorder: "#D8E1EC",
  card: "#FFFFFF",
  cardInner: "#F8FAFC",
  border: "#E2E8F0",
  model: "#2563EB",
  modelTint: "rgba(37,99,235,0.1)",
  market: "#7C3AED",
  marketMuted: "#64748B",
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

/** Primary / soft background for lineup cards & pred-run bars */
function getTeamTheme(fullTeamName) {
  const n = (fullTeamName || "").toLowerCase();
  const fallback = { primary: "#2563EB", soft: "rgba(37,99,235,0.06)", stroke: "rgba(37,99,235,0.25)", onPrimary: "#FFFFFF" };
  if (n.includes("astros")) return { primary: "#EB6E1F", soft: "#FFF7ED", stroke: "rgba(235,110,31,0.38)", onPrimary: "#FFFFFF" };
  if (n.includes("guardians")) return { primary: "#0C2340", soft: "#EEF2F7", stroke: "rgba(12,35,64,0.3)", onPrimary: "#FFFFFF" };
  if (n.includes("yankees")) return { primary: "#0C2340", soft: "#EEF2F7", stroke: "rgba(12,35,64,0.3)", onPrimary: "#FFFFFF" };
  if (n.includes("dodgers")) return { primary: "#005A9C", soft: "#E8F4FC", stroke: "rgba(0,90,156,0.3)", onPrimary: "#FFFFFF" };
  if (n.includes("red sox")) return { primary: "#BD3039", soft: "#FEF2F2", stroke: "rgba(189,48,57,0.3)", onPrimary: "#FFFFFF" };
  if (n.includes("cubs")) return { primary: "#0E3386", soft: "#EEF2FB", stroke: "rgba(14,51,134,0.28)", onPrimary: "#FFFFFF" };
  if (n.includes("tigers")) return { primary: "#0C2340", soft: "#E8EEF5", stroke: "rgba(12,35,64,0.32)", onPrimary: "#FFFFFF" };
  if (n.includes("rockies")) return { primary: "#33006F", soft: "#F3EEFE", stroke: "rgba(51,0,111,0.28)", onPrimary: "#FFFFFF" };
  if (n.includes("braves")) return { primary: "#CE1141", soft: "#FEF2F4", stroke: "rgba(206,17,65,0.28)", onPrimary: "#FFFFFF" };
  if (n.includes("mets")) return { primary: "#FF5910", soft: "#FFF5ED", stroke: "rgba(255,89,16,0.3)", onPrimary: "#FFFFFF" };
  if (n.includes("phillies")) return { primary: "#E81828", soft: "#FEF2F2", stroke: "rgba(232,24,40,0.28)", onPrimary: "#FFFFFF" };
  if (n.includes("cardinals")) return { primary: "#C41E3A", soft: "#FEF2F4", stroke: "rgba(196,30,58,0.28)", onPrimary: "#FFFFFF" };
  if (n.includes("twins")) return { primary: "#002B5C", soft: "#E8EEF5", stroke: "rgba(0,43,92,0.28)", onPrimary: "#FFFFFF" };
  if (n.includes("rangers")) return { primary: "#003278", soft: "#E8EEF8", stroke: "rgba(0,50,120,0.28)", onPrimary: "#FFFFFF" };
  if (n.includes("orioles")) return { primary: "#DF4601", soft: "#FFF5ED", stroke: "rgba(223,70,1,0.3)", onPrimary: "#FFFFFF" };
  return fallback;
}

/** Pred runs: two proportional bars (away vs home). */
function PredRunsBars({ awayTeam, homeTeam, awayPred, homePred }) {
  const a = awayPred != null && awayPred !== "" ? Number(awayPred) : NaN;
  const h = homePred != null && homePred !== "" ? Number(homePred) : NaN;
  const max = Math.max(0.01, Number.isFinite(a) ? a : 0, Number.isFinite(h) ? h : 0);
  const themeAway = getTeamTheme(awayTeam);
  const themeHome = getTeamTheme(homeTeam);
  const rows = [
    { abbr: teamAbbr(awayTeam), v: a, theme: themeAway },
    { abbr: teamAbbr(homeTeam), v: h, theme: themeHome },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {rows.map(({ abbr, v, theme }, i) => {
        const pct = Number.isFinite(v) && max > 0 ? Math.min(100, (v / max) * 100) : 0;
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
            <span style={{ fontSize: 11, fontWeight: 800, color: COL.textSecondary, width: 34, flexShrink: 0, letterSpacing: "0.02em" }}>{abbr}</span>
            <div style={{ flex: 1, minWidth: 0, height: 13, borderRadius: 6, background: "rgba(15, 23, 42, 0.09)", overflow: "hidden" }}>
              <div style={{
                width: `${pct}%`,
                height: "100%",
                borderRadius: 6,
                background: `linear-gradient(90deg, ${theme.primary} 0%, ${theme.primary}CC 100%)`,
                minWidth: pct > 0 ? 3 : 0,
              }}
              />
            </div>
            <span style={{ fontSize: 13, fontWeight: 800, color: COL.text, fontVariantNumeric: "tabular-nums", width: 40, textAlign: "right", flexShrink: 0 }}>
              {Number.isFinite(v) ? v.toFixed(2) : "—"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

const GP_STAT_WRAP = {
  display: "flex",
  flexDirection: "column",
  minWidth: 0,
  alignSelf: "stretch",
  borderRadius: 10,
  overflow: "hidden",
  border: `1px solid ${COL.border}`,
  boxShadow: "0 1px 3px rgba(15, 23, 42, 0.06)",
  background: COL.card,
};
const GP_STAT_HEADER = {
  background: "#1a202c",
  color: "#FFFFFF",
  fontSize: 10,
  fontWeight: 800,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  padding: "8px 8px",
  textAlign: "center",
};
const GP_STAT_BODY = {
  background: "#FFFFFF",
  padding: "10px 10px",
  flex: 1,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  minHeight: 0,
};

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

function fmtMoneyline(price) {
  return fmt(price);
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

/** Split ET/PT for bold styling on the schedule cards. */
function formatFirstPitchParts(utcStr) {
  if (!utcStr) return null;
  const d = new Date(utcStr);
  const et = d.toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit", hour12: true });
  const pt = d.toLocaleTimeString("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit", hour12: true });
  return { et, pt };
}

function formatGameDetailTimestamp(utcStr) {
  if (!utcStr) return null;
  const d = new Date(utcStr);
  const t = d.toLocaleTimeString("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit", hour12: true });
  const ds = d.toLocaleDateString("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", year: "numeric" });
  return `${t} · ${ds}`;
}

function formatRelativeAgo(dateObj) {
  if (!dateObj) return null;
  const s = Math.floor((Date.now() - dateObj.getTime()) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s} sec ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  return formatTime(dateObj);
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
 * Group schedule rows: live, not-yet-started (incl. postponed), final.
 * Mirrors GamesTableRow's gameLive / gameFinished / gamePostponed logic.
 */
function getHomepageGameSection(g, liveMap) {
  const lr = liveMap?.[g.game_id];
  const detailed = lr?.status ?? g.status ?? "";
  const abstract = lr?.abstractGameState ?? null;
  const coded = lr?.codedGameState ?? null;
  const gameFinished = isMlbGameFinished(detailed, abstract, coded);
  const gamePostponed = isPostponedOrCancelled(detailed, abstract);
  const gameLive = !gameFinished && !gamePostponed && (isLiveStatus(detailed) || isLiveStatus(g.status));
  if (gameFinished) return "completed";
  if (gameLive) return "live";
  return "upcoming";
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

/** Model % minus market % — edge pill (compact in metrics strip). */
function EdgeVsMarketPill({ modelPct, marketPct, compact = false }) {
  const fs = compact ? 11 : 13;
  if (modelPct == null || marketPct == null) {
    return <span style={{ fontSize: fs, fontWeight: 700, color: COL.textMuted }}>—</span>;
  }
  const e = Number(modelPct) - Number(marketPct);
  if (!Number.isFinite(e)) return <span style={{ fontSize: fs, fontWeight: 700, color: COL.textMuted }}>—</span>;
  if (Math.abs(e) < 0.2) {
    return (
      <span style={{
        display: "inline-block",
        padding: compact ? "2px 6px" : "3px 8px",
        borderRadius: 999,
        fontSize: compact ? 10 : 11,
        fontWeight: 600,
        color: COL.textMuted,
        background: "rgba(148, 163, 184, 0.14)",
        border: "1px solid rgba(148, 163, 184, 0.3)",
      }}
      >Flat</span>
    );
  }
  const pos = e > 0;
  return (
    <span style={{
      display: "inline-block",
      padding: compact ? "2px 7px" : "3px 9px",
      borderRadius: 999,
      fontSize: compact ? 11 : 12,
      fontWeight: 800,
      fontVariantNumeric: "tabular-nums",
      letterSpacing: "-0.02em",
      background: pos ? "rgba(22, 163, 74, 0.12)" : "rgba(220, 38, 38, 0.1)",
      color: pos ? COL.positive : COL.negative,
      border: `1px solid ${pos ? "rgba(22, 163, 74, 0.35)" : "rgba(220, 38, 38, 0.32)"}`,
    }}
    >
      {pos ? "+" : ""}{e.toFixed(1)}%
    </span>
  );
}

/** MODEL / MARKET / ODDS / EDGE as aligned columns (homepage cards). */
function TeamMetricsColumns({ r, teamIndex, gameLive, gameFinished, awayRunsLive, homeRunsLive, theme }) {
  const th = theme || { primary: COL.model, soft: COL.cardInner, stroke: COL.border, onPrimary: "#FFFFFF" };
  const oppRuns = teamIndex === 0 ? homeRunsLive : awayRunsLive;
  const runsVal = r.score != null && r.score !== "" ? Math.round(Number(r.score)) : null;
  const runsLead = runsVal != null && oppRuns != null && Number(runsVal) > Number(oppRuns);

  const labelStyle = {
    fontSize: 8,
    fontWeight: 800,
    color: COL.textMuted,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    lineHeight: 1,
  };
  const tile = {
    minWidth: 0,
    padding: "6px 6px",
    textAlign: "center",
    background: "#FFFFFF",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
  };

  return (
    <div style={{ flexShrink: 0, display: "flex", flexDirection: "column", gap: 6, minWidth: 240 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, minmax(54px, 1fr))",
          borderRadius: 8,
          border: `1px solid ${COL.border}`,
          overflow: "hidden",
          background: "#FFFFFF",
          boxShadow: "0 1px 3px rgba(15,23,42,0.04)",
        }}
      >
        <div style={{ ...tile, borderRight: `1px solid ${COL.border}`, background: th.soft }}>
          <div style={labelStyle}>Model</div>
          <div style={{
            marginTop: 4,
            fontSize: 17,
            fontWeight: 900,
            fontVariantNumeric: "tabular-nums",
            lineHeight: 1.1,
            letterSpacing: "-0.02em",
            color: th.primary,
            textAlign: "center",
            width: "100%",
          }}
          >
            {r.pct != null ? `${Number(r.pct).toFixed(1)}%` : "—"}
          </div>
        </div>
        <div style={{ ...tile, borderRight: `1px solid ${COL.border}` }}>
          <div style={labelStyle}>Market</div>
          <div style={{
            marginTop: 4,
            fontSize: 16,
            fontWeight: 800,
            fontVariantNumeric: "tabular-nums",
            lineHeight: 1.1,
            letterSpacing: "-0.02em",
            color: COL.marketMuted,
            textAlign: "center",
            width: "100%",
          }}
          >
            {r.marketP != null ? `${Number(r.marketP).toFixed(1)}%` : "—"}
          </div>
        </div>
        <div style={{ ...tile, borderRight: `1px solid ${COL.border}` }}>
          <div style={labelStyle}>Odds</div>
          <div style={{
            marginTop: 4,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 3,
            width: "100%",
            fontSize: 13,
            fontWeight: 800,
            fontVariantNumeric: "tabular-nums",
            color: COL.text,
            lineHeight: 1.1,
          }}
          >
            <span>{r.ml}</span>
            <MlOddsArrow deltaProb={r.mlDelta} />
          </div>
        </div>
        <div style={tile}>
          <div style={labelStyle}>Edge</div>
          <div style={{ marginTop: 4, display: "flex", justifyContent: "center", width: "100%" }}>
            <EdgeVsMarketPill modelPct={r.pct} marketPct={r.marketP} compact />
          </div>
        </div>
      </div>
      {(gameLive || (gameFinished && runsVal != null)) && (
        <div style={{
          display: "flex",
          alignItems: "stretch",
          gap: 6,
          justifyContent: "flex-end",
        }}
        >
          {gameLive && (
            <div style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "4px 10px",
              borderRadius: 6,
              background: "rgba(220,38,38,0.07)",
              border: `1px solid rgba(220,38,38,0.18)`,
              minWidth: 0,
            }}
            >
              <span style={{
                fontSize: 8,
                fontWeight: 800,
                color: COL.negative,
                letterSpacing: "0.1em",
              }}
              >LIVE ML
              </span>
              <span style={{
                fontSize: 12,
                fontWeight: 800,
                color: COL.text,
                fontVariantNumeric: "tabular-nums",
              }}
              >{r.liveMl != null ? fmtMoneyline(r.liveMl) : "—"}</span>
            </div>
          )}
          {(gameFinished || gameLive) && runsVal != null && (
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "4px 10px",
              borderRadius: 6,
              background: gameFinished ? "rgba(234,179,8,0.14)" : th.soft,
              border: `1px solid ${gameFinished ? "rgba(234,179,8,0.35)" : th.stroke}`,
            }}
            >
              <span style={{
                fontSize: 8,
                fontWeight: 800,
                color: COL.textMuted,
                letterSpacing: "0.1em",
              }}
              >{gameFinished ? "FINAL" : "RUNS"}</span>
              <span style={{
                fontSize: 14,
                fontWeight: 900,
                fontVariantNumeric: "tabular-nums",
                color: runsLead ? COL.positive : COL.text,
              }}
              >{runsVal}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Runners on 1st / 2nd / 3rd from MLB linescore.offense (schedule linescore hydrate). */
function BaseDiamond({ onFirst, onSecond, onThird, compact = false }) {
  const d = compact ? 8 : 11;
  const wrapW = compact ? 40 : 52;
  const wrapH = compact ? 36 : 48;
  const baseDot = (on) => ({
    width: d,
    height: d,
    borderRadius: 2,
    transform: "rotate(45deg)",
    background: on ? COL.model : COL.card,
    border: `${compact ? 1.5 : 2}px solid ${on ? COL.model : COL.border}`,
    boxSizing: "border-box",
  });
  const occupied = [onFirst && "1st", onSecond && "2nd", onThird && "3rd"].filter(Boolean);
  const label = occupied.length ? `Runners: ${occupied.join(", ")}` : "Bases empty";
  return (
    <div
      role="img"
      aria-label={label}
      style={{ position: "relative", width: wrapW, height: wrapH, flexShrink: 0 }}
    >
      <div style={{ position: "absolute", left: "50%", top: 0, transform: "translateX(-50%)", width: d, height: d }}>
        <div style={baseDot(onSecond)} />
      </div>
      <div style={{ position: "absolute", left: 0, top: "50%", transform: "translateY(-50%)", width: d, height: d }}>
        <div style={baseDot(onThird)} />
      </div>
      <div style={{ position: "absolute", right: 0, top: "50%", transform: "translateY(-50%)", width: d, height: d }}>
        <div style={baseDot(onFirst)} />
      </div>
    </div>
  );
}

function eraTone(era) {
  const x = parseFloat(era);
  if (!Number.isFinite(x)) return COL.textSecondary;
  if (x >= 5) return COL.negative;
  if (x <= 3.5) return COL.positive;
  return COL.textSecondary;
}

const PITCHER_MONO = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace';

function lineupBatterMatches(lineName, atBatName) {
  if (!lineName || !atBatName) return false;
  return lineName.trim().toLowerCase() === atBatName.trim().toLowerCase();
}

function BatIcon({ size = 15 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden
      style={{ display: "inline-block", verticalAlign: "-3px", marginRight: 4, flexShrink: 0, color: COL.model }}
    >
      <path
        fill="currentColor"
        d="M21.2 2.8c.5-.5.5-1.35 0-1.85s-1.35-.5-1.85 0L2.8 17.6c-.5.5-.5 1.35 0 1.85s1.35.5 1.85 0L21.2 2.8z"
      />
      <ellipse cx="20.8" cy="2.4" rx="2.4" ry="1.5" transform="rotate(42 20.8 2.4)" fill="currentColor" />
    </svg>
  );
}

/** Compact pitcher info bar (not a tall card). */
function PitcherStarterCard({ spName, stats, theme }) {
  const name = spName || "SP TBD";
  const gs = stats?.gs != null && stats.gs !== "—" ? String(stats.gs) : null;
  const th = theme || { primary: COL.model, soft: COL.cardInner, stroke: COL.border, onPrimary: "#FFFFFF" };

  const compact = !!theme;
  const lblSize = compact ? 8 : 9;
  const valSize = compact ? 12 : 14;
  const hdrPad = compact ? "6px 10px" : "10px 12px";
  const bodyPad = compact ? "6px 4px" : "10px 6px";
  const nameSize = compact ? 13 : 14;

  const StatCell = ({ label, value, color }) => (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      flex: 1,
      minWidth: 0,
      padding: "2px 4px",
    }}
    >
      <span style={{
        fontSize: lblSize,
        fontWeight: 800,
        color: COL.textMuted,
        letterSpacing: "0.08em",
      }}
      >{label}</span>
      <span style={{
        marginTop: 2,
        fontSize: valSize,
        fontWeight: 800,
        fontFamily: PITCHER_MONO,
        fontVariantNumeric: "tabular-nums",
        color: color || COL.text,
      }}
      >{value}</span>
    </div>
  );

  return (
    <div style={{
      borderRadius: 10,
      overflow: "hidden",
      background: "#FFFFFF",
      border: `1px solid ${COL.border}`,
      boxShadow: "0 2px 6px rgba(15, 23, 42, 0.06)",
      minWidth: 0,
    }}
    >
      <div style={{
        padding: hdrPad,
        background: `linear-gradient(135deg, ${th.soft} 0%, #FFFFFF 100%)`,
        borderBottom: `1px solid ${COL.border}`,
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
      >
        <div style={{
          width: 3,
          alignSelf: "stretch",
          borderRadius: 2,
          background: th.primary,
        }}
        />
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{
            fontSize: lblSize,
            fontWeight: 800,
            color: COL.textMuted,
            letterSpacing: "0.08em",
          }}
          >STARTING PITCHER
          </div>
          <div style={{
            fontSize: nameSize,
            fontWeight: 800,
            color: COL.text,
            letterSpacing: "-0.01em",
            marginTop: 2,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
          >{name}</div>
        </div>
        {stats && stats.wins != null && stats.losses != null && (
          <span style={{
            fontSize: compact ? 10 : 11,
            fontWeight: 900,
            color: th.onPrimary,
            background: th.primary,
            padding: compact ? "2px 8px" : "3px 10px",
            borderRadius: 999,
            letterSpacing: "0.04em",
            fontVariantNumeric: "tabular-nums",
            boxShadow: `0 1px 3px ${th.stroke}`,
          }}
          >{stats.wins}-{stats.losses}</span>
        )}
      </div>
      {stats ? (
        <div style={{
          display: "flex",
          alignItems: "center",
          padding: bodyPad,
          gap: 0,
        }}
        >
          <StatCell label="ERA" value={stats.era ?? "—"} color={eraTone(stats.era)} />
          <div style={{ width: 1, alignSelf: "stretch", background: COL.border }} />
          <StatCell label="WHIP" value={stats.whip ?? "—"} />
          <div style={{ width: 1, alignSelf: "stretch", background: COL.border }} />
          <StatCell label="K/9" value={stats.k9 ?? "—"} />
          {gs != null && (
            <>
              <div style={{ width: 1, alignSelf: "stretch", background: COL.border }} />
              <StatCell label="GS" value={gs} />
            </>
          )}
        </div>
      ) : (
        <div style={{ fontSize: 11, color: COL.textMuted, padding: compact ? "8px 12px" : "12px 14px" }}>Starter not confirmed</div>
      )}
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
            awayRecord: formatLeagueRecord(g.teams?.away?.leagueRecord),
            homeRecord: formatLeagueRecord(g.teams?.home?.leagueRecord),
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

  // Initial + date-driven schedule fetch; load() updates map state from MLB API.
  // eslint-disable-next-line react-hooks/set-state-in-effect -- async fetch pattern
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

function mlbEventRunlineUrl(eventId, key) {
  return `https://api.the-odds-api.com/v4/sports/baseball_mlb/events/${encodeURIComponent(eventId)}/odds?apiKey=${encodeURIComponent(key)}&regions=us&markets=spreads&oddsFormat=american`;
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
        // The Odds API drops in-progress games from the bulk /odds feed,
        // so for live games go directly to /events + per-event /odds.
        const r2 = await fetch(`${MLB_EVENTS_LIST}?apiKey=${encodeURIComponent(key)}`);
        if (!r2.ok) throw new Error("events");
        const eventRows = await r2.json();
        let ev = findEventForGame(eventRows, awayTeam, homeTeam);

        if (!ev?.id) {
          const oddsUrl = `${MLB_ODDS_LIST}?apiKey=${encodeURIComponent(key)}&regions=us&markets=h2h&oddsFormat=american`;
          const r1 = await fetch(oddsUrl);
          if (r1.ok) {
            const oddsList = await r1.json();
            ev = findEventForGame(oddsList, awayTeam, homeTeam);
            const extracted = ev ? extractMoneylinesFromEvent(ev, awayTeam, homeTeam) : null;
            if (extracted && (extracted.away != null || extracted.home != null)) {
              if (!cancelled) setLines(extracted);
              return;
            }
          }
          if (!cancelled) setLines(null);
          return;
        }

        const r3 = await fetch(mlbEventOddsUrl(ev.id, key));
        if (!r3.ok) {
          if (!cancelled) setLines(null);
          return;
        }
        const detail = await r3.json();
        const extracted = extractMoneylinesFromEvent(detail, awayTeam, homeTeam);
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

/** All sportsbooks with h2h moneylines for both sides (when available). */
function extractAllBooksMoneylines(ev, awayTeam, homeTeam) {
  const books = [...(ev.bookmakers || [])].sort((a, b) => bookmakerSortKey(a) - bookmakerSortKey(b));
  const rows = [];
  for (const book of books) {
    const market = book.markets?.find(m => m.key === "h2h");
    const outs = market?.outcomes || [];
    const away = h2hPriceForTeam(outs, awayTeam);
    const home = h2hPriceForTeam(outs, homeTeam);
    if (away != null || home != null) {
      rows.push({
        key: book.key,
        title: book.title || book.key,
        away,
        home,
      });
    }
  }
  return rows;
}

function extractAllBooksRunline(ev, awayTeam, homeTeam) {
  const books = [...(ev.bookmakers || [])].sort((a, b) => bookmakerSortKey(a) - bookmakerSortKey(b));
  const rows = [];
  for (const book of books) {
    const market = book.markets?.find(m => m.key === "spreads");
    const outs = market?.outcomes || [];
    const awayOut = outs.find(x => teamNamesMatchOdds(awayTeam, x.name));
    const homeOut = outs.find(x => teamNamesMatchOdds(homeTeam, x.name));
    const awayPoint = awayOut?.point != null ? Number(awayOut.point) : null;
    const homePoint = homeOut?.point != null ? Number(homeOut.point) : null;
    const awayPrice = awayOut?.price != null ? Number(awayOut.price) : null;
    const homePrice = homeOut?.price != null ? Number(homeOut.price) : null;
    if (awayPoint != null || homePoint != null) {
      rows.push({ key: book.key, title: book.title || book.key, awayPoint, homePoint, awayPrice, homePrice });
    }
  }
  return rows;
}

function useAllBookRunlines(awayTeam, homeTeam, enabled) {
  const key = import.meta.env.VITE_ODDS_API_KEY;
  const [books, setBooks] = useState(null);
  const [quotaExhausted, setQuotaExhausted] = useState(false);

  useEffect(() => {
    if (!enabled || !key || !awayTeam || !homeTeam) { setBooks(null); return; }
    let cancelled = false;
    const load = async () => {
      try {
        const r1 = await fetch(`${MLB_ODDS_LIST}?apiKey=${encodeURIComponent(key)}&regions=us&markets=spreads&oddsFormat=american`);
        if (r1.status === 401 || r1.status === 429) { if (!cancelled) setQuotaExhausted(true); return; }
        if (!r1.ok) throw new Error("odds list");
        const oddsList = await r1.json();
        let ev = findEventForGame(oddsList, awayTeam, homeTeam);
        let rows = ev ? extractAllBooksRunline(ev, awayTeam, homeTeam) : [];
        if (rows.length) { if (!cancelled) { setBooks(rows); setQuotaExhausted(false); } return; }

        const r2 = await fetch(`${MLB_EVENTS_LIST}?apiKey=${encodeURIComponent(key)}`);
        if (r2.status === 401 || r2.status === 429) { if (!cancelled) setQuotaExhausted(true); return; }
        if (!r2.ok) throw new Error("events");
        const eventRows = await r2.json();
        ev = findEventForGame(eventRows, awayTeam, homeTeam);
        if (!ev?.id) { if (!cancelled) setBooks(null); return; }

        const r3 = await fetch(mlbEventRunlineUrl(ev.id, key));
        if (r3.status === 401 || r3.status === 429) { if (!cancelled) setQuotaExhausted(true); return; }
        if (!r3.ok) throw new Error("event runline");
        const detail = await r3.json();
        rows = extractAllBooksRunline(detail, awayTeam, homeTeam);
        if (!cancelled) { setBooks(rows.length ? rows : null); setQuotaExhausted(false); }
      } catch { if (!cancelled) setBooks(null); }
    };
    load();
    const id = setInterval(load, 120000);
    return () => { cancelled = true; clearInterval(id); };
  }, [enabled, key, awayTeam, homeTeam]);

  return { books, quotaExhausted };
}

function extractAllBooksTotals(ev) {
  const books = [...(ev.bookmakers || [])].sort((a, b) => bookmakerSortKey(a) - bookmakerSortKey(b));
  const rows = [];
  for (const book of books) {
    const market = book.markets?.find(m => m.key === "totals");
    const outs = market?.outcomes || [];
    const over = outs.find(x => x.name === "Over");
    const under = outs.find(x => x.name === "Under");
    const line = over?.point != null ? Number(over.point) : (under?.point != null ? Number(under.point) : null);
    const overPrice = over?.price != null ? Number(over.price) : null;
    const underPrice = under?.price != null ? Number(under.price) : null;
    if (line != null) {
      rows.push({ key: book.key, title: book.title || book.key, line, overPrice, underPrice });
    }
  }
  return rows;
}

function useAllBookTotals(awayTeam, homeTeam, enabled) {
  const key = import.meta.env.VITE_ODDS_API_KEY;
  const [books, setBooks] = useState(null);
  const [quotaExhausted, setQuotaExhausted] = useState(false);

  useEffect(() => {
    if (!enabled || !key || !awayTeam || !homeTeam) { setBooks(null); return; }
    let cancelled = false;
    const load = async () => {
      try {
        const r1 = await fetch(`${MLB_ODDS_LIST}?apiKey=${encodeURIComponent(key)}&regions=us&markets=totals&oddsFormat=american`);
        if (r1.status === 401 || r1.status === 429) { if (!cancelled) setQuotaExhausted(true); return; }
        if (!r1.ok) throw new Error("odds list");
        const oddsList = await r1.json();
        let ev = findEventForGame(oddsList, awayTeam, homeTeam);
        let rows = ev ? extractAllBooksTotals(ev) : [];
        if (rows.length) { if (!cancelled) { setBooks(rows); setQuotaExhausted(false); } return; }

        const r2 = await fetch(`${MLB_EVENTS_LIST}?apiKey=${encodeURIComponent(key)}`);
        if (r2.status === 401 || r2.status === 429) { if (!cancelled) setQuotaExhausted(true); return; }
        if (!r2.ok) throw new Error("events");
        const eventRows = await r2.json();
        ev = findEventForGame(eventRows, awayTeam, homeTeam);
        if (!ev?.id) { if (!cancelled) setBooks(null); return; }

        const r3 = await fetch(`https://api.the-odds-api.com/v4/sports/baseball_mlb/events/${encodeURIComponent(ev.id)}/odds?apiKey=${encodeURIComponent(key)}&regions=us&markets=totals&oddsFormat=american`);
        if (r3.status === 401 || r3.status === 429) { if (!cancelled) setQuotaExhausted(true); return; }
        if (!r3.ok) throw new Error("event totals");
        const detail = await r3.json();
        rows = extractAllBooksTotals(detail);
        if (!cancelled) { setBooks(rows.length ? rows : null); setQuotaExhausted(false); }
      } catch { if (!cancelled) setBooks(null); }
    };
    load();
    const id = setInterval(load, 120000);
    return () => { cancelled = true; clearInterval(id); };
  }, [enabled, key, awayTeam, homeTeam]);

  return { books, quotaExhausted };
}

function useGameWeather(gamePk) {
  const [weather, setWeather] = useState(null);
  useEffect(() => {
    if (!gamePk) {
      queueMicrotask(() => setWeather(null));
      return;
    }
    let cancelled = false;
    fetch(`${MLB_API}/${gamePk}/feed/live`)
      .then(r => r.json())
      .then((d) => {
        if (cancelled) return;
        const wx = d?.gameData?.weather;
        const venue = d?.gameData?.venue;
        const dt = d?.gameData?.datetime?.dateTime || null;
        const out = {
          condition: wx?.condition ?? null,
          temp: wx?.temp ?? null,
          wind: wx?.wind ?? null,
          venueId: venue?.id ?? null,
          venueName: venue?.name ?? null,
          city: venue?.location?.city ?? null,
          state: venue?.location?.stateAbbrev || venue?.location?.state || null,
          lat: venue?.location?.defaultCoordinates?.latitude ?? null,
          lng: venue?.location?.defaultCoordinates?.longitude ?? null,
          startUtc: dt,
        };
        if (!wx && !venue) {
          setWeather(null);
        } else {
          setWeather(out);
        }
      })
      .catch(() => {
        if (!cancelled) setWeather(null);
      });
    return () => { cancelled = true; };
  }, [gamePk]);
  return weather;
}

const WEATHER_CODE_LABEL = {
  0: "Clear", 1: "Mostly Clear", 2: "Partly Cloudy", 3: "Cloudy",
  45: "Fog", 48: "Freezing Fog",
  51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
  61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
  66: "Freezing Rain", 67: "Freezing Rain",
  71: "Light Snow", 73: "Snow", 75: "Heavy Snow", 77: "Snow Grains",
  80: "Rain Showers", 81: "Rain Showers", 82: "Heavy Rain Showers",
  85: "Snow Showers", 86: "Heavy Snow Showers",
  95: "Thunderstorm", 96: "Thunderstorm", 99: "Severe Thunderstorm",
};

function degreesToCompass(deg) {
  if (deg == null || !Number.isFinite(Number(deg))) return null;
  const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  const idx = Math.round(((Number(deg) % 360) / 22.5)) % 16;
  return dirs[idx];
}

function useBallparkForecast(lat, lng, startUtc, hours = 4) {
  const [data, setData] = useState(null);
  useEffect(() => {
    if (lat == null || lng == null) {
      setData(null);
      return;
    }
    let cancelled = false;
    const url = `https://api.open-meteo.com/v1/forecast`
      + `?latitude=${lat}&longitude=${lng}`
      + `&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,weather_code,wind_speed_10m,wind_direction_10m`
      + `&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto`;
    fetch(url)
      .then(r => r.json())
      .then((j) => {
        if (cancelled) return;
        const h = j?.hourly;
        if (!h?.time || !Array.isArray(h.time)) {
          setData(null);
          return;
        }
        const tz = j.timezone || "UTC";
        const allRows = h.time.map((t, i) => ({
          timeLocal: t,
          tempF: Number.isFinite(h.temperature_2m?.[i]) ? Math.round(h.temperature_2m[i]) : null,
          humidity: Number.isFinite(h.relative_humidity_2m?.[i]) ? Math.round(h.relative_humidity_2m[i]) : null,
          precipProb: Number.isFinite(h.precipitation_probability?.[i]) ? Math.round(h.precipitation_probability[i]) : null,
          code: h.weather_code?.[i] ?? null,
          windMph: Number.isFinite(h.wind_speed_10m?.[i]) ? Math.round(h.wind_speed_10m[i]) : null,
          windDeg: Number.isFinite(h.wind_direction_10m?.[i]) ? Math.round(h.wind_direction_10m[i]) : null,
        }));

        let startIdx = 0;
        if (startUtc) {
          const startLocal = new Date(startUtc).toLocaleString("en-US", { timeZone: tz, hour12: false });
          const [d, t] = startLocal.split(", ");
          const [mo, da, yr] = d.split("/");
          const [hh] = t.split(":");
          const yyyy = yr.length === 2 ? `20${yr}` : yr;
          const target = `${yyyy}-${String(mo).padStart(2, "0")}-${String(da).padStart(2, "0")}T${String(hh).padStart(2, "0")}:00`;
          const idx = allRows.findIndex((r) => r.timeLocal.startsWith(target.slice(0, 13)));
          if (idx >= 0) startIdx = idx;
        } else {
          const now = new Date().toISOString().slice(0, 13);
          const idx = allRows.findIndex((r) => r.timeLocal.startsWith(now));
          if (idx >= 0) startIdx = idx;
        }

        const rows = allRows.slice(startIdx, startIdx + hours);
        setData({ timezone: tz, rows });
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => { cancelled = true; };
  }, [lat, lng, startUtc, hours]);
  return data;
}

/** Parse the MLB feed's wind string.
 * Formats seen in the wild: "11 mph, SSE", "11 mph, Out To LF", "Calm", "8 mph, In From RF".
 */
function parseMlbWind(windStr) {
  if (!windStr || typeof windStr !== "string") return null;
  const raw = windStr.trim();
  if (/^calm$/i.test(raw)) return { mph: 0, dir: "Calm", ballpark: null, ou: "Neutral" };
  const mphMatch = raw.match(/(\d+)\s*mph/i);
  const mph = mphMatch ? Number(mphMatch[1]) : null;
  const after = raw.replace(/^\s*\d+\s*mph\s*,?\s*/i, "").trim();
  let ou = "Neutral";
  if (/^out to/i.test(after)) ou = mph != null && mph >= 7 ? "High on Over" : "Slight Over";
  else if (/^in from/i.test(after)) ou = mph != null && mph >= 7 ? "Suppresses Offense" : "Slight Under";
  else if (mph != null && mph >= 15) ou = "Blustery";
  return { mph, dir: after || null, ballpark: /out to|in from/i.test(after) ? after : null, ou };
}

function prettifyBallparkDir(dir) {
  if (!dir) return null;
  return dir
    .replace(/LF/gi, "Left Field")
    .replace(/CF/gi, "Center Field")
    .replace(/RF/gi, "Right Field")
    .replace(/LCF/gi, "Left Center")
    .replace(/RCF/gi, "Right Center");
}

function usePitcherLastStarts(personId, seasonYear, n = 3) {
  const [rows, setRows] = useState(null);
  useEffect(() => {
    if (!personId || !seasonYear) {
      setRows(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const url = `${MLB_PEOPLE}/${personId}/stats?stats=gameLog&group=pitching&season=${seasonYear}&sportId=1`;
        const r = await fetch(url);
        if (!r.ok) throw new Error("gameLog");
        const j = await r.json();
        const splits = j.stats?.[0]?.splits || [];
        const sorted = [...splits].sort((a, b) => String(b.date || "").localeCompare(String(a.date || "")));
        const top = sorted.slice(0, n).map((s) => {
          const st = s.stat || {};
          return {
            date: s.date,
            opponent: s.opponent?.name || "—",
            isHome: !!s.isHome,
            summary: st.summary || null,
            ip: st.inningsPitched ?? "—",
            er: st.earnedRuns != null ? st.earnedRuns : st.runs,
            h: st.hits,
            bb: st.baseOnBalls,
            so: st.strikeOuts,
          };
        });
        if (!cancelled) setRows(top);
      } catch {
        if (!cancelled) setRows(null);
      }
    })();
    return () => { cancelled = true; };
  }, [personId, seasonYear, n]);
  return rows;
}

function packStandingTeam(tr, divisionNameShort) {
  if (!tr) return null;
  const gp = Math.max(1, Number(tr.gamesPlayed) || 1);
  const rs = Number(tr.runsScored) || 0;
  const ra = Number(tr.runsAllowed) || 0;
  const dr = tr.divisionRank ?? tr.leagueRank;
  const rank = dr != null && dr !== "" ? Number(dr) : null;
  let divisionLabel = null;
  if (rank != null && divisionNameShort) {
    const o = rankOrdinal(rank);
    divisionLabel = o ? `${o} ${divisionNameShort}` : null;
  }
  return {
    rank,
    rg: (rs / gp).toFixed(1),
    rag: (ra / gp).toFixed(1),
    divisionLabel,
    wins: tr.wins != null ? Number(tr.wins) : null,
    losses: tr.losses != null ? Number(tr.losses) : null,
  };
}

function rankOrdinal(n) {
  if (n == null || !Number.isFinite(Number(n))) return null;
  const v = Math.floor(Number(n));
  const j = v % 10;
  const k = v % 100;
  if (j === 1 && k !== 11) return `${v}st`;
  if (j === 2 && k !== 12) return `${v}nd`;
  if (j === 3 && k !== 13) return `${v}rd`;
  return `${v}th`;
}

/** Division/league ranks + R/G and RA/G from MLB standings. */
function useTeamStandingsForGame(awayName, homeName, seasonYear, enabled = true) {
  const [data, setData] = useState(null);
  useEffect(() => {
    if (!enabled) {
      queueMicrotask(() => setData(null));
      return;
    }
    const idA = TEAM_IDS[awayName];
    const idH = TEAM_IDS[homeName];
    if (!idA || !idH || !seasonYear) {
      queueMicrotask(() => setData(null));
      return;
    }
    let cancelled = false;
    fetch(`https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=${seasonYear}&group=division&hydrate=division`)
      .then(r => r.json())
      .then((j) => {
        if (cancelled) return;
        const find = (tid) => {
          for (const rec of j.records || []) {
            const divShort = rec.division?.nameShort || rec.division?.name || "";
            for (const tr of rec.teamRecords || []) {
              if (tr.team?.id === tid) return packStandingTeam(tr, divShort);
            }
          }
          return null;
        };
        setData({
          away: find(idA),
          home: find(idH),
        });
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => { cancelled = true; };
  }, [awayName, homeName, seasonYear, enabled]);
  return data;
}

/** Fetch all MLB standings once and return a map keyed by team full name. */
function useAllTeamStandings(seasonYear, enabled = true) {
  const [map, setMap] = useState({});
  useEffect(() => {
    if (!enabled || !seasonYear) return;
    let cancelled = false;
    fetch(`https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=${seasonYear}&group=division&hydrate=division`)
      .then(r => r.json())
      .then((j) => {
        if (cancelled) return;
        const out = {};
        for (const rec of j.records || []) {
          const divShort = rec.division?.nameShort || rec.division?.name || "";
          for (const tr of rec.teamRecords || []) {
            const name = tr.team?.name;
            if (name) out[name] = packStandingTeam(tr, divShort);
          }
        }
        setMap(out);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [seasonYear, enabled]);
  return map;
}

function isScheduleGameFinal(game) {
  const a = game?.status?.abstractGameState;
  const c = game?.status?.codedGameState;
  const d = game?.status?.detailedState || "";
  return a === "Final" || c === "F" || String(d).toLowerCase().includes("final");
}

function resultForTeamInGame(game, teamId) {
  const a = game?.teams?.away?.team?.id;
  const h = game?.teams?.home?.team?.id;
  if (!a || !h || !teamId) return null;
  const side = a === teamId ? "away" : h === teamId ? "home" : null;
  if (!side) return null;
  const my = game.teams[side]?.score;
  const op = game.teams[side === "away" ? "home" : "away"]?.score;
  if (my == null || op == null) return null;
  if (my > op) return "W";
  if (my < op) return "L";
  return "T";
}

/** Last ~10 final games for a team (MLB schedule). */
function useTeamRecentForm(teamName, seasonYear, excludeGamePk) {
  const [rows, setRows] = useState(null);
  const teamId = TEAM_IDS[teamName];
  useEffect(() => {
    if (!teamId || !seasonYear) {
      queueMicrotask(() => setRows(null));
      return;
    }
    let cancelled = false;
    const end = new Date();
    const start = new Date();
    start.setDate(start.getDate() - 120);
    const ds = (d) => d.toISOString().slice(0, 10);
    fetch(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=${teamId}&startDate=${ds(start)}&endDate=${ds(end)}`)
      .then((r) => r.json())
      .then((j) => {
        if (cancelled) return;
        const games = (j.dates || []).flatMap((d) => d.games || []).filter(isScheduleGameFinal);
        games.sort((a, b) => {
          const da = new Date(a.gameDate || a.officialDate || 0).getTime();
          const db = new Date(b.gameDate || b.officialDate || 0).getTime();
          return db - da;
        });
        const ex = excludeGamePk != null ? String(excludeGamePk) : null;
        const out = [];
        for (const g of games) {
          if (ex && String(g.gamePk) === ex) continue;
          const res = resultForTeamInGame(g, teamId);
          if (!res) continue;
          const side = g.teams?.away?.team?.id === teamId ? "away" : "home";
          const opp = g.teams?.[side === "away" ? "home" : "away"]?.team;
          const myR = g.teams?.[side]?.score;
          const opR = g.teams?.[side === "away" ? "home" : "away"]?.score;
          out.push({
            gamePk: g.gamePk,
            date: (g.officialDate || g.gameDate || "").slice(0, 10),
            result: res,
            myRuns: myR,
            oppRuns: opR,
            oppAbbr: teamAbbr(opp?.name || ""),
          });
          if (out.length >= 10) break;
        }
        setRows(out);
      })
      .catch(() => {
        if (!cancelled) setRows([]);
      });
    return () => { cancelled = true; };
  }, [teamId, seasonYear, excludeGamePk]);
  return rows;
}

/** Model win probability may be 0–1 or already 0–100. */
function winProbPercent(p) {
  if (p == null || p === "") return null;
  const x = Number(p);
  if (!Number.isFinite(x)) return null;
  if (x > 1) return x;
  return x * 100;
}

function useMlbGameFeed(gamePk, enabled) {
  const [data, setData] = useState(null);
  useEffect(() => {
    if (!enabled || !gamePk) {
      queueMicrotask(() => setData(null));
      return;
    }
    let cancelled = false;
    const load = () => {
      fetch(`${MLB_API}/${gamePk}/feed/live`)
        .then(r => r.json())
        .then((d) => {
          if (cancelled) return;
          const ls = d?.liveData?.linescore;
          const box = d?.liveData?.boxscore;
          const decisions = d?.liveData?.decisions;
          setData(parseGameFeed(ls, box, decisions));
        })
        .catch(() => {});
    };
    load();
    const id = setInterval(load, 30000);
    return () => { cancelled = true; clearInterval(id); };
  }, [gamePk, enabled]);
  return data;
}

/**
 * Live current at-bat with pitch-by-pitch data (MLB Stats API).
 * Polls every ~10s while `enabled` is true.
 */
function useLiveAtBat(gamePk, enabled) {
  const [atBat, setAtBat] = useState(null);
  useEffect(() => {
    if (!enabled || !gamePk) {
      setAtBat(null);
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const r = await fetch(`${MLB_API}/${gamePk}/feed/live`);
        const d = await r.json();
        if (cancelled) return;
        const cp = d?.liveData?.plays?.currentPlay;
        if (!cp) {
          setAtBat(null);
          return;
        }
        const matchup = cp.matchup || {};
        const events = Array.isArray(cp.playEvents) ? cp.playEvents : [];
        const pitches = events
          .filter((ev) => ev?.isPitch || ev?.type === "pitch")
          .map((ev) => {
            const pd = ev.pitchData || {};
            const coords = pd.coordinates || {};
            return {
              pitchNumber: ev.pitchNumber ?? null,
              pX: typeof coords.pX === "number" ? coords.pX : null,
              pZ: typeof coords.pZ === "number" ? coords.pZ : null,
              zoneTop: typeof pd.strikeZoneTop === "number" ? pd.strikeZoneTop : null,
              zoneBottom: typeof pd.strikeZoneBottom === "number" ? pd.strikeZoneBottom : null,
              startSpeed: typeof pd.startSpeed === "number" ? pd.startSpeed : null,
              typeCode: ev.details?.type?.code || null,
              typeDesc: ev.details?.type?.description || null,
              callCode: ev.details?.call?.code || null,
              callDesc: ev.details?.call?.description || null,
              isBall: !!ev.details?.isBall,
              isStrike: !!ev.details?.isStrike,
              isInPlay: !!ev.details?.isInPlay,
            };
          });
        setAtBat({
          batter: matchup.batter?.fullName || null,
          pitcher: matchup.pitcher?.fullName || null,
          batSide: matchup.batSide?.code || null,
          pitchHand: matchup.pitchHand?.code || null,
          balls: cp.count?.balls ?? null,
          strikes: cp.count?.strikes ?? null,
          outs: cp.count?.outs ?? null,
          inning: cp.about?.inning ?? null,
          halfInning: cp.about?.halfInning ?? null,
          pitches,
        });
      } catch {
        if (!cancelled) setAtBat(null);
      }
    };
    load();
    const t = setInterval(load, 10000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [gamePk, enabled]);
  return atBat;
}

/**
 * Compact strike-zone SVG with pitch locations (pX/pZ in feet, MLB convention).
 * Pitches are colored by outcome: strike (red), ball (blue), in-play (green).
 */
function StrikeZone({ pitches, zoneTop, zoneBottom, batSide }) {
  const W = 130;
  const H = 160;
  // Feet ranges to visualize (a bit beyond the plate on each side)
  const xMin = -1.4;
  const xMax = 1.4;
  const yMin = 0.3;
  const yMax = 4.8;
  const fx = (x) => ((x - xMin) / (xMax - xMin)) * W;
  const fy = (z) => H - ((z - yMin) / (yMax - yMin)) * H;

  const zt = typeof zoneTop === "number" ? zoneTop : 3.4;
  const zb = typeof zoneBottom === "number" ? zoneBottom : 1.5;
  const zoneX1 = fx(-0.83);
  const zoneX2 = fx(0.83);
  const zoneY1 = fy(zt);
  const zoneY2 = fy(zb);

  const pitchColor = (p) => {
    if (p.isInPlay) return "#16A34A";
    if (p.isStrike) return "#DC2626";
    if (p.isBall) return "#2563EB";
    return "#64748B";
  };

  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      style={{ display: "block", flexShrink: 0 }}
      role="img"
      aria-label="Current at-bat strike zone"
    >
      <rect x={0} y={0} width={W} height={H} rx={8} fill="#F8FAFC" stroke="rgba(15,23,42,0.12)" />

      <line x1={0} y1={fy(0)} x2={W} y2={fy(0)} stroke="rgba(15,23,42,0.18)" strokeWidth={1} />

      <rect
        x={zoneX1}
        y={zoneY1}
        width={zoneX2 - zoneX1}
        height={zoneY2 - zoneY1}
        fill="rgba(15,23,42,0.04)"
        stroke="#0F172A"
        strokeWidth={1.2}
      />
      {[1, 2].map((i) => (
        <line
          key={`v${i}`}
          x1={zoneX1 + ((zoneX2 - zoneX1) * i) / 3}
          y1={zoneY1}
          x2={zoneX1 + ((zoneX2 - zoneX1) * i) / 3}
          y2={zoneY2}
          stroke="rgba(15,23,42,0.18)"
          strokeWidth={0.8}
          strokeDasharray="2 2"
        />
      ))}
      {[1, 2].map((i) => (
        <line
          key={`h${i}`}
          x1={zoneX1}
          y1={zoneY1 + ((zoneY2 - zoneY1) * i) / 3}
          x2={zoneX2}
          y2={zoneY1 + ((zoneY2 - zoneY1) * i) / 3}
          stroke="rgba(15,23,42,0.18)"
          strokeWidth={0.8}
          strokeDasharray="2 2"
        />
      ))}

      {batSide && (
        <text
          x={batSide === "L" ? W - 6 : 6}
          y={H - 6}
          textAnchor={batSide === "L" ? "end" : "start"}
          fontSize={8}
          fontWeight={800}
          fill="rgba(15,23,42,0.5)"
          letterSpacing="0.08em"
        >{batSide}HB
        </text>
      )}

      {pitches.map((p, i) => {
        if (p.pX == null || p.pZ == null) return null;
        const cx = fx(p.pX);
        const cy = fy(p.pZ);
        const isLast = i === pitches.length - 1;
        const c = pitchColor(p);
        return (
          <g key={i}>
            <circle
              cx={cx}
              cy={cy}
              r={isLast ? 8 : 6}
              fill={c}
              fillOpacity={isLast ? 0.9 : 0.55}
              stroke={isLast ? "#0F172A" : "#FFFFFF"}
              strokeWidth={isLast ? 1.5 : 1}
            />
            <text
              x={cx}
              y={cy + 3}
              textAnchor="middle"
              fontSize={isLast ? 9 : 8}
              fontWeight={800}
              fill="#FFFFFF"
            >{p.pitchNumber ?? i + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

function parseBattersSide(teamBox) {
  const batters = teamBox?.batters || [];
  const players = teamBox?.players || {};
  const rows = [];
  for (const id of batters) {
    const p = players[`ID${id}`];
    if (!p) continue;
    const gb = p.stats?.batting || {};
    const sb = p.seasonStats?.batting || {};
    const pos = p.position?.abbreviation || p.allPositions?.[0]?.position?.abbreviation || "";
    const name = p.person?.fullName || "";
    rows.push({
      id,
      name,
      pos,
      ab: gb.atBats,
      r: gb.runs,
      h: gb.hits,
      rbi: gb.rbi,
      bb: gb.baseOnBalls,
      k: gb.strikeOuts,
      avg: sb.avg,
      ops: sb.ops,
    });
  }
  const ts = teamBox?.teamStats?.batting;
  if (ts) {
    rows.push({
      isTotal: true,
      id: "totals",
      name: "Totals",
      pos: "",
      ab: ts.atBats,
      r: ts.runs,
      h: ts.hits,
      rbi: ts.rbi,
      bb: ts.baseOnBalls,
      k: ts.strikeOuts,
      avg: null,
      ops: null,
    });
  }
  return rows;
}

function parsePitchingDecisions(decisions, awayBox, homeBox) {
  if (!decisions) return null;
  const players = { ...(awayBox?.players || {}), ...(homeBox?.players || {}) };
  const pick = (d) => {
    if (!d?.id) return null;
    const p = players[`ID${d.id}`];
    const ss = p?.seasonStats?.pitching;
    const last = (d.fullName || "").split(" ").pop() || d.fullName;
    return {
      label: last,
      fullName: d.fullName,
      wins: ss?.wins,
      losses: ss?.losses,
      saves: ss?.saves,
      era: ss?.era != null ? Number(ss.era).toFixed(2) : null,
    };
  };
  return {
    win: pick(decisions.winner),
    loss: pick(decisions.loser),
    save: decisions.save ? pick(decisions.save) : null,
  };
}

function parseGameFeed(linescore, box, decisions) {
  if (!box?.teams) return null;
  const innArr = linescore?.innings || [];
  const innings = innArr.map((inn) => ({
    num: inn.num,
    away: inn.away?.runs,
    home: inn.home?.runs,
  }));
  const awayBox = box.teams.away;
  const homeBox = box.teams.home;
  const rhe = {
    away: {
      r: linescore?.teams?.away?.runs,
      h: linescore?.teams?.away?.hits,
      e: linescore?.teams?.away?.errors,
    },
    home: {
      r: linescore?.teams?.home?.runs,
      h: linescore?.teams?.home?.hits,
      e: linescore?.teams?.home?.errors,
    },
  };
  return {
    innings,
    rhe,
    awayBatters: parseBattersSide(awayBox),
    homeBatters: parseBattersSide(homeBox),
    pitching: parsePitchingDecisions(decisions, awayBox, homeBox),
  };
}

/** Per-book moneylines; same fetch strategy as useLiveMoneylines. */
function useAllBookMoneylines(awayTeam, homeTeam, enabled) {
  const key = import.meta.env.VITE_ODDS_API_KEY;
  const [books, setBooks] = useState(null);
  const [quotaExhausted, setQuotaExhausted] = useState(false);

  useEffect(() => {
    if (!enabled || !key || !awayTeam || !homeTeam) {
      setBooks(null);
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const oddsUrl = `${MLB_ODDS_LIST}?apiKey=${encodeURIComponent(key)}&regions=us&markets=h2h&oddsFormat=american`;
        const r1 = await fetch(oddsUrl);
        if (r1.status === 401 || r1.status === 429) {
          if (!cancelled) setQuotaExhausted(true);
          return;
        }
        if (!r1.ok) throw new Error("odds list");
        const oddsList = await r1.json();
        let ev = findEventForGame(oddsList, awayTeam, homeTeam);
        let rows = ev ? extractAllBooksMoneylines(ev, awayTeam, homeTeam) : [];
        if (rows.length) {
          if (!cancelled) { setBooks(rows); setQuotaExhausted(false); }
          return;
        }

        const r2 = await fetch(`${MLB_EVENTS_LIST}?apiKey=${encodeURIComponent(key)}`);
        if (r2.status === 401 || r2.status === 429) {
          if (!cancelled) setQuotaExhausted(true);
          return;
        }
        if (!r2.ok) throw new Error("events");
        const eventRows = await r2.json();
        ev = findEventForGame(eventRows, awayTeam, homeTeam);
        if (!ev?.id) {
          if (!cancelled) setBooks(null);
          return;
        }

        const r3 = await fetch(mlbEventOddsUrl(ev.id, key));
        if (r3.status === 401 || r3.status === 429) {
          if (!cancelled) setQuotaExhausted(true);
          return;
        }
        if (!r3.ok) throw new Error("event odds");
        const detail = await r3.json();
        rows = extractAllBooksMoneylines(detail, awayTeam, homeTeam);
        if (!cancelled) { setBooks(rows.length ? rows : null); setQuotaExhausted(false); }
      } catch {
        if (!cancelled) setBooks(null);
      }
    };
    load();
    const id = setInterval(load, 120000);
    return () => { cancelled = true; clearInterval(id); };
  }, [enabled, key, awayTeam, homeTeam]);

  return { books, quotaExhausted };
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
            wins: st.wins != null ? Number(st.wins) : null,
            losses: st.losses != null ? Number(st.losses) : null,
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

function LineupTableHalf({ slots, teamTitle, fullTeam = null, theme = null, showScorecardBatting, atBatName = null, compactRows = false, maxSlots = null }) {
  const cellFs = compactRows ? 10 : LINEUP_CELL_FS;
  const padY = compactRows ? 3 : 5;
  const displaySlots = maxSlots != null && maxSlots > 0 ? (slots || []).slice(0, maxSlots) : (slots || []);
  const th = theme || { primary: COL.model, soft: COL.cardInner, stroke: COL.border, onPrimary: "#FFFFFF" };
  const accent = th.primary;
  const onHeader = th.onPrimary || "#FFFFFF";
  const cellBorder = "rgba(15, 23, 42, 0.075)";
  const useThemedCard = !!(fullTeam && theme);
  const rowBd = useThemedCard ? cellBorder : COL.border;

  const inner = (
    <table
      style={{
        width: "100%",
        borderCollapse: "collapse",
        tableLayout: "fixed",
        fontSize: cellFs,
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
            style={useThemedCard ? {
              textAlign: "left",
              fontSize: 11,
              fontWeight: 800,
              color: onHeader,
              padding: "11px 12px",
              background: `linear-gradient(135deg, ${accent} 0%, ${accent}CC 100%)`,
              borderBottom: `1px solid ${th.stroke}`,
            } : {
              textAlign: "left",
              fontSize: 11,
              fontWeight: 700,
              color: COL.textSecondary,
              padding: "0 0 8px",
              borderBottom: `1px solid ${COL.border}`,
            }}
          >
            {useThemedCard ? (
              <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
                <Logo team={fullTeam} size={26} />
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", letterSpacing: "0.02em" }}>
                  {fullTeam}
                </span>
              </div>
            ) : (
              teamTitle
            )}
          </th>
        </tr>
      </thead>
      <tbody style={useThemedCard ? { background: th.soft } : undefined}>
        {displaySlots.flatMap((slot) => {
          const entries = slot.entries || [];
          if (!entries.length) {
            return [(
              <tr key={`empty-${slot.order}`}>
                <td style={{ color: COL.textMuted, fontVariantNumeric: "tabular-nums", verticalAlign: "top", padding: `${padY}px 6px ${padY}px 0`, borderBottom: `1px solid ${rowBd}`, fontSize: cellFs }}>{slot.order}</td>
                <td style={{ color: COL.textMuted, verticalAlign: "top", padding: `${padY}px 8px ${padY}px 0`, borderBottom: `1px solid ${rowBd}`, fontSize: cellFs }}>—</td>
                <td style={{ color: COL.textMuted, borderBottom: `1px solid ${rowBd}`, fontSize: cellFs }}>—</td>
              </tr>
            )];
          }
          const rs = entries.length;
          return entries.map((row, ei) => {
            const cell = showScorecardBatting && row.gameLine
              ? row.gameLine
              : (row.avg != null ? row.avg : "—");
            const sub = row.isSub;
            const statColor = showScorecardBatting && row.gameLine ? COL.model : (useThemedCard && row.avg != null && row.avg !== "—" ? accent : COL.textSecondary);
            return (
              <tr key={`${slot.order}-${row.id}-${ei}`}>
                {ei === 0 && (
                  <td
                    rowSpan={rs}
                    style={{
                      color: COL.textMuted,
                      fontVariantNumeric: "tabular-nums",
                      verticalAlign: "top",
                      padding: `${padY}px 6px ${padY}px 0`,
                      borderBottom: `1px solid ${rowBd}`,
                      fontSize: cellFs,
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
                    padding: sub ? `${Math.max(2, padY - 1)}px 8px ${Math.max(2, padY - 1)}px 10px` : `${padY}px 8px ${padY}px 0`,
                    borderBottom: `1px solid ${rowBd}`,
                    fontSize: cellFs,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {sub ? <span style={{ color: COL.textMuted, marginRight: 5, fontWeight: 600 }}>↳</span> : null}
                  {atBatName && lineupBatterMatches(row.name, atBatName) ? (
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 3, maxWidth: "100%" }}>
                      <BatIcon size={compactRows ? 12 : 13} />
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{row.name}</span>
                    </span>
                  ) : (
                    row.name
                  )}
                </td>
                <td
                  style={{
                    color: statColor,
                    fontVariantNumeric: "tabular-nums",
                    textAlign: "right",
                    verticalAlign: "top",
                    padding: sub ? `${Math.max(2, padY - 1)}px 0` : `${padY}px 0`,
                    borderBottom: `1px solid ${rowBd}`,
                    lineHeight: 1.35,
                    whiteSpace: "normal",
                    wordBreak: "break-word",
                    fontSize: cellFs,
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

  if (!useThemedCard) return inner;

  return (
    <div style={{
      borderRadius: 12,
      overflow: "hidden",
      border: `1px solid ${th.stroke}`,
      boxShadow: "0 2px 10px rgba(15, 23, 42, 0.08)",
      background: "#FFFFFF",
    }}
    >
      <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
        {inner}
      </div>
    </div>
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

function PitcherLastStartsTable({ rows }) {
  if (rows === null) {
    return <div style={{ fontSize: 12, color: COL.textMuted, marginTop: 8 }}>Loading last starts…</div>;
  }
  if (!rows.length) {
    return <div style={{ fontSize: 12, color: COL.textMuted, marginTop: 8 }}>No starts in the game log yet.</div>;
  }
  return (
    <div style={{ overflowX: "auto", marginTop: 8, WebkitOverflowScrolling: "touch" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: PITCHER_MONO }}>
        <thead>
          <tr style={{ color: COL.textMuted, textAlign: "left" }}>
            <th style={{ padding: "4px 8px 6px 0", fontWeight: 700 }}>Date</th>
            <th style={{ padding: "4px 8px", fontWeight: 700 }}>Opp</th>
            <th style={{ padding: "4px 6px", fontWeight: 700 }}>IP</th>
            <th style={{ padding: "4px 6px", fontWeight: 700 }}>ER</th>
            <th style={{ padding: "4px 6px", fontWeight: 700 }}>H</th>
            <th style={{ padding: "4px 6px", fontWeight: 700 }}>BB</th>
            <th style={{ padding: "4px 6px", fontWeight: 700 }}>SO</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.date}-${i}`} style={{ borderTop: `1px solid ${COL.border}` }}>
              <td style={{ padding: "7px 8px 7px 0", color: COL.textSecondary, whiteSpace: "nowrap" }}>{r.date}</td>
              <td style={{ padding: "7px 8px", color: COL.text, maxWidth: 140 }}>
                {r.isHome ? "vs" : "@"}{" "}
                {teamAbbr(r.opponent)}
              </td>
              <td style={{ padding: "7px 6px", color: COL.text, fontVariantNumeric: "tabular-nums" }}>{r.ip ?? "—"}</td>
              <td style={{ padding: "7px 6px", color: COL.text, fontVariantNumeric: "tabular-nums" }}>{r.er != null ? r.er : "—"}</td>
              <td style={{ padding: "7px 6px", color: COL.text, fontVariantNumeric: "tabular-nums" }}>{r.h != null ? r.h : "—"}</td>
              <td style={{ padding: "7px 6px", color: COL.text, fontVariantNumeric: "tabular-nums" }}>{r.bb != null ? r.bb : "—"}</td>
              <td style={{ padding: "7px 6px", color: COL.text, fontVariantNumeric: "tabular-nums" }}>{r.so != null ? r.so : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const DETAIL_NAV = [
  { id: "pitching", label: "Pitching" },
  { id: "lineup", label: "Lineup" },
  { id: "last10", label: "Last 10" },
  { id: "weather", label: "Weather" },
  { id: "umpire", label: "Umpire" },
  { id: "moneyline", label: "Money Line" },
  { id: "runline", label: "Run Line" },
  { id: "totals", label: "O/U Totals" },
];

/** Final score, line score (R/H/E), pitching decisions — MLB feed/live. */
function GameFinalScoreboardCard({
  awayTeamName,
  homeTeamName,
  awayAbbr,
  homeAbbr,
  awayRuns,
  homeRuns,
  awayRec,
  homeRec,
  awayDiv,
  homeDiv,
  feed,
  themeAway: themeAwayProp,
  themeHome: themeHomeProp,
  marginBottom: marginBottomProp = 24,
}) {
  const themeAway = themeAwayProp ?? getTeamTheme(awayTeamName);
  const themeHome = themeHomeProp ?? getTeamTheme(homeTeamName);
  const mb = marginBottomProp;
  const loadingShell = (
    <div style={{
      border: `2px solid #0f172a`,
      borderRadius: 14,
      overflow: "hidden",
      background: COL.card,
      boxShadow: `0 4px 24px rgba(15,23,42,0.1), 0 0 0 1px ${themeAway.stroke}`,
      marginBottom: mb,
    }}
    >
      <div style={{
        height: 5,
        background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)`,
      }}
      />
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
        padding: "14px 18px",
        flexWrap: "wrap",
        background: `linear-gradient(135deg, ${themeAway.soft} 0%, #fff 45%, #fff 55%, ${themeHome.soft} 100%)`,
      }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0, flex: 1 }}>
          <Logo team={awayTeamName} size={48} />
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: COL.text, textTransform: "uppercase", letterSpacing: "0.04em" }}>{awayAbbr}</div>
            <div style={{ fontSize: 12, color: COL.textSecondary, fontWeight: 600 }}>{awayRec || "—"}</div>
            {awayDiv && <div style={{ fontSize: 11, color: COL.textMuted, marginTop: 2 }}>{awayDiv}</div>}
          </div>
          <div style={{ fontSize: 34, fontWeight: 900, color: themeAway.primary, fontVariantNumeric: "tabular-nums", marginLeft: 8 }}>{awayRuns ?? "—"}</div>
        </div>
        <div style={{ fontSize: 13, fontWeight: 900, letterSpacing: "0.35em", color: COL.text }}>FINAL</div>
        <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0, flex: 1, justifyContent: "flex-end" }}>
          <div style={{ fontSize: 34, fontWeight: 900, color: themeHome.primary, fontVariantNumeric: "tabular-nums", marginRight: 8 }}>{homeRuns ?? "—"}</div>
          <Logo team={homeTeamName} size={48} />
          <div style={{ minWidth: 0, textAlign: "right" }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: COL.text, textTransform: "uppercase", letterSpacing: "0.04em" }}>{homeAbbr}</div>
            <div style={{ fontSize: 12, color: COL.textSecondary, fontWeight: 600 }}>{homeRec || "—"}</div>
            {homeDiv && <div style={{ fontSize: 11, color: COL.textMuted, marginTop: 2 }}>{homeDiv}</div>}
          </div>
        </div>
      </div>
      <div style={{ padding: "12px 18px 16px", borderTop: `1px solid ${COL.border}`, background: COL.cardInner }}>
        <p style={{ margin: 0, fontSize: 13, color: COL.textMuted }}>Loading line score & pitching decisions…</p>
      </div>
    </div>
  );
  if (!feed) {
    return loadingShell;
  }
  const { innings, rhe, pitching } = feed;
  const maxInn = Math.max(9, innings.length || 0);
  const innNums = Array.from({ length: maxInn }, (_, i) => i + 1);

  return (
    <div style={{
      border: `2px solid #0f172a`,
      borderRadius: 14,
      overflow: "hidden",
      background: COL.card,
      boxShadow: `0 8px 28px rgba(15,23,42,0.12), 0 0 0 1px ${themeAway.stroke}`,
      marginBottom: mb,
    }}
    >
      <div style={{
        height: 5,
        background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)`,
      }}
      />
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
        padding: "16px 18px",
        flexWrap: "wrap",
        background: `linear-gradient(135deg, ${themeAway.soft} 0%, #fff 45%, #fff 55%, ${themeHome.soft} 100%)`,
      }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0, flex: 1 }}>
          <Logo team={awayTeamName} size={48} />
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: COL.text, textTransform: "uppercase", letterSpacing: "0.04em" }}>{awayAbbr}</div>
            <div style={{ fontSize: 12, color: COL.textSecondary, fontWeight: 600 }}>{awayRec || "—"}</div>
            {awayDiv && <div style={{ fontSize: 11, color: COL.textMuted, marginTop: 2 }}>{awayDiv}</div>}
          </div>
          <div style={{ fontSize: 34, fontWeight: 900, color: themeAway.primary, fontVariantNumeric: "tabular-nums", marginLeft: 8 }}>{awayRuns ?? "—"}</div>
        </div>
        <div style={{ fontSize: 13, fontWeight: 900, letterSpacing: "0.35em", color: COL.text }}>FINAL</div>
        <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0, flex: 1, justifyContent: "flex-end" }}>
          <div style={{ fontSize: 34, fontWeight: 900, color: themeHome.primary, fontVariantNumeric: "tabular-nums", marginRight: 8 }}>{homeRuns ?? "—"}</div>
          <Logo team={homeTeamName} size={48} />
          <div style={{ minWidth: 0, textAlign: "right" }}>
            <div style={{ fontSize: 11, fontWeight: 800, color: COL.text, textTransform: "uppercase", letterSpacing: "0.04em" }}>{homeAbbr}</div>
            <div style={{ fontSize: 12, color: COL.textSecondary, fontWeight: 600 }}>{homeRec || "—"}</div>
            {homeDiv && <div style={{ fontSize: 11, color: COL.textMuted, marginTop: 2 }}>{homeDiv}</div>}
          </div>
        </div>
      </div>
      <div style={{ borderTop: `1px solid ${COL.border}`, display: "flex", flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 280px", padding: "12px 14px", overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontVariantNumeric: "tabular-nums" }}>
            <thead>
              <tr style={{ color: COL.textMuted, fontSize: 10, fontWeight: 700 }}>
                <th style={{ textAlign: "left", padding: "4px 6px" }} />
                {innNums.map((n) => (
                  <th key={n} style={{ padding: "4px 3px", minWidth: 22 }}>{n}</th>
                ))}
                <th style={{ padding: "4px 6px", fontWeight: 900, color: COL.text }}>R</th>
                <th style={{ padding: "4px 6px" }}>H</th>
                <th style={{ padding: "4px 6px" }}>E</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ fontWeight: 800, padding: "6px 6px 6px 0", whiteSpace: "nowrap" }}>{awayAbbr}</td>
                {innNums.map((n) => {
                  const inn = innings.find((x) => x.num === n);
                  const v = inn ? inn.away : undefined;
                  return (
                    <td key={n} style={{ padding: "6px 3px", textAlign: "center" }}>{v != null ? v : ""}</td>
                  );
                })}
                <td style={{ fontWeight: 900, padding: "6px 6px" }}>{rhe.away.r ?? "—"}</td>
                <td style={{ padding: "6px 6px" }}>{rhe.away.h ?? "—"}</td>
                <td style={{ padding: "6px 6px" }}>{rhe.away.e ?? "—"}</td>
              </tr>
              <tr style={{ borderTop: `1px solid ${COL.border}` }}>
                <td style={{ fontWeight: 800, padding: "6px 6px 6px 0", whiteSpace: "nowrap" }}>{homeAbbr}</td>
                {innNums.map((n) => {
                  const inn = innings.find((x) => x.num === n);
                  const raw = inn?.home;
                  const cell = (raw === null || raw === undefined) ? "x" : raw;
                  return (
                    <td key={n} style={{ padding: "6px 3px", textAlign: "center" }}>{cell}</td>
                  );
                })}
                <td style={{ fontWeight: 900, padding: "6px 6px" }}>{rhe.home.r ?? "—"}</td>
                <td style={{ padding: "6px 6px" }}>{rhe.home.h ?? "—"}</td>
                <td style={{ padding: "6px 6px" }}>{rhe.home.e ?? "—"}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div style={{ flex: "0 0 260px", borderLeft: `1px solid ${COL.border}`, padding: "14px 16px", background: COL.cardInner, minWidth: 200 }}>
          {pitching?.win && (
            <div style={{ fontSize: 12, marginBottom: 8, lineHeight: 1.45 }}>
              <span style={{ fontWeight: 800, color: COL.text }}>W</span>
              <span style={{ color: COL.textMuted, margin: "0 6px" }}> </span>
              <span style={{ fontWeight: 800 }}>{pitching.win.label}</span>
              <span style={{ color: COL.textSecondary }}>
                {" "}{pitching.win.wins != null && pitching.win.losses != null ? `${pitching.win.wins}-${pitching.win.losses}` : ""}
                {pitching.win.era != null ? ` | ${pitching.win.era} ERA` : ""}
              </span>
            </div>
          )}
          {pitching?.loss && (
            <div style={{ fontSize: 12, marginBottom: 8, lineHeight: 1.45 }}>
              <span style={{ fontWeight: 800, color: COL.text }}>L</span>
              <span style={{ color: COL.textMuted, margin: "0 6px" }}> </span>
              <span style={{ fontWeight: 800 }}>{pitching.loss.label}</span>
              <span style={{ color: COL.textSecondary }}>
                {" "}{pitching.loss.wins != null && pitching.loss.losses != null ? `${pitching.loss.wins}-${pitching.loss.losses}` : ""}
                {pitching.loss.era != null ? ` | ${pitching.loss.era} ERA` : ""}
              </span>
            </div>
          )}
          {pitching?.save && (
            <div style={{ fontSize: 12, lineHeight: 1.45 }}>
              <span style={{ fontWeight: 800, color: COL.text }}>S</span>
              <span style={{ color: COL.textMuted, margin: "0 6px" }}> </span>
              <span style={{ fontWeight: 800 }}>{pitching.save.label}</span>
              <span style={{ color: COL.textSecondary }}>
                {" "}{pitching.save.saves != null ? `${pitching.save.saves}` : ""}
                {pitching.save.era != null ? ` | ${pitching.save.era} ERA` : ""}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const BATTER_H = { fontSize: 9, fontWeight: 700, color: COL.textMuted, textTransform: "uppercase", padding: "6px 4px", textAlign: "center" };
const BATTER_CELL = { fontSize: 11, padding: "5px 4px", textAlign: "center", fontVariantNumeric: "tabular-nums" };

/** Drop pitcher-only batting rows; keep team totals. */
function filterBattersOnly(rows) {
  if (!rows?.length) return [];
  return rows.filter((r) => r.isTotal || String(r.pos || "").toUpperCase() !== "P");
}

function BattersBoxGrid({ awayAbbr, homeAbbr, awayRows, homeRows, themeAway, themeHome, awayTeamName, homeTeamName }) {
  const cols = [
    { key: "ab", label: "AB" },
    { key: "r", label: "R" },
    { key: "h", label: "H" },
    { key: "rbi", label: "RBI" },
    { key: "bb", label: "BB" },
    { key: "k", label: "K" },
    { key: "avg", label: "AVG" },
    { key: "ops", label: "OPS" },
  ];
  const awayF = filterBattersOnly(awayRows);
  const homeF = filterBattersOnly(homeRows);

  const renderTable = (abbr, teamName, rows, theme) => {
    const th = theme || { primary: COL.border, soft: COL.cardInner, stroke: COL.border, onPrimary: "#FFFFFF" };
    return (
      <div style={{
        minWidth: 0,
        overflowX: "auto",
        borderRadius: 12,
        border: `1px solid ${COL.border}`,
        background: COL.card,
        boxShadow: `0 4px 16px rgba(15,23,42,0.07), 0 0 0 1px ${th.stroke}`,
        overflow: "hidden",
      }}
      >
        <div style={{ height: 4, background: th.primary }} />
        <div style={{
          padding: "12px 14px",
          background: `linear-gradient(135deg, ${th.soft} 0%, #FFFFFF 100%)`,
          display: "flex",
          alignItems: "center",
          gap: 10,
          borderBottom: `1px solid ${COL.border}`,
        }}
        >
          {teamName ? <Logo team={teamName} size={24} /> : null}
          <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.1 }}>
            <span style={{
              fontSize: 10,
              fontWeight: 800,
              color: COL.textMuted,
              letterSpacing: "0.08em",
            }}
            >BATTERS
            </span>
            <span style={{
              fontSize: 14,
              fontWeight: 800,
              color: COL.text,
              letterSpacing: "-0.01em",
              marginTop: 2,
            }}
            >
              {teamName || abbr}
            </span>
          </div>
          <span style={{
            marginLeft: "auto",
            fontSize: 10,
            fontWeight: 900,
            color: th.onPrimary,
            background: th.primary,
            padding: "3px 10px",
            borderRadius: 999,
            letterSpacing: "0.08em",
            boxShadow: `0 1px 3px ${th.stroke}`,
          }}
          >{abbr}</span>
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ background: th.soft, borderBottom: `2px solid ${th.primary}` }}>
              <th style={{
                ...BATTER_H,
                textAlign: "left",
                minWidth: 140,
                color: COL.textSecondary,
                letterSpacing: "0.06em",
              }}
              >Player</th>
              {cols.map((c) => (
                <th
                  key={c.key}
                  style={{
                    ...BATTER_H,
                    color: COL.textSecondary,
                    letterSpacing: "0.06em",
                  }}
                >{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => {
              const isTotal = !!row.isTotal;
              return (
                <tr
                  key={row.id || row.name}
                  style={{
                    borderBottom: isTotal ? `2px solid ${th.primary}` : `1px solid ${COL.border}`,
                    background: isTotal
                      ? `${th.soft}`
                      : ri % 2 === 1
                        ? "rgba(15,23,42,0.02)"
                        : "#FFFFFF",
                  }}
                >
                  <td style={{
                    ...BATTER_CELL,
                    textAlign: "left",
                    fontWeight: isTotal ? 900 : 700,
                    color: COL.text,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    maxWidth: 200,
                    borderLeft: isTotal ? `3px solid ${th.primary}` : "3px solid transparent",
                    paddingLeft: 12,
                  }}
                  >
                    {row.pos ? (
                      <>
                        {row.name}
                        <span style={{
                          marginLeft: 6,
                          fontSize: 9,
                          fontWeight: 800,
                          color: th.primary,
                          background: th.soft,
                          border: `1px solid ${th.stroke}`,
                          padding: "1px 5px",
                          borderRadius: 4,
                          letterSpacing: "0.04em",
                        }}
                        >{row.pos}</span>
                      </>
                    ) : row.name}
                  </td>
                  {cols.map((c) => (
                    <td
                      key={c.key}
                      style={{
                        ...BATTER_CELL,
                        fontWeight: isTotal ? 900 : 500,
                        color: isTotal ? COL.text : COL.textSecondary,
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {row[c.key] != null && row[c.key] !== "" ? row[c.key] : "—"}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
      {renderTable(awayAbbr, awayTeamName, awayF, themeAway)}
      {renderTable(homeAbbr, homeTeamName, homeF, themeHome)}
    </div>
  );
}

const TAB_ACCENT = "#EA5800";

function formatShortGameDate(isoDate) {
  if (!isoDate || !/^\d{4}-\d{2}-\d{2}$/.test(isoDate)) return isoDate || "—";
  const d = new Date(`${isoDate}T12:00:00`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function TeamRecentFormColumn({ teamName, seasonYear, excludeGamePk, theme }) {
  const rows = useTeamRecentForm(teamName, seasonYear, excludeGamePk);
  const th = theme ?? getTeamTheme(teamName);

  const wins = rows ? rows.filter((r) => r.result === "W").length : 0;
  const losses = rows ? rows.filter((r) => r.result === "L").length : 0;

  return (
    <div style={{
      borderRadius: 12,
      border: `1px solid ${COL.border}`,
      background: COL.card,
      boxShadow: `0 4px 16px rgba(15,23,42,0.07), 0 0 0 1px ${th.stroke}`,
      minWidth: 0,
      overflow: "hidden",
    }}
    >
      <div style={{ height: 4, background: th.primary }} />
      <div style={{
        padding: "12px 14px",
        background: `linear-gradient(135deg, ${th.soft} 0%, #FFFFFF 100%)`,
        display: "flex",
        alignItems: "center",
        gap: 10,
        borderBottom: `1px solid ${COL.border}`,
      }}
      >
        <Logo team={teamName} size={26} />
        <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.1, minWidth: 0, flex: 1 }}>
          <span style={{ fontSize: 10, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em" }}>LAST 10</span>
          <span style={{
            fontSize: 14,
            fontWeight: 800,
            color: COL.text,
            letterSpacing: "-0.01em",
            marginTop: 2,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
          >{teamName}</span>
        </div>
        {rows && rows.length > 0 && (
          <span style={{
            fontSize: 11,
            fontWeight: 900,
            color: th.onPrimary,
            background: th.primary,
            padding: "4px 10px",
            borderRadius: 999,
            letterSpacing: "0.04em",
            fontVariantNumeric: "tabular-nums",
            boxShadow: `0 1px 3px ${th.stroke}`,
          }}
          >{wins}-{losses}</span>
        )}
      </div>

      {rows === null && (
        <div style={{ padding: "14px 16px", fontSize: 12, color: COL.textMuted }}>Loading…</div>
      )}
      {rows && rows.length === 0 && (
        <div style={{ padding: "14px 16px", fontSize: 12, color: COL.textMuted }}>No recent final games found.</div>
      )}
      {rows && rows.length > 0 && (
        <div>
          {rows.map((gr, i) => {
            const isW = gr.result === "W";
            const isL = gr.result === "L";
            const resultColor = isW ? COL.positive : isL ? COL.negative : COL.textMuted;
            const resultBg = isW ? "rgba(22,163,74,0.12)" : isL ? "rgba(220,38,38,0.1)" : "rgba(15,23,42,0.06)";
            const resultBorder = isW ? "rgba(22,163,74,0.35)" : isL ? "rgba(220,38,38,0.32)" : "rgba(15,23,42,0.15)";
            return (
              <div
                key={gr.gamePk}
                style={{
                  display: "grid",
                  gridTemplateColumns: "76px 1fr auto auto",
                  alignItems: "center",
                  gap: 10,
                  fontSize: 12,
                  padding: "10px 14px",
                  background: i % 2 === 1 ? "rgba(15,23,42,0.02)" : "#FFFFFF",
                  borderTop: i === 0 ? "none" : `1px solid ${COL.border}`,
                }}
              >
                <span style={{
                  color: COL.textSecondary,
                  fontWeight: 700,
                  fontSize: 11,
                  letterSpacing: "0.02em",
                }}
                >{formatShortGameDate(gr.date)}</span>
                <span style={{
                  color: COL.text,
                  fontWeight: 700,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
                >
                  <span style={{ color: COL.textMuted, fontWeight: 600, marginRight: 4 }}>vs</span>
                  {gr.oppAbbr || "—"}
                </span>
                <span style={{
                  fontSize: 11,
                  fontWeight: 900,
                  color: resultColor,
                  background: resultBg,
                  border: `1px solid ${resultBorder}`,
                  padding: "2px 8px",
                  borderRadius: 6,
                  minWidth: 22,
                  textAlign: "center",
                  letterSpacing: "0.04em",
                }}
                >{gr.result}</span>
                <span style={{
                  color: COL.text,
                  fontVariantNumeric: "tabular-nums",
                  fontWeight: 800,
                  fontSize: 12,
                  minWidth: 40,
                  textAlign: "right",
                }}
                >
                  {gr.myRuns != null && gr.oppRuns != null ? `${gr.myRuns}–${gr.oppRuns}` : "—"}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** Probability implied by an American price (no devig, just inversion). */
function americanImpliedProb(price) {
  if (price == null || price === "") return null;
  const n = Number(price);
  if (!Number.isFinite(n) || n === 0) return null;
  return n > 0 ? 100 / (n + 100) : (-n) / (-n + 100);
}

function SteamMoveCallout({ g, themeAway, themeHome }) {
  const rows = [];
  const push = (team, theme, morning, closing) => {
    if (morning == null || closing == null) return;
    const m = Number(morning);
    const c = Number(closing);
    if (!Number.isFinite(m) || !Number.isFinite(c) || m === c) return;
    const pM = americanImpliedProb(m);
    const pC = americanImpliedProb(c);
    const shortened = pM != null && pC != null ? pC > pM : null;
    rows.push({
      team,
      theme,
      from: m,
      to: c,
      deltaProb: pM != null && pC != null ? (pC - pM) * 100 : null,
      shortened,
    });
  };
  push(g.away_team, themeAway, g.morning_away_price, g.closing_away_price);
  push(g.home_team, themeHome, g.morning_home_price, g.closing_home_price);
  if (!rows.length) return null;

  return (
    <div style={{
      border: `1px solid ${COL.border}`,
      borderRadius: 12,
      overflow: "hidden",
      background: COL.card,
      boxShadow: `0 4px 14px rgba(15,23,42,0.07)`,
      marginTop: 14,
    }}
    >
      <div style={{
        ...GP_STAT_HEADER,
        textAlign: "left",
        padding: "10px 14px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 8,
      }}
      >
        <span>Latest ML move</span>
        <span style={{
          fontSize: 9,
          fontWeight: 700,
          color: "rgba(255,255,255,0.6)",
          letterSpacing: "0.08em",
        }}
        >MORNING → CLOSING</span>
      </div>
      <div style={{ padding: "6px 0" }}>
        {rows.map((row, i) => {
          const arrowColor = row.shortened ? COL.positive : COL.negative;
          const toBg = row.shortened ? "rgba(22,163,74,0.1)" : "rgba(220,38,38,0.08)";
          const toBorder = row.shortened ? "rgba(22,163,74,0.35)" : "rgba(220,38,38,0.32)";
          return (
            <div
              key={row.team}
              style={{
                display: "grid",
                gridTemplateColumns: "4px 28px 1fr auto auto auto auto",
                alignItems: "center",
                gap: 10,
                padding: "10px 14px",
                borderTop: i === 0 ? "none" : `1px solid ${COL.border}`,
              }}
            >
              <div style={{
                width: 4,
                height: 22,
                borderRadius: 2,
                background: row.theme.primary,
              }}
              />
              <Logo team={row.team} size={22} />
              <span style={{ fontSize: 13, fontWeight: 800, color: COL.text, letterSpacing: "-0.01em" }}>
                {row.team}
                <span style={{ fontWeight: 600, color: COL.textMuted, marginLeft: 6, fontSize: 11 }}>ML</span>
              </span>
              <span style={{
                fontSize: 13,
                fontWeight: 700,
                fontVariantNumeric: "tabular-nums",
                color: COL.textSecondary,
                padding: "3px 8px",
                borderRadius: 6,
                background: "rgba(15,23,42,0.05)",
              }}
              >{fmt(row.from)}</span>
              <span style={{ color: arrowColor, fontWeight: 900, fontSize: 14, lineHeight: 1 }}>→</span>
              <span style={{
                fontSize: 14,
                fontWeight: 800,
                fontVariantNumeric: "tabular-nums",
                color: COL.text,
                padding: "3px 10px",
                borderRadius: 6,
                background: toBg,
                border: `1px solid ${toBorder}`,
              }}
              >{fmt(row.to)}</span>
              <span style={{
                fontSize: 11,
                fontWeight: 800,
                color: arrowColor,
                fontVariantNumeric: "tabular-nums",
                minWidth: 44,
                textAlign: "right",
              }}
              >
                {row.deltaProb != null
                  ? `${row.deltaProb > 0 ? "+" : ""}${row.deltaProb.toFixed(1)}%`
                  : ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PregameMoneylineSummary({ g, themeAway, themeHome }) {
  const morningH = g.morning_home_price ?? null;
  const morningA = g.morning_away_price ?? null;
  const closingH = g.closing_home_price ?? morningH;
  const closingA = g.closing_away_price ?? morningA;

  const side = (team, theme, morning, closing, alignRight) => (
    <div style={{
      padding: "14px 16px",
      background: `linear-gradient(${alignRight ? "225deg" : "135deg"}, ${theme.soft} 0%, #FFFFFF 70%)`,
      display: "flex",
      flexDirection: alignRight ? "row-reverse" : "row",
      gap: 12,
      alignItems: "center",
      minWidth: 0,
    }}
    >
      <Logo team={team} size={34} />
      <div style={{ minWidth: 0, flex: 1, textAlign: alignRight ? "right" : "left" }}>
        <div style={{ fontSize: 13, fontWeight: 800, color: COL.text, letterSpacing: "-0.01em", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {team}
        </div>
        <div style={{
          display: "flex",
          flexDirection: alignRight ? "row-reverse" : "row",
          gap: 8,
          marginTop: 6,
          flexWrap: "wrap",
        }}
        >
          <span style={{
            fontSize: 10,
            fontWeight: 800,
            color: COL.textMuted,
            background: "rgba(15,23,42,0.05)",
            padding: "3px 8px",
            borderRadius: 999,
            letterSpacing: "0.04em",
            fontVariantNumeric: "tabular-nums",
          }}
          >OPEN {morning != null ? fmt(morning) : "—"}</span>
          <span style={{
            fontSize: 11,
            fontWeight: 900,
            color: theme.onPrimary,
            background: theme.primary,
            padding: "3px 10px",
            borderRadius: 999,
            letterSpacing: "0.04em",
            fontVariantNumeric: "tabular-nums",
            boxShadow: `0 1px 3px ${theme.stroke}`,
          }}
          >CLOSE {closing != null ? fmt(closing) : "—"}</span>
        </div>
      </div>
    </div>
  );

  return (
    <div style={{
      border: `1px solid ${COL.border}`,
      borderRadius: 12,
      overflow: "hidden",
      boxShadow: `0 4px 16px rgba(15,23,42,0.07), 0 0 0 1px ${themeAway.stroke}`,
      background: COL.card,
    }}
    >
      <div style={{
        height: 4,
        background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)`,
      }}
      />
      <div style={{ ...GP_STAT_HEADER, textAlign: "left", padding: "10px 14px" }}>Pregame moneylines</div>
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1px 1fr",
        minHeight: 92,
      }}
      >
        {side(g.away_team, themeAway, morningA, closingA, false)}
        <div style={{ background: COL.border }} />
        {side(g.home_team, themeHome, morningH, closingH, true)}
      </div>
    </div>
  );
}

function BreakdownDarkCard({ g, standings, awayRecStr, homeRecStr }) {
  const pA = g.p_win_away != null ? Number(g.p_win_away) : null;
  const pH = g.p_win_home != null ? Number(g.p_win_home) : null;
  const sa = standings?.away;
  const sh = standings?.home;
  const aPct = winProbPercent(pA);
  const hPct = winProbPercent(pH);
  const sum = (aPct != null && hPct != null) ? Math.max(0.001, aPct + hPct) : 1;
  const aW = aPct != null ? (aPct / sum) * 100 : 50;
  const hW = hPct != null ? (hPct / sum) * 100 : 50;
  const ar = g.away_runs_pred != null ? Number(g.away_runs_pred).toFixed(2) : "—";
  const hr = g.home_runs_pred != null ? Number(g.home_runs_pred).toFixed(2) : "—";
  const awayShort = g.away_team?.split(" ").pop() || "Away";
  const homeShort = g.home_team?.split(" ").pop() || "Home";

  const statSide = (s, alignRight) => {
    const row = (label, val) => (
      <div
        key={label}
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 12,
          fontSize: 12,
          marginBottom: 8,
          color: "rgba(255,255,255,0.9)",
        }}
      >
        <span style={{ color: "rgba(255,255,255,0.5)" }}>{label}</span>
        <span style={{ fontVariantNumeric: "tabular-nums", fontWeight: 700, textAlign: alignRight ? "right" : "left" }}>{val ?? "—"}</span>
      </div>
    );
    return (
      <>
        {row("R/G", s?.rg ?? null)}
        {row("RA/G", s?.rag ?? null)}
      </>
    );
  };

  return (
    <div style={{
      borderRadius: 14,
      overflow: "hidden",
      border: "1px solid #2d2d2d",
      background: "#141414",
      boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
    }}
    >
      <div style={{ display: "grid", gridTemplateColumns: "1fr minmax(200px, 240px) 1fr", minHeight: 220 }}>
        <div style={{ padding: "18px 16px", borderRight: "1px solid #2a2a2a" }}>
          <div style={{ fontSize: 14, fontWeight: 800, color: "#fff", marginBottom: 4 }}>{g.away_team}</div>
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.5)", marginBottom: 14 }}>{awayRecStr ? `(${awayRecStr})` : ""}</div>
          {statSide(sa, false)}
        </div>
        <div style={{ background: "#1f1f1f", padding: "16px 12px", display: "flex", flexDirection: "column", justifyContent: "center", gap: 14 }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 11, fontWeight: 800 }}>
              <span style={{ color: "#f87171" }}>{aPct != null ? `${aPct.toFixed(1)}%` : "—"}</span>
              <span style={{ color: "#4ade80" }}>{hPct != null ? `${hPct.toFixed(1)}%` : "—"}</span>
            </div>
            <div style={{ display: "flex", height: 12, borderRadius: 6, overflow: "hidden", background: "#333" }}>
              <div style={{ width: `${aW}%`, background: "linear-gradient(90deg,#dc2626,#b91c1c)" }} />
              <div style={{ width: `${hW}%`, background: "linear-gradient(90deg,#15803d,#22c55e)" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, fontSize: 11, fontWeight: 700 }}>
              <span style={{ color: "#f87171" }}>{awayShort} win</span>
              <span style={{ color: "#4ade80" }}>{homeShort} win</span>
            </div>
          </div>
          <div style={{ background: "#141414", borderRadius: 8, padding: "10px 8px", display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ flex: 1, textAlign: "center", background: "#fff", color: "#111", borderRadius: 6, padding: "8px 4px", fontWeight: 800, fontSize: 15, fontVariantNumeric: "tabular-nums" }}>{ar}</div>
            <div style={{ fontSize: 10, fontWeight: 800, color: "rgba(255,255,255,0.85)", letterSpacing: "0.12em", textAlign: "center", flex: 1.2 }}>PROJECTED RUNS</div>
            <div style={{ flex: 1, textAlign: "center", background: "#fff", color: "#111", borderRadius: 6, padding: "8px 4px", fontWeight: 800, fontSize: 15, fontVariantNumeric: "tabular-nums" }}>{hr}</div>
          </div>
        </div>
        <div style={{ padding: "18px 16px", borderLeft: "1px solid #2a2a2a" }}>
          <div style={{ fontSize: 14, fontWeight: 800, color: "#fff", marginBottom: 4, textAlign: "right" }}>{g.home_team}</div>
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.5)", marginBottom: 14, textAlign: "right" }}>{homeRecStr ? `(${homeRecStr})` : ""}</div>
          {statSide(sh, true)}
        </div>
      </div>
    </div>
  );
}

function GameDetailPage({
  g,
  liveRow,
  enrich,
  seasonYear,
  onBack,
  mlbStatus,
  gameLive,
  gameFinished,
  lastUpdatedAt,
}) {
  const [activeNav, setActiveNav] = useState("pitching");
  const feed = useMlbGameFeed(g.game_id, gameLive || gameFinished);
  const weather = useGameWeather(g.game_id);
  const weatherForecast = useBallparkForecast(weather?.lat, weather?.lng, weather?.startUtc ?? g.first_pitch_utc, 4);
  // Only fetch live per-book odds before first pitch; once live/final, the API returns in-game lines.
  const { books: bookRows, quotaExhausted: oddsQuotaExhausted } = useAllBookMoneylines(g.away_team, g.home_team, !gameLive && !gameFinished);
  const { books: runlineRows, quotaExhausted: rlQuotaExhausted } = useAllBookRunlines(g.away_team, g.home_team, !gameLive && !gameFinished);
  const { books: totalsRows, quotaExhausted: totalsQuotaExhausted } = useAllBookTotals(g.away_team, g.home_team, !gameLive && !gameFinished);
  const awayStarts = usePitcherLastStarts(enrich?.away?.spId, seasonYear);
  const homeStarts = usePitcherLastStarts(enrich?.home?.spId, seasonYear);
  const standings = useTeamStandingsForGame(g.away_team, g.home_team, seasonYear);
  const themeAway = getTeamTheme(g.away_team);
  const themeHome = getTeamTheme(g.home_team);
  const showScorecardBatting = shouldShowGameBattingLine(mlbStatus);
  const e = enrich;
  const hasLineupData = !!(e && (e.away?.lineup?.some(s => s.entries?.length > 0) || e.home?.lineup?.some(s => s.entries?.length > 0)));
  const venueLabel = liveRow?.venueName || e?.venueName || null;
  const arFinal = pickFinishedGameRuns(liveRow?.awayRuns, g.away_runs);
  const hrFinal = pickFinishedGameRuns(liveRow?.homeRuns, g.home_runs);
  const oddsKeyMissing = !import.meta.env.VITE_ODDS_API_KEY;
  const abA = teamAbbr(g.away_team);
  const abH = teamAbbr(g.home_team);
  const titleShort = `${g.away_team?.split(" ").pop() || "Away"} at ${g.home_team?.split(" ").pop() || "Home"}`;
  const awayRecStr = liveRow?.awayRecord ?? null;
  const homeRecStr = liveRow?.homeRecord ?? null;
  const awayDiv = standings?.away?.divisionLabel ?? null;
  const homeDiv = standings?.home?.divisionLabel ?? null;

  const hasFeedBatters = !!(feed?.awayBatters?.length || feed?.homeBatters?.length);

  const scrollToSection = (id) => {
    setActiveNav(id);
    const el = document.getElementById(`detail-section-${id}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const placeholder = (msg) => (
    <div style={{
      border: `1px dashed ${COL.border}`,
      borderRadius: 12,
      padding: 28,
      color: COL.textMuted,
      fontSize: 14,
      textAlign: "center",
      background: COL.card,
    }}>{msg}</div>
  );

  const sectionStyle = { scrollMarginTop: 72, marginBottom: 32 };
  const sectionTitle = (text) => (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: 10,
      marginBottom: 14,
      marginTop: 4,
    }}>
      <div style={{ width: 5, height: 26, borderRadius: 3, background: COL.model, flexShrink: 0 }} />
      <span style={{
        fontSize: 18,
        fontWeight: 800,
        color: COL.textPrimary,
        letterSpacing: "0.06em",
        textTransform: "uppercase",
      }}>{text}</span>
      <div style={{ flex: 1, height: 1, background: COL.border }} />
    </div>
  );

  return (
    <div style={{ maxWidth: 1120, margin: "0 auto", padding: "0 12px 48px" }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        gap: 16,
        flexWrap: "wrap",
        marginBottom: 20,
        paddingBottom: 16,
        borderBottom: `1px solid ${COL.border}`,
      }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", minWidth: 0 }}>
          <button
            type="button"
            onClick={onBack}
            style={{
              fontSize: 18,
              fontWeight: 700,
              color: COL.text,
              background: "transparent",
              border: "none",
              padding: 0,
              cursor: "pointer",
              fontFamily: "inherit",
              lineHeight: 1,
            }}
            aria-label="Back to schedule"
          >
            ←
          </button>
          <h2 style={{
            fontSize: "clamp(18px, 4vw, 24px)",
            fontWeight: 800,
            color: COL.text,
            margin: 0,
            letterSpacing: "-0.02em",
          }}
          >
            {titleShort}
          </h2>
        </div>
        <div style={{ textAlign: "right", flexShrink: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: COL.textSecondary }}>
            {formatGameDetailTimestamp(g.first_pitch_utc) ?? "—"}
          </div>
          {lastUpdatedAt && (
            <div style={{ fontSize: 12, color: COL.textMuted, marginTop: 4 }}>
              Updated {formatRelativeAgo(lastUpdatedAt)}
            </div>
          )}
        </div>
      </div>

      {venueLabel && (gameLive || gameFinished) && (
        <div style={{ fontSize: 12, color: COL.textSecondary, fontWeight: 600, marginBottom: 14 }}>📍 {venueLabel}</div>
      )}

      <div style={{ display: "flex", gap: 0, alignItems: "stretch", flexDirection: "row", flexWrap: "wrap" }}>
        <nav
          aria-label="Game sections"
          style={{
            width: 200,
            flexShrink: 0,
            borderRight: `1px solid ${COL.border}`,
            paddingRight: 12,
            marginRight: 16,
            marginBottom: 16,
            position: "sticky",
            top: 12,
            alignSelf: "flex-start",
            maxHeight: "calc(100vh - 24px)",
            overflowY: "auto",
          }}
        >
          {DETAIL_NAV.map((item) => {
            const active = activeNav === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => scrollToSection(item.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  width: "100%",
                  textAlign: "left",
                  border: "none",
                  background: active ? "rgba(234,88,0,0.08)" : "transparent",
                  borderRadius: 8,
                  padding: "10px 10px 10px 6px",
                  marginBottom: 4,
                  cursor: "pointer",
                  fontFamily: "inherit",
                  fontSize: 13,
                  fontWeight: active ? 800 : 600,
                  color: active ? COL.text : COL.textSecondary,
                }}
              >
                <span style={{
                  width: 6,
                  height: 6,
                  borderRadius: 2,
                  background: active ? TAB_ACCENT : "transparent",
                  flexShrink: 0,
                }}
                />
                {item.label}
              </button>
            );
          })}
        </nav>
        <div style={{ flex: 1, minWidth: 280, paddingBottom: 48 }}>
          {gameLive && (
            <LiveGameBanner g={g} liveRow={liveRow} feed={feed} />
          )}
          <div style={{ marginBottom: 24 }}>
            <BreakdownDarkCard g={g} standings={standings} awayRecStr={awayRecStr} homeRecStr={homeRecStr} />
          </div>
          {gameFinished && (
            <GameFinalScoreboardCard
              awayTeamName={g.away_team}
              homeTeamName={g.home_team}
              awayAbbr={abA}
              homeAbbr={abH}
              awayRuns={arFinal}
              homeRuns={hrFinal}
              awayRec={awayRecStr}
              homeRec={homeRecStr}
              awayDiv={awayDiv}
              homeDiv={homeDiv}
              feed={feed}
              themeAway={themeAway}
              themeHome={themeHome}
            />
          )}

          <section id="detail-section-pitching" style={sectionStyle}>
            {sectionTitle("STARTING PITCHING")}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
              {[
                { side: "away", team: g.away_team, theme: themeAway, sp: e?.away?.spName || g.away_sp_name, stats: e?.away?.stats, starts: awayStarts },
                { side: "home", team: g.home_team, theme: themeHome, sp: e?.home?.spName || g.home_sp_name, stats: e?.home?.stats, starts: homeStarts },
              ].map((col) => (
                <div
                  key={col.side}
                  style={{
                    borderRadius: 12,
                    border: `1px solid ${COL.border}`,
                    background: COL.card,
                    boxShadow: `0 4px 16px rgba(15,23,42,0.07), 0 0 0 1px ${col.theme.stroke}`,
                    overflow: "hidden",
                  }}
                >
                  <div style={{ height: 4, background: col.theme.primary }} />
                  <div style={{
                    padding: "12px 14px",
                    background: `linear-gradient(135deg, ${col.theme.soft} 0%, #FFFFFF 100%)`,
                    borderBottom: `1px solid ${COL.border}`,
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                  }}
                  >
                    <Logo team={col.team} size={26} />
                    <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.1, minWidth: 0 }}>
                      <span style={{ fontSize: 10, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em" }}>PITCHING</span>
                      <span style={{
                        fontSize: 14,
                        fontWeight: 800,
                        color: COL.text,
                        letterSpacing: "-0.01em",
                        marginTop: 2,
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                      >{col.team}</span>
                    </div>
                  </div>
                  <div style={{ padding: 12 }}>
                    <PitcherStarterCard spName={col.sp} stats={col.stats} theme={col.theme} />
                    <div style={{
                      fontSize: 10,
                      fontWeight: 800,
                      color: COL.textMuted,
                      letterSpacing: "0.08em",
                      marginTop: 14,
                      marginBottom: 6,
                      paddingLeft: 2,
                    }}
                    >LAST 3 STARTS
                    </div>
                    <PitcherLastStartsTable rows={col.starts} />
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section id="detail-section-lineup" style={sectionStyle}>
            {sectionTitle("LINEUP — BATTERS")}
            {hasFeedBatters ? (
              <BattersBoxGrid
                awayAbbr={abA}
                homeAbbr={abH}
                awayTeamName={g.away_team}
                homeTeamName={g.home_team}
                awayRows={feed.awayBatters}
                homeRows={feed.homeBatters}
                themeAway={themeAway}
                themeHome={themeHome}
              />
            ) : hasLineupData ? (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
                {["away", "home"].map((side) => {
                  const slots = e[side]?.lineup || [];
                  const title = side === "away" ? g.away_team : g.home_team;
                  const abbr = side === "away" ? abA : abH;
                  const th = (side === "away" ? themeAway : themeHome) || { primary: COL.model, soft: COL.cardInner, stroke: COL.border, onPrimary: "#fff" };
                  const BATTER_H_PREGAME = { padding: "7px 10px", fontWeight: 800, fontSize: 10.5, letterSpacing: "0.06em", color: COL.textSecondary, textTransform: "uppercase", whiteSpace: "nowrap", background: th.soft };
                  const rows = slots.flatMap((slot) =>
                    (slot.entries || []).map((row, ei) => ({
                      order: ei === 0 ? slot.order : null,
                      name: row.name,
                      avg: row.avg,
                      isSub: row.isSub,
                      atBat: !!(liveRow?.batterName && lineupBatterMatches(row.name, liveRow.batterName)),
                    }))
                  );
                  return (
                    <div key={side} style={{ minWidth: 0, borderRadius: 12, border: `1px solid ${COL.border}`, background: COL.card, boxShadow: `0 4px 16px rgba(15,23,42,0.07), 0 0 0 1px ${th.stroke}`, overflow: "hidden" }}>
                      <div style={{ height: 4, background: th.primary }} />
                      <div style={{ padding: "12px 14px", background: `linear-gradient(135deg, ${th.soft} 0%, #FFFFFF 100%)`, display: "flex", alignItems: "center", gap: 10, borderBottom: `1px solid ${COL.border}` }}>
                        <Logo team={title} size={24} />
                        <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.1 }}>
                          <span style={{ fontSize: 10, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em" }}>BATTERS</span>
                          <span style={{ fontSize: 14, fontWeight: 800, color: COL.text, letterSpacing: "-0.01em", marginTop: 2 }}>{title}</span>
                        </div>
                        <span style={{ marginLeft: "auto", fontSize: 10, fontWeight: 900, color: th.onPrimary, background: th.primary, padding: "3px 10px", borderRadius: 999, letterSpacing: "0.08em", boxShadow: `0 1px 3px ${th.stroke}` }}>{abbr}</span>
                      </div>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                        <thead>
                          <tr style={{ background: th.soft, borderBottom: `2px solid ${th.primary}` }}>
                            <th style={{ ...BATTER_H_PREGAME, textAlign: "left", width: 28 }}>#</th>
                            <th style={{ ...BATTER_H_PREGAME, textAlign: "left" }}>PLAYER</th>
                            <th style={{ ...BATTER_H_PREGAME, textAlign: "right", width: 52 }}>AVG</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((row, i) => (
                            <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : th.soft, borderBottom: `1px solid ${COL.border}` }}>
                              <td style={{ padding: "7px 10px", color: COL.textMuted, fontVariantNumeric: "tabular-nums", fontSize: 11, fontWeight: 700 }}>
                                {row.order != null ? row.order : ""}
                              </td>
                              <td style={{ padding: "7px 10px", color: COL.text, fontWeight: row.isSub ? 500 : 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 180 }}>
                                {row.isSub && <span style={{ color: COL.textMuted, marginRight: 5 }}>↳</span>}
                                {row.atBat ? (
                                  <span style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
                                    <BatIcon size={12} />
                                    {row.name}
                                  </span>
                                ) : row.name}
                              </td>
                              <td style={{ padding: "7px 10px", textAlign: "right", fontVariantNumeric: "tabular-nums", color: th.primary, fontWeight: 700, fontSize: 12 }}>
                                {row.avg != null && row.avg !== "—" ? row.avg : <span style={{ color: COL.textMuted }}>—</span>}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  );
                })}
              </div>
            ) : (
              placeholder(gameFinished ? "Box score batting lines load when the MLB feed is available." : "Lineups appear when projected or announced lineups are available.")
            )}
          </section>

          <section id="detail-section-last10" style={sectionStyle}>
            {sectionTitle("LAST 10")}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
              <TeamRecentFormColumn
                teamName={g.away_team}
                seasonYear={seasonYear}
                excludeGamePk={g.game_id}
                theme={themeAway}
              />
              <TeamRecentFormColumn
                teamName={g.home_team}
                seasonYear={seasonYear}
                excludeGamePk={g.game_id}
                theme={themeHome}
              />
            </div>
          </section>

          <section id="detail-section-weather" style={sectionStyle}>
            {sectionTitle("WEATHER")}
            <GameWeatherSection
              weather={weather}
              forecast={weatherForecast}
              themeAway={themeAway}
              themeHome={themeHome}
            />
          </section>

          <section id="detail-section-umpire" style={sectionStyle}>
            {sectionTitle("UMPIRE")}
            <div style={{
              border: `1px solid ${COL.border}`,
              borderRadius: 12,
              overflow: "hidden",
              maxWidth: 480,
              background: COL.card,
              boxShadow: `0 4px 14px rgba(15,23,42,0.08), 0 0 0 1px ${themeHome.stroke}`,
            }}
            >
              <div style={{
                height: 4,
                background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeHome.primary} 100%)`,
              }}
              />
              <div style={{ ...GP_STAT_HEADER, textAlign: "left", padding: "10px 14px" }}>Home plate umpire</div>
              <div style={{ padding: "16px 18px", background: "#FFFFFF" }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: COL.text, letterSpacing: "-0.02em" }}>{e?.umpireName ?? "—"}</div>
              </div>
            </div>
          </section>

          <section id="detail-section-moneyline" style={sectionStyle}>
            {sectionTitle("MONEY LINE")}
            <SteamMoveCallout g={g} themeAway={themeAway} themeHome={themeHome} />
            {!oddsKeyMissing && bookRows && bookRows.length > 0 && (
              <div style={{
                marginTop: 14,
                border: `1px solid ${COL.border}`,
                borderRadius: 12,
                overflow: "hidden",
                background: COL.card,
                boxShadow: `0 4px 14px rgba(15,23,42,0.07)`,
              }}
              >
                <div style={{
                  height: 4,
                  background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)`,
                }}
                />
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: "#1a202c" }}>
                  <span style={{ fontSize: 10, fontWeight: 800, color: "#FFFFFF", letterSpacing: "0.1em", textTransform: "uppercase" }}>Per-book lines</span>
                  <span style={{ fontSize: 10.5, fontWeight: 600, color: "rgba(255,255,255,0.55)", background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 999, padding: "2px 10px", letterSpacing: 0.2 }}>
                    📌 Pre-first pitch odds
                  </span>
                </div>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: COL.cardInner }}>
                      <th style={{
                        textAlign: "left",
                        padding: "10px 14px",
                        fontSize: 10,
                        fontWeight: 800,
                        color: COL.textSecondary,
                        letterSpacing: "0.06em",
                        borderBottom: `1px solid ${COL.border}`,
                      }}
                      >BOOK</th>
                      <th style={{
                        textAlign: "right",
                        padding: "10px 14px",
                        fontSize: 10,
                        fontWeight: 800,
                        color: themeAway.primary,
                        letterSpacing: "0.06em",
                        borderBottom: `2px solid ${themeAway.primary}`,
                        background: themeAway.soft,
                      }}
                      >{abA}</th>
                      <th style={{
                        textAlign: "right",
                        padding: "10px 14px",
                        fontSize: 10,
                        fontWeight: 800,
                        color: themeHome.primary,
                        letterSpacing: "0.06em",
                        borderBottom: `2px solid ${themeHome.primary}`,
                        background: themeHome.soft,
                      }}
                      >{abH}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bookRows.map((b, i) => (
                      <tr
                        key={b.key}
                        style={{
                          borderTop: `1px solid ${COL.border}`,
                          background: i % 2 === 1 ? "rgba(15,23,42,0.015)" : "#fff",
                        }}
                      >
                        <td style={{ padding: "11px 14px", fontWeight: 700, color: COL.text }}>{b.title}</td>
                        <td style={{
                          padding: "11px 14px",
                          textAlign: "right",
                          fontVariantNumeric: "tabular-nums",
                          fontWeight: 800,
                          color: COL.text,
                          background: `${themeAway.soft}`,
                        }}
                        >{b.away != null ? fmt(b.away) : "—"}</td>
                        <td style={{
                          padding: "11px 14px",
                          textAlign: "right",
                          fontVariantNumeric: "tabular-nums",
                          fontWeight: 800,
                          color: COL.text,
                          background: `${themeHome.soft}`,
                        }}
                        >{b.home != null ? fmt(b.home) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {(oddsKeyMissing || oddsQuotaExhausted) && (
              <div style={{
                marginTop: 14,
                border: `1px solid rgba(245,158,11,0.35)`,
                borderRadius: 12,
                background: "rgba(245,158,11,0.06)",
                padding: "12px 16px",
                fontSize: 13,
                color: COL.textSecondary,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
              >
                <span style={{ fontSize: 16 }}>⚠️</span>
                {oddsQuotaExhausted
                  ? "Odds API monthly quota exhausted — per-book lines unavailable until the quota resets. Upgrade at the-odds-api.com if needed."
                  : <>Set <code style={{ fontSize: 12 }}>VITE_ODDS_API_KEY</code> to load sportsbook moneylines.</>
                }
              </div>
            )}
            {!oddsKeyMissing && !oddsQuotaExhausted && bookRows && bookRows.length === 0 && !gameLive && !gameFinished && (
              <div style={{
                marginTop: 14,
                border: `1px solid ${COL.border}`,
                borderRadius: 12,
                background: COL.card,
                padding: "14px 16px",
                fontSize: 13,
                color: COL.textMuted,
              }}
              >
                No per-book lines available for this matchup yet.
              </div>
            )}
            {gameFinished && !gameLive && (() => {
              const ch = g.closing_home_price ?? g.morning_home_price;
              const ca = g.closing_away_price ?? g.morning_away_price;
              if (!ch && !ca) return null;
              return (
                <div style={{ marginTop: 14, border: `1px solid ${COL.border}`, borderRadius: 12, overflow: "hidden", background: COL.card, boxShadow: `0 4px 14px rgba(15,23,42,0.07)` }}>
                  <div style={{ height: 4, background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)` }} />
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: "#1a202c" }}>
                    <span style={{ fontSize: 10, fontWeight: 800, color: "#FFFFFF", letterSpacing: "0.1em", textTransform: "uppercase" }}>Closing consensus lines</span>
                    <span style={{ fontSize: 10.5, fontWeight: 600, color: "rgba(255,255,255,0.55)", background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 999, padding: "2px 10px", letterSpacing: 0.2 }}>
                      📌 Pre-first pitch odds
                    </span>
                  </div>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                    <thead>
                      <tr style={{ background: COL.cardInner }}>
                        <th style={{ textAlign: "left", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: COL.textSecondary, letterSpacing: "0.06em", borderBottom: `1px solid ${COL.border}` }}>LINE</th>
                        <th style={{ textAlign: "right", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: themeAway.primary, letterSpacing: "0.06em", borderBottom: `2px solid ${themeAway.primary}`, background: themeAway.soft }}>{abA}</th>
                        <th style={{ textAlign: "right", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: themeHome.primary, letterSpacing: "0.06em", borderBottom: `2px solid ${themeHome.primary}`, background: themeHome.soft }}>{abH}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {g.morning_away_price != null && (
                        <tr style={{ borderTop: `1px solid ${COL.border}`, background: "#fff" }}>
                          <td style={{ padding: "11px 14px", fontWeight: 700, color: COL.text }}>Opening</td>
                          <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: COL.text, background: themeAway.soft }}>{fmt(g.morning_away_price)}</td>
                          <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: COL.text, background: themeHome.soft }}>{fmt(g.morning_home_price)}</td>
                        </tr>
                      )}
                      {g.closing_away_price != null && (
                        <tr style={{ borderTop: `1px solid ${COL.border}`, background: "rgba(15,23,42,0.015)" }}>
                          <td style={{ padding: "11px 14px", fontWeight: 700, color: COL.text }}>Closing</td>
                          <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: COL.text, background: themeAway.soft }}>{fmt(g.closing_away_price)}</td>
                          <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: COL.text, background: themeHome.soft }}>{fmt(g.closing_home_price)}</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              );
            })()}
          </section>

          {/* ── RUN LINE ────────────────────────────────────────── */}
          <section id="detail-section-runline" style={sectionStyle}>
            {sectionTitle("RUN LINE")}
            {!gameLive && !gameFinished && runlineRows && runlineRows.length > 0 && (() => {
              // Compute consensus line (average of first-book point) for move display
              const firstPoint = runlineRows[0]?.awayPoint;
              return (
                <div style={{ border: `1px solid ${COL.border}`, borderRadius: 12, overflow: "hidden", background: COL.card, boxShadow: `0 4px 14px rgba(15,23,42,0.07)` }}>
                  <div style={{ height: 4, background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)` }} />
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: "#1a202c" }}>
                    <span style={{ fontSize: 10, fontWeight: 800, color: "#FFFFFF", letterSpacing: "0.1em", textTransform: "uppercase" }}>Per-book run lines</span>
                    <span style={{ fontSize: 10.5, fontWeight: 600, color: "rgba(255,255,255,0.55)", background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 999, padding: "2px 10px", letterSpacing: 0.2 }}>
                      📌 Pre-first pitch odds
                    </span>
                  </div>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                    <thead>
                      <tr style={{ background: COL.cardInner }}>
                        <th style={{ textAlign: "left", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: COL.textSecondary, letterSpacing: "0.06em", borderBottom: `1px solid ${COL.border}` }}>BOOK</th>
                        <th style={{ textAlign: "right", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: themeAway.primary, letterSpacing: "0.06em", borderBottom: `2px solid ${themeAway.primary}`, background: themeAway.soft }}>{abA} LINE</th>
                        <th style={{ textAlign: "right", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: themeAway.primary, letterSpacing: "0.06em", borderBottom: `2px solid ${themeAway.primary}`, background: themeAway.soft }}>{abA} ODDS</th>
                        <th style={{ textAlign: "right", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: themeHome.primary, letterSpacing: "0.06em", borderBottom: `2px solid ${themeHome.primary}`, background: themeHome.soft }}>{abH} LINE</th>
                        <th style={{ textAlign: "right", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: themeHome.primary, letterSpacing: "0.06em", borderBottom: `2px solid ${themeHome.primary}`, background: themeHome.soft }}>{abH} ODDS</th>
                      </tr>
                    </thead>
                    <tbody>
                      {runlineRows.map((b, i) => {
                        const awayLineStr = b.awayPoint != null ? (b.awayPoint > 0 ? `+${b.awayPoint}` : String(b.awayPoint)) : "—";
                        const homeLineStr = b.homePoint != null ? (b.homePoint > 0 ? `+${b.homePoint}` : String(b.homePoint)) : "—";
                        return (
                          <tr key={b.key} style={{ borderTop: `1px solid ${COL.border}`, background: i % 2 === 1 ? "rgba(15,23,42,0.015)" : "#fff" }}>
                            <td style={{ padding: "11px 14px", fontWeight: 700, color: COL.text }}>{b.title}</td>
                            <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: COL.text, background: themeAway.soft }}>{awayLineStr}</td>
                            <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 700, color: COL.textSecondary, background: themeAway.soft }}>{b.awayPrice != null ? fmt(b.awayPrice) : "—"}</td>
                            <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: COL.text, background: themeHome.soft }}>{homeLineStr}</td>
                            <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 700, color: COL.textSecondary, background: themeHome.soft }}>{b.homePrice != null ? fmt(b.homePrice) : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              );
            })()}
            {!gameLive && !gameFinished && !rlQuotaExhausted && (!runlineRows || runlineRows.length === 0) && (
              <div style={{ marginTop: 4, border: `1px solid ${COL.border}`, borderRadius: 12, background: COL.card, padding: "14px 16px", fontSize: 13, color: COL.textMuted }}>
                No run line data available yet for this matchup.
              </div>
            )}
            {(gameLive || gameFinished) && (() => {
              const ch = g.closing_home_price ?? g.morning_home_price;
              const ca = g.closing_away_price ?? g.morning_away_price;
              if (!ch && !ca) return (
                <div style={{ marginTop: 4, border: `1px solid ${COL.border}`, borderRadius: 12, background: COL.card, padding: "14px 16px", fontSize: 13, color: COL.textMuted }}>
                  Run line data not available for this game.
                </div>
              );
              return (
                <div style={{ border: `1px solid ${COL.border}`, borderRadius: 12, overflow: "hidden", background: COL.card, boxShadow: `0 4px 14px rgba(15,23,42,0.07)` }}>
                  <div style={{ height: 4, background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)` }} />
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: "#1a202c" }}>
                    <span style={{ fontSize: 10, fontWeight: 800, color: "#FFFFFF", letterSpacing: "0.1em", textTransform: "uppercase" }}>Run line not tracked live</span>
                  </div>
                  <div style={{ padding: "14px 16px", fontSize: 13, color: COL.textMuted }}>
                    Per-book run line odds are only available before first pitch.
                  </div>
                </div>
              );
            })()}
          </section>

          {/* ── O/U TOTALS ──────────────────────────────────────── */}
          <section id="detail-section-totals" style={sectionStyle}>
            {sectionTitle("O/U TOTALS")}
            {!gameLive && !gameFinished && totalsRows && totalsRows.length > 0 && (
              <div style={{ border: `1px solid ${COL.border}`, borderRadius: 12, overflow: "hidden", background: COL.card, boxShadow: `0 4px 14px rgba(15,23,42,0.07)` }}>
                <div style={{ height: 4, background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)` }} />
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: "#1a202c" }}>
                  <span style={{ fontSize: 10, fontWeight: 800, color: "#FFFFFF", letterSpacing: "0.1em", textTransform: "uppercase" }}>Per-book O/U totals</span>
                  <span style={{ fontSize: 10.5, fontWeight: 600, color: "rgba(255,255,255,0.55)", background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 999, padding: "2px 10px", letterSpacing: 0.2 }}>
                    📌 Pre-first pitch odds
                  </span>
                </div>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: COL.cardInner }}>
                      <th style={{ textAlign: "left", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: COL.textSecondary, letterSpacing: "0.06em", borderBottom: `1px solid ${COL.border}` }}>BOOK</th>
                      <th style={{ textAlign: "right", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: COL.textSecondary, letterSpacing: "0.06em", borderBottom: `1px solid ${COL.border}` }}>LINE</th>
                      <th style={{ textAlign: "right", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: "#22c55e", letterSpacing: "0.06em", borderBottom: `2px solid #22c55e`, background: "rgba(34,197,94,0.05)" }}>OVER</th>
                      <th style={{ textAlign: "right", padding: "10px 14px", fontSize: 10, fontWeight: 800, color: "#ef4444", letterSpacing: "0.06em", borderBottom: `2px solid #ef4444`, background: "rgba(239,68,68,0.05)" }}>UNDER</th>
                    </tr>
                  </thead>
                  <tbody>
                    {totalsRows.map((b, i) => (
                      <tr key={b.key} style={{ borderTop: `1px solid ${COL.border}`, background: i % 2 === 1 ? "rgba(15,23,42,0.015)" : "#fff" }}>
                        <td style={{ padding: "11px 14px", fontWeight: 700, color: COL.text }}>{b.title}</td>
                        <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: COL.text }}>{b.line}</td>
                        <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 700, color: "#22c55e", background: "rgba(34,197,94,0.04)" }}>{b.overPrice != null ? fmt(b.overPrice) : "—"}</td>
                        <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 700, color: "#ef4444", background: "rgba(239,68,68,0.04)" }}>{b.underPrice != null ? fmt(b.underPrice) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {!gameLive && !gameFinished && !totalsQuotaExhausted && (!totalsRows || totalsRows.length === 0) && (
              <div style={{ marginTop: 4, border: `1px solid ${COL.border}`, borderRadius: 12, background: COL.card, padding: "14px 16px", fontSize: 13, color: COL.textMuted }}>
                No O/U totals data available yet for this matchup.
              </div>
            )}
            {(gameLive || gameFinished) && (
              <div style={{ border: `1px solid ${COL.border}`, borderRadius: 12, overflow: "hidden", background: COL.card, boxShadow: `0 4px 14px rgba(15,23,42,0.07)` }}>
                <div style={{ height: 4, background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)` }} />
                <div style={{ padding: "10px 14px", background: "#1a202c" }}>
                  <span style={{ fontSize: 10, fontWeight: 800, color: "#FFFFFF", letterSpacing: "0.1em", textTransform: "uppercase" }}>O/U totals not tracked live</span>
                </div>
                <div style={{ padding: "14px 16px", fontSize: 13, color: COL.textMuted }}>
                  Per-book O/U totals are only available before first pitch.
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Weather section (dark banner + hourly forecast)                   */
/* ------------------------------------------------------------------ */

function WindArrowBadge({ deg, size = 44 }) {
  // Points in the direction the wind is going. Open-Meteo wind_direction is "where it comes from",
  // so we add 180° for visual flow. If deg is null, show a simple compass dot.
  const rotation = deg != null && Number.isFinite(Number(deg)) ? (Number(deg) + 180) % 360 : null;
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: 10,
        background: "linear-gradient(135deg, #22C55E 0%, #16A34A 100%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        boxShadow: "0 2px 6px rgba(34,197,94,0.35)",
        flexShrink: 0,
        position: "relative",
      }}
      aria-hidden
    >
      <svg
        width={size * 0.55}
        height={size * 0.55}
        viewBox="0 0 24 24"
        style={{ transform: rotation != null ? `rotate(${rotation}deg)` : "none", transition: "transform 0.3s ease" }}
      >
        <path d="M12 2 L6 20 L12 16 L18 20 Z" fill="#FFFFFF" stroke="#FFFFFF" strokeWidth="1" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function formatForecastHourLabel(isoLocal, timezone) {
  if (!isoLocal) return "—";
  try {
    const d = new Date(`${isoLocal}:00`);
    return d.toLocaleTimeString("en-US", { timeZone: timezone, hour: "numeric", minute: "2-digit", hour12: true });
  } catch {
    return isoLocal.slice(11, 16);
  }
}

function GameWeatherSection({ weather, forecast, themeAway, themeHome }) {
  if (!weather && !forecast) {
    return (
      <div
        style={{
          border: `1px dashed ${COL.border}`,
          borderRadius: 12,
          padding: 24,
          color: COL.textMuted,
          fontSize: 13,
          textAlign: "center",
          background: COL.card,
        }}
      >
        Weather data isn't available for this game yet.
      </div>
    );
  }

  const parsed = parseMlbWind(weather?.wind);
  const headerPrimaryDeg = forecast?.rows?.[0]?.windDeg ?? null;
  const headerCompass = degreesToCompass(headerPrimaryDeg);
  const headerMph = parsed?.mph ?? forecast?.rows?.[0]?.windMph ?? null;
  const headerDirLabel = (() => {
    if (parsed?.dir) return parsed.dir;
    if (headerCompass) return headerCompass;
    return "Calm";
  })();
  const ballparkDir = prettifyBallparkDir(parsed?.ballpark);
  const ouEffect = parsed?.ou || "Neutral";

  const locationLine = (() => {
    const parts = [];
    if (weather?.venueName) parts.push(weather.venueName);
    if (weather?.city) parts.push(weather.city + (weather.state ? `, ${weather.state}` : ""));
    else if (weather?.state) parts.push(weather.state);
    return parts.join(", ");
  })();

  const rows = forecast?.rows || [];
  const timezone = forecast?.timezone || "UTC";
  const fallbackSingleRow = !rows.length && weather
    ? [{
      timeLocal: null,
      tempF: weather.temp ? Number(String(weather.temp).replace(/[^\d.-]/g, "")) : null,
      humidity: null,
      precipProb: null,
      code: null,
      windMph: parsed?.mph ?? null,
      windDeg: null,
      conditionLabel: weather.condition || null,
    }]
    : null;
  const tableRows = fallbackSingleRow || rows;

  const ouEffectColor = (() => {
    if (ouEffect === "High on Over" || ouEffect === "Slight Over") return "#86EFAC";
    if (ouEffect === "Suppresses Offense" || ouEffect === "Slight Under") return "#FCA5A5";
    if (ouEffect === "Blustery") return "#FBBF24";
    return "#CBD5E1";
  })();

  const numFmt = (v, suffix = "") => (v != null ? `${v}${suffix}` : "—");

  return (
    <div
      style={{
        border: `1px solid ${COL.border}`,
        borderRadius: 14,
        overflow: "hidden",
        background: "#FFFFFF",
        boxShadow: `0 4px 18px rgba(15,23,42,0.08), 0 0 0 1px ${themeAway?.stroke || COL.border}`,
      }}
    >
      <div
        style={{
          background: "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)",
          padding: "18px 20px",
          color: "#FFFFFF",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 20,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14, minWidth: 0, flex: "1 1 260px" }}>
          <WindArrowBadge deg={headerPrimaryDeg} size={48} />
          <div style={{ minWidth: 0 }}>
            <div
              style={{
                fontSize: 22,
                fontWeight: 900,
                letterSpacing: "-0.01em",
                lineHeight: 1.1,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {headerDirLabel === "Calm" ? "Calm" : `${headerDirLabel} ${headerMph ?? ""}${headerMph != null ? " mph" : ""}`.trim()}
            </div>
            {locationLine && (
              <div
                style={{
                  fontSize: 13,
                  color: "rgba(226,232,240,0.75)",
                  fontWeight: 600,
                  marginTop: 2,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {locationLine}
              </div>
            )}
          </div>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            columnGap: 36,
            flex: "0 1 auto",
            minWidth: 260,
            borderLeft: "1px solid rgba(255,255,255,0.12)",
            paddingLeft: 24,
          }}
        >
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "rgba(226,232,240,0.65)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
              Wind
            </div>
            <div style={{ fontSize: 14, fontWeight: 800, color: "#FFFFFF", marginTop: 6 }}>
              {ballparkDir || headerCompass || "—"}
            </div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "rgba(226,232,240,0.65)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
              O/U Effect
            </div>
            <div style={{ fontSize: 14, fontWeight: 800, color: ouEffectColor, marginTop: 6 }}>
              {ouEffect}
            </div>
          </div>
        </div>
      </div>

      {tableRows.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              minWidth: 520,
              borderCollapse: "collapse",
              fontFamily: "inherit",
            }}
          >
            <thead>
              <tr>
                <th
                  style={{
                    padding: "12px 16px",
                    width: 140,
                    textAlign: "left",
                    borderBottom: `1px solid ${COL.border}`,
                    background: "#FFFFFF",
                  }}
                />
                {tableRows.map((r, i) => (
                  <th
                    key={i}
                    style={{
                      padding: "12px 8px",
                      textAlign: "center",
                      fontSize: 12.5,
                      fontWeight: 700,
                      color: COL.textSecondary,
                      borderBottom: `1px solid ${COL.border}`,
                      background: "#FFFFFF",
                    }}
                  >
                    {r.timeLocal ? formatForecastHourLabel(r.timeLocal, timezone) : "Now"}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { label: "Temp (°F)", render: (r) => numFmt(r.tempF, "°") },
                { label: "Humidity", render: (r) => numFmt(r.humidity, "%") },
                { label: "Precipitation", render: (r) => numFmt(r.precipProb, "%") },
                {
                  label: "Condition",
                  render: (r) => r.conditionLabel ?? WEATHER_CODE_LABEL[r.code] ?? "—",
                },
                {
                  label: "Wind",
                  render: (r) => {
                    const c = degreesToCompass(r.windDeg);
                    if (r.windMph == null && !c) return "—";
                    if (r.windMph == null) return c || "—";
                    return `${c ? c + " " : ""}${r.windMph} mph`;
                  },
                },
              ].map((row, rowIdx) => (
                <tr
                  key={row.label}
                  style={{
                    background: rowIdx % 2 === 0 ? "#FFFFFF" : "#F8FAFC",
                  }}
                >
                  <td
                    style={{
                      padding: "12px 16px",
                      fontSize: 13,
                      fontWeight: 800,
                      color: COL.text,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {row.label}
                  </td>
                  {tableRows.map((r, i) => (
                    <td
                      key={i}
                      style={{
                        padding: "12px 8px",
                        textAlign: "center",
                        fontSize: 13,
                        color: COL.textSecondary,
                        fontWeight: 600,
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {row.render(r)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Inning-by-inning linescore table                                  */
/* ------------------------------------------------------------------ */

function LinescoreTable({
  awayTeamName,
  homeTeamName,
  awayAbbr,
  homeAbbr,
  innings,
  rhe,
  themeAway,
  themeHome,
  currentInning = null,
  inningState = null,
}) {
  const minInnings = 9;
  const maxInn = Math.max(minInnings, (innings && innings.length) || 0);
  const innNums = Array.from({ length: maxInn }, (_, i) => i + 1);
  const awayInningRuns = (n) => {
    const inn = innings?.find((x) => x.num === n);
    return inn ? inn.away : undefined;
  };
  const homeInningRuns = (n) => {
    const inn = innings?.find((x) => x.num === n);
    return inn ? inn.home : undefined;
  };
  const isTop = typeof inningState === "string" && inningState.toLowerCase().startsWith("t");
  const currentCol = currentInning != null ? Number(currentInning) : null;
  const cellStyle = (isCurrent, isAway) => ({
    padding: "8px 4px",
    textAlign: "center",
    fontSize: 14,
    fontWeight: 700,
    color: COL.text,
    fontVariantNumeric: "tabular-nums",
    background: isCurrent && ((isAway && isTop) || (!isAway && !isTop)) ? "rgba(220,38,38,0.08)" : "transparent",
  });
  const rheCell = (v, bold = false) => (
    <td
      style={{
        padding: "8px 8px",
        textAlign: "center",
        fontSize: 14,
        fontWeight: bold ? 900 : 700,
        color: bold ? COL.text : COL.textSecondary,
        fontVariantNumeric: "tabular-nums",
      }}
    >{v ?? "—"}</td>
  );

  return (
    <div
      style={{
        background: "#FFFFFF",
        border: `1px solid ${COL.border}`,
        borderRadius: 12,
        overflow: "hidden",
        boxShadow: "0 1px 3px rgba(15,23,42,0.04)",
      }}
    >
      <div style={{ padding: "10px 14px 4px", overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "inherit" }}>
          <thead>
            <tr>
              <th style={{ padding: "4px 10px 4px 6px" }} />
              {innNums.map((n) => (
                <th
                  key={n}
                  style={{
                    padding: "4px 4px",
                    minWidth: 26,
                    textAlign: "center",
                    fontSize: 12,
                    fontWeight: 700,
                    color: currentCol === n ? COL.negative : COL.textMuted,
                  }}
                >
                  {n}
                </th>
              ))}
              <th style={{ padding: "4px 8px", textAlign: "center", fontSize: 12, fontWeight: 900, color: COL.text }}>R</th>
              <th style={{ padding: "4px 8px", textAlign: "center", fontSize: 12, fontWeight: 700, color: COL.textMuted }}>H</th>
              <th style={{ padding: "4px 8px", textAlign: "center", fontSize: 12, fontWeight: 700, color: COL.textMuted }}>E</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ padding: "6px 10px 6px 0", whiteSpace: "nowrap" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Logo team={awayTeamName} size={20} />
                  <span style={{ fontSize: 13, fontWeight: 800, color: themeAway?.primary || COL.text, letterSpacing: 0.2 }}>
                    {awayAbbr}
                  </span>
                </div>
              </td>
              {innNums.map((n) => {
                const v = awayInningRuns(n);
                return (
                  <td key={n} style={cellStyle(currentCol === n, true)}>
                    {v != null ? v : ""}
                  </td>
                );
              })}
              {rheCell(rhe?.away?.r, true)}
              {rheCell(rhe?.away?.h)}
              {rheCell(rhe?.away?.e)}
            </tr>
            <tr style={{ borderTop: `1px solid ${COL.border}` }}>
              <td style={{ padding: "6px 10px 6px 0", whiteSpace: "nowrap" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Logo team={homeTeamName} size={20} />
                  <span style={{ fontSize: 13, fontWeight: 800, color: themeHome?.primary || COL.text, letterSpacing: 0.2 }}>
                    {homeAbbr}
                  </span>
                </div>
              </td>
              {innNums.map((n) => {
                const raw = homeInningRuns(n);
                const played = raw !== null && raw !== undefined;
                return (
                  <td key={n} style={cellStyle(currentCol === n, false)}>
                    {played ? raw : ""}
                  </td>
                );
              })}
              {rheCell(rhe?.home?.r, true)}
              {rheCell(rhe?.home?.h)}
              {rheCell(rhe?.home?.e)}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Live game banner (used by the game detail page)                   */
/* ------------------------------------------------------------------ */

function LiveGameBanner({ g, liveRow, feed }) {
  const atBat = useLiveAtBat(g.game_id, true);

  const awayRunsLive = liveRow?.awayRuns ?? feed?.awayRuns ?? 0;
  const homeRunsLive = liveRow?.homeRuns ?? feed?.homeRuns ?? 0;
  const inningLabel = liveRow?.currentInning && liveRow?.inningState
    ? `${liveRow.inningState} ${ordinal(liveRow.currentInning)}`
    : null;
  const venueLabel = liveRow?.venueName || null;
  const themeAway = getTeamTheme(g.away_team);
  const themeHome = getTeamTheme(g.home_team);
  const hasLinescore = !!(feed?.innings?.length || feed?.rhe);

  return (
    <div
      style={{
        borderRadius: 14,
        overflow: "hidden",
        border: "1px solid rgba(220,38,38,0.25)",
        background: "linear-gradient(180deg, rgba(220,38,38,0.08) 0%, rgba(220,38,38,0.02) 100%)",
        boxShadow: "0 6px 20px rgba(220,38,38,0.10), 0 1px 3px rgba(15,23,42,0.06)",
        marginBottom: 16,
      }}
    >
      <div
        style={{
          padding: "14px 18px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
          borderBottom: "1px solid rgba(220,38,38,0.18)",
        }}
      >
        <LiveBadge />
        <div style={{ display: "flex", alignItems: "center", gap: 16, flex: 1, justifyContent: "center", minWidth: 0 }}>
          <Logo team={g.away_team} size={36} />
          <span
            style={{
              fontSize: 34,
              fontWeight: 900,
              color: COL.text,
              fontVariantNumeric: "tabular-nums",
              letterSpacing: "-0.02em",
              lineHeight: 1,
            }}
          >{awayRunsLive}</span>
          <span style={{ color: COL.textMuted, fontWeight: 700, fontSize: 22 }}>—</span>
          <span
            style={{
              fontSize: 34,
              fontWeight: 900,
              color: COL.text,
              fontVariantNumeric: "tabular-nums",
              letterSpacing: "-0.02em",
              lineHeight: 1,
            }}
          >{homeRunsLive}</span>
          <Logo team={g.home_team} size={36} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", flexShrink: 0, fontSize: 11, lineHeight: 1.3, minWidth: 100 }}>
          {inningLabel && (
            <span style={{ fontWeight: 800, color: COL.text, fontSize: 14 }}>{inningLabel}</span>
          )}
          {venueLabel && (
            <span style={{ color: COL.textSecondary, fontWeight: 600, marginTop: 2 }}>
              📍 {venueLabel}
            </span>
          )}
        </div>
      </div>

      {liveRow && (
        <div
          style={{
            padding: "12px 16px",
            display: "flex",
            alignItems: "stretch",
            gap: 14,
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12, flex: 1, minWidth: 260 }}>
            {(atBat?.pitches?.length > 0 || atBat?.batter) ? (
              <StrikeZone
                pitches={atBat?.pitches || []}
                zoneTop={atBat?.pitches?.find((p) => p.zoneTop != null)?.zoneTop}
                zoneBottom={atBat?.pitches?.find((p) => p.zoneBottom != null)?.zoneBottom}
                batSide={atBat?.batSide || null}
              />
            ) : null}
            <div style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 0, flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <BaseDiamond
                  onFirst={!!liveRow.onFirst}
                  onSecond={!!liveRow.onSecond}
                  onThird={!!liveRow.onThird}
                />
                <div
                  style={{
                    background: "#FFFFFF",
                    border: "1px solid rgba(220,38,38,0.25)",
                    borderRadius: 10,
                    padding: "6px 12px",
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    boxShadow: "0 1px 3px rgba(15,23,42,0.04)",
                  }}
                >
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <span style={{ fontSize: 9, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em" }}>COUNT</span>
                    <span style={{ fontSize: 18, fontWeight: 900, color: COL.text, fontVariantNumeric: "tabular-nums", lineHeight: 1.1 }}>
                      {formatPitchCount(liveRow.balls, liveRow.strikes)}
                    </span>
                  </div>
                  <div style={{ width: 1, alignSelf: "stretch", background: COL.border }} />
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <span style={{ fontSize: 9, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em" }}>OUTS</span>
                    <span style={{ fontSize: 18, fontWeight: 900, color: COL.text, fontVariantNumeric: "tabular-nums", lineHeight: 1.1 }}>
                      {liveRow.outs ?? 0}
                    </span>
                  </div>
                </div>
              </div>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 4,
                  fontSize: 12.5,
                  lineHeight: 1.3,
                }}
              >
                <div style={{ color: COL.text }}>
                  <span style={{ fontSize: 10, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginRight: 6 }}>AT BAT</span>
                  <span style={{ fontWeight: 800 }}>{liveRow.batterName ?? atBat?.batter ?? "—"}</span>
                  {atBat?.batSide && (
                    <span style={{ color: COL.textMuted, fontWeight: 700, marginLeft: 6, fontSize: 10.5 }}>({atBat.batSide}HB)</span>
                  )}
                </div>
                <div style={{ color: COL.text }}>
                  <span style={{ fontSize: 10, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginRight: 6 }}>PITCHING</span>
                  <span style={{ fontWeight: 800 }}>{liveRow.pitcherName ?? atBat?.pitcher ?? "—"}</span>
                  {atBat?.pitchHand && (
                    <span style={{ color: COL.textMuted, fontWeight: 700, marginLeft: 6, fontSize: 10.5 }}>({atBat.pitchHand}HP)</span>
                  )}
                </div>
                {atBat?.pitches?.length > 0 && (() => {
                  const last = atBat.pitches[atBat.pitches.length - 1];
                  const parts = [];
                  if (last.pitchNumber != null) parts.push(`#${last.pitchNumber}`);
                  if (last.typeDesc) parts.push(last.typeDesc);
                  if (last.startSpeed != null) parts.push(`${Math.round(last.startSpeed)} mph`);
                  if (last.callDesc) parts.push(last.callDesc);
                  if (!parts.length) return null;
                  return (
                    <div style={{ color: COL.textSecondary, fontSize: 11.5 }}>
                      <span style={{ fontSize: 10, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginRight: 6 }}>LAST PITCH</span>
                      <span style={{ fontWeight: 700, color: COL.text }}>{parts.join(" · ")}</span>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
        </div>
      )}

      {hasLinescore && (
        <div
          style={{
            padding: "0 16px 14px",
          }}
        >
          <LinescoreTable
            awayTeamName={g.away_team}
            homeTeamName={g.home_team}
            awayAbbr={teamAbbr(g.away_team)}
            homeAbbr={teamAbbr(g.home_team)}
            innings={feed?.innings || []}
            rhe={feed?.rhe}
            themeAway={themeAway}
            themeHome={themeHome}
            currentInning={liveRow?.currentInning ?? null}
            inningState={liveRow?.inningState ?? null}
          />
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Compact schedule table for the homepage                           */
/* ------------------------------------------------------------------ */

function LiveBadge({ size = "md" }) {
  const sm = size === "sm";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        background: COL.negative,
        color: "#fff",
        padding: sm ? "2px 7px" : "3px 10px",
        borderRadius: 999,
        fontSize: sm ? 9 : 10,
        fontWeight: 800,
        letterSpacing: 0.6,
        textTransform: "uppercase",
        boxShadow: "0 1px 4px rgba(239,68,68,0.4)",
        whiteSpace: "nowrap",
      }}
    >
      <span
        style={{
          width: sm ? 5 : 6,
          height: sm ? 5 : 6,
          borderRadius: "50%",
          background: "#fff",
          display: "inline-block",
          animation: "mlbLivePulse 1.6s ease-in-out infinite",
        }}
      />
      LIVE
    </span>
  );
}

const GAMES_TABLE_HEADER_STYLE = {
  textAlign: "left",
  fontSize: 10,
  fontWeight: 800,
  letterSpacing: 0.6,
  textTransform: "uppercase",
  color: COL.textMuted,
  padding: "10px 10px",
  background: "#F8FAFC",
  whiteSpace: "nowrap",
};
const GAMES_TABLE_NUM_HEADER = { ...GAMES_TABLE_HEADER_STYLE, textAlign: "center" };

const GAMES_TABLE_COLS = [
  { key: "time", width: 96, label: "Time", style: GAMES_TABLE_HEADER_STYLE },
  { key: "score", width: 60, label: "Score", style: GAMES_TABLE_NUM_HEADER },
  { key: "teams", width: 220, label: "Teams", style: GAMES_TABLE_HEADER_STYLE },
  { key: "pitchers", width: 170, label: "Pitchers", style: GAMES_TABLE_HEADER_STYLE },
  { key: "model", label: "Model", style: GAMES_TABLE_NUM_HEADER },
  { key: "market", label: "Market", style: GAMES_TABLE_NUM_HEADER },
  { key: "odds", label: "Odds", style: GAMES_TABLE_NUM_HEADER },
  { key: "edge", label: "Edge", style: GAMES_TABLE_NUM_HEADER },
  { key: "runs", label: "Runs", style: GAMES_TABLE_NUM_HEADER },
  { key: "total", label: "Total", style: GAMES_TABLE_NUM_HEADER },
  { key: "ouline", label: "O/U Line", style: GAMES_TABLE_NUM_HEADER },
  { key: "rec", width: 110, label: "Rec", style: GAMES_TABLE_NUM_HEADER },
  { key: "arrow", width: 44, label: "", style: GAMES_TABLE_HEADER_STYLE },
];

const GAMES_SECTION_TITLE_STYLE = {
  padding: "12px 12px 10px",
  fontSize: 12,
  fontWeight: 800,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  color: "#334155",
  background: "linear-gradient(180deg, #EEF2F7 0%, #E2E8F0 100%)",
  borderBottom: "1px solid #CBD5E1",
  borderTop: "1px solid #CBD5E1",
};

function GamesTable({ sortedGames, live, onOpenDetail, standingsMap }) {
  // Ensure the pulse keyframe is injected once.
  useEffect(() => {
    const id = "mlb-live-pulse-kf";
    if (typeof document === "undefined" || document.getElementById(id)) return;
    const s = document.createElement("style");
    s.id = id;
    s.textContent = `@keyframes mlbLivePulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }`;
    document.head.appendChild(s);
  }, []);

  const tableRows = useMemo(() => {
    const liveG = [];
    const upG = [];
    const compG = [];
    for (const g of sortedGames) {
      const s = getHomepageGameSection(g, live);
      if (s === "live") liveG.push(g);
      else if (s === "completed") compG.push(g);
      else upG.push(g);
    }
    const parts = [
      { key: "live", title: "Games in progress", games: liveG },
      { key: "upcoming", title: "Upcoming", games: upG },
      { key: "completed", title: "Completed games", games: compG },
    ].filter((p) => p.games.length > 0);

    const n = sortedGames.length;
    let globalIdx = 0;
    const out = [];
    for (const p of parts) {
      out.push({ type: "section", k: p.key, title: p.title });
      for (const g of p.games) {
        out.push({
          type: "game",
          g,
          isFirst: globalIdx === 0,
          isLast: globalIdx === n - 1,
        });
        globalIdx += 1;
      }
    }
    return out;
  }, [sortedGames, live]);

  return (
    <div
      style={{
        background: COL.cardBg,
        border: `1px solid ${COL.border}`,
        borderRadius: 14,
        overflow: "hidden",
        boxShadow: "0 6px 20px rgba(15,23,42,0.06), 0 1px 3px rgba(15,23,42,0.04)",
      }}
    >
      <div
        style={{
          padding: "9px 14px",
          fontSize: 12.5,
          lineHeight: 1.4,
          color: COL.textMuted,
          background: "linear-gradient(180deg, #F0F4FA 0%, #EEF2F7 100%)",
          borderBottom: `1px solid ${COL.pageBorder || COL.border}`,
        }}
      >
        Games are listed here only after starting lineups are confirmed.
      </div>
      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            minWidth: 1140,
            borderCollapse: "collapse",
            fontFamily: "inherit",
            tableLayout: "fixed",
          }}
        >
          <colgroup>
            {GAMES_TABLE_COLS.map((c) => (
              <col key={c.key} style={{ width: c.width ? `${c.width}px` : undefined }} />
            ))}
          </colgroup>
          <tbody>
            {tableRows.map((row, rowIdx) => {
              if (row.type === "section") {
                const isFirstSection = !tableRows.slice(0, rowIdx).some((r) => r.type === "section");
                return (
                  <tr key={`sec-${row.k}`}>
                    <td
                      colSpan={GAMES_TABLE_COLS.length}
                      style={{
                        ...GAMES_SECTION_TITLE_STYLE,
                        borderTop: isFirstSection ? "none" : GAMES_SECTION_TITLE_STYLE.borderTop,
                      }}
                    >
                      {row.title}
                    </td>
                  </tr>
                );
              }
              return (
                <Fragment key={row.g.game_id}>
                  <GamesTableRow
                    g={row.g}
                    live={live}
                    onOpenDetail={onOpenDetail}
                    standingsMap={standingsMap}
                    isLast={row.isLast}
                    isFirst={row.isFirst}
                  />
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GamesTableRow({ g, live, onOpenDetail, standingsMap, isLast, isFirst }) {
  const [hover, setHover] = useState(false);

  const liveRow = live?.[g.game_id];
  const detailed = liveRow?.status ?? g.status ?? "";
  const abstract = liveRow?.abstractGameState ?? null;
  const coded = liveRow?.codedGameState ?? null;
  const gameFinished = isMlbGameFinished(detailed, abstract, coded);
  const gamePostponed = isPostponedOrCancelled(detailed, abstract);
  const gameLive = !gameFinished && !gamePostponed && (isLiveStatus(detailed) || isLiveStatus(g.status));

  const themeAway = getTeamTheme(g.away_team);
  const themeHome = getTeamTheme(g.home_team);

  const morningH = g.morning_home_price ?? null;
  const morningA = g.morning_away_price ?? null;
  const closingH = g.closing_home_price ?? g.morning_home_price ?? null;
  const closingA = g.closing_away_price ?? g.morning_away_price ?? null;
  const mph = g.closing_p_home ?? g.morning_p_home;
  const rawH = closingH ?? morningH;
  const rawA = closingA ?? morningA;
  const homeML = rawH != null ? fmt(rawH) : (toAmerican(mph) ?? "—");
  const awayML = rawA != null ? fmt(rawA) : (toAmerican(mph ? 1 - mph : null) ?? "—");

  const mpDev = deviggedMarketPct(rawH ?? morningH, rawA ?? morningA);
  const marketPHome = winProbPercent(g.market_p_home) ?? mpDev.home ?? null;
  const marketPAway = winProbPercent(g.market_p_away) ?? mpDev.away ?? null;

  const modelPHome = winProbPercent(g.p_win_home);
  const modelPAway = winProbPercent(g.p_win_away);

  const edgeHome = (modelPHome != null && marketPHome != null) ? (modelPHome - marketPHome) : null;
  const edgeAway = (modelPAway != null && marketPAway != null) ? (modelPAway - marketPAway) : null;

  const awayRunsPred = g.away_runs_pred != null ? Number(g.away_runs_pred) : null;
  const homeRunsPred = g.home_runs_pred != null ? Number(g.home_runs_pred) : null;
  const totalPred = g.total_runs_pred != null ? Number(g.total_runs_pred) : null;

  const ouLine = g.closing_ou_line ?? g.morning_ou_line ?? null;
  const ouRec = computeOU(totalPred, ouLine);

  const awayRunsActual = pickFinishedGameRuns(liveRow?.awayRuns, g.away_runs);
  const homeRunsActual = pickFinishedGameRuns(liveRow?.homeRuns, g.home_runs);
  const totalRunsActual =
    awayRunsActual != null && homeRunsActual != null
      ? Number(awayRunsActual) + Number(homeRunsActual)
      : null;
  const ouResult = gameFinished ? gradeOuResult(ouRec, totalRunsActual, ouLine) : null;
  const awayRunsLive = gameLive ? (liveRow?.awayRuns ?? 0) : (gameFinished ? awayRunsActual : null);
  const homeRunsLive = gameLive ? (liveRow?.homeRuns ?? 0) : (gameFinished ? homeRunsActual : null);

  let inningLabel = null;
  if (gameLive && liveRow?.currentInning && liveRow?.inningState) {
    inningLabel = `${liveRow.inningState} ${ordinal(liveRow.currentInning)}`;
  }

  const firstPitchParts = formatFirstPitchParts(g.first_pitch_utc);
  const awayRec = standingsMap?.[g.away_team];
  const homeRec = standingsMap?.[g.home_team];

  const onRowClick = () => onOpenDetail?.(g.game_id);

  const finishedBg = "rgba(245,158,11,0.05)";
  const finishedHoverBg = "rgba(245,158,11,0.09)";
  const baseRowStyle = {
    cursor: "pointer",
    background: gameFinished
      ? (hover ? finishedHoverBg : finishedBg)
      : (hover ? "#F8FAFC" : "#FFFFFF"),
    transition: "background 0.12s ease",
  };
  const finishedAccent = gameFinished
    ? { borderLeft: "4px solid #F59E0B" }
    : {};
  const cellBase = {
    padding: "10px 10px",
    verticalAlign: "middle",
    fontSize: 13,
    color: COL.textPrimary,
    borderTop: `1px solid ${COL.borderSoft || "#EEF2F6"}`,
  };
  const thickTop = { borderTop: "none" };
  const noBottom = { borderBottom: "none" };
  const midBetweenTeams = { borderTop: `1px dashed ${COL.borderSoft || "#EEF2F6"}` };

  const teamRowCell = (isAway) => {
    const theme = isAway ? themeAway : themeHome;
    const name = isAway ? g.away_team : g.home_team;
    const rec = isAway ? awayRec : homeRec;
    const isWinner = gameFinished && awayRunsActual != null && homeRunsActual != null
      && (isAway ? awayRunsActual > homeRunsActual : homeRunsActual > awayRunsActual);
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <Logo team={name} size={22} />
        <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
          <span
            style={{
              fontSize: 13.5,
              fontWeight: isWinner ? 800 : 700,
              color: theme?.primary || COL.textPrimary,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              maxWidth: 150,
            }}
            title={name}
          >
            {name}
          </span>
          {rec?.wins != null && rec?.losses != null && (
            <span style={{ fontSize: 11, color: COL.textMuted, fontWeight: 600 }}>
              ({rec.wins}-{rec.losses})
            </span>
          )}
        </div>
      </div>
    );
  };

  const scoreCell = (isAway) => {
    const score = isAway ? awayRunsLive : homeRunsLive;
    if (score == null || (!gameLive && !gameFinished)) {
      return <span style={{ color: COL.textMuted, fontWeight: 600 }}>—</span>;
    }
    const isWinner = gameFinished && awayRunsActual != null && homeRunsActual != null
      && (isAway ? awayRunsActual > homeRunsActual : homeRunsActual > awayRunsActual);
    return (
      <span
        style={{
          fontSize: 16,
          fontWeight: 800,
          color: isWinner ? COL.positive : (gameLive ? COL.negative : COL.textPrimary),
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {score}
      </span>
    );
  };

  const pitcherCell = (name) => (
    <span
      style={{
        fontSize: 13,
        color: COL.textPrimary,
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
        display: "inline-block",
        maxWidth: 160,
      }}
      title={name || "TBD"}
    >
      {name || <span style={{ color: COL.textMuted, fontStyle: "italic" }}>TBD</span>}
    </span>
  );

  const pctCell = (v, strong = false) => (
    <span
      style={{
        fontSize: strong ? 13.5 : 13,
        fontWeight: strong ? 800 : 600,
        color: strong ? COL.model : COL.textSecondary,
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {v != null && Number.isFinite(v) ? `${v.toFixed(1)}%` : "—"}
    </span>
  );

  /** Per matchup: higher model win% = green, lower = red, tie = dark text. */
  const modelPctCell = (isAway) => {
    const v = isAway ? modelPAway : modelPHome;
    if (v == null || !Number.isFinite(v)) {
      return <span style={{ color: COL.textMuted, fontWeight: 800, fontSize: 13.5, fontVariantNumeric: "tabular-nums" }}>—</span>;
    }
    const a = modelPAway;
    const h = modelPHome;
    let color = COL.model;
    if (a != null && h != null && Number.isFinite(a) && Number.isFinite(h)) {
      const d = a - h;
      if (Math.abs(d) < 0.05) color = COL.text;
      else if (d > 0) color = isAway ? COL.positive : COL.negative;
      else color = isAway ? COL.negative : COL.positive;
    }
    return (
      <span
        style={{
          fontSize: 13.5,
          fontWeight: 800,
          color,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {v.toFixed(1)}%
      </span>
    );
  };

  const edgeCell = (v) => {
    if (v == null || !Number.isFinite(v)) {
      return <span style={{ color: COL.textMuted, fontWeight: 700 }}>—</span>;
    }
    const positive = v >= 0;
    return (
      <span
        style={{
          fontSize: 12.5,
          fontWeight: 800,
          color: positive ? COL.positive : COL.negative,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {positive ? "+" : ""}{v.toFixed(1)}%
      </span>
    );
  };

  const mlCell = (v) => (
    <span
      style={{
        fontSize: 12.5,
        fontWeight: 700,
        color: COL.textPrimary,
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {v}
    </span>
  );

  const runsCell = (v) => (
    <span style={{ fontSize: 12.5, fontWeight: 700, color: COL.textPrimary, fontVariantNumeric: "tabular-nums" }}>
      {v != null ? v.toFixed(2) : "—"}
    </span>
  );

  const timeCell = () => {
    if (gamePostponed) {
      return (
        <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-start" }}>
          <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: 0.6, color: COL.textMuted, textTransform: "uppercase" }}>Postponed</span>
        </div>
      );
    }
    if (gameFinished) {
      return (
        <div style={{ display: "flex", flexDirection: "column", gap: 3, alignItems: "flex-start" }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 800,
              letterSpacing: 0.6,
              color: "#92400E",
              textTransform: "uppercase",
              border: "1px solid rgba(245,158,11,0.4)",
              padding: "2px 10px",
              borderRadius: 999,
              background: "rgba(245,158,11,0.15)",
            }}
          >
            Final
          </span>
        </div>
      );
    }
    if (gameLive) {
      return (
        <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-start" }}>
          <LiveBadge />
          {inningLabel && (
            <span style={{ fontSize: 10.5, fontWeight: 700, color: COL.textSecondary, letterSpacing: 0.2 }}>
              {inningLabel}
            </span>
          )}
        </div>
      );
    }
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {firstPitchParts?.et ? (
          <>
            <span style={{ fontSize: 12.5, fontWeight: 800, color: COL.textPrimary, letterSpacing: 0.2 }}>
              {firstPitchParts.et} ET
            </span>
            {firstPitchParts.pt && (
              <span style={{ fontSize: 10.5, color: COL.textMuted, fontWeight: 600 }}>
                {firstPitchParts.pt} PT
              </span>
            )}
          </>
        ) : (
          <span style={{ fontSize: 12, color: COL.textMuted, fontWeight: 600 }}>—</span>
        )}
      </div>
    );
  };

  const totalCell = () => (
    <span style={{ fontSize: 13.5, fontWeight: 800, color: COL.model, fontVariantNumeric: "tabular-nums" }}>
      {totalPred != null ? totalPred.toFixed(2) : "—"}
    </span>
  );

  const ouLineCell = () => (
    <span style={{ fontSize: 13, fontWeight: 700, color: COL.textPrimary, fontVariantNumeric: "tabular-nums" }}>
      {ouLine != null ? Number(ouLine).toFixed(1) : "—"}
    </span>
  );

  const recCell = () => {
    if (gamePostponed) return <span style={{ color: COL.textMuted }}>—</span>;
    let pill = null;
    if (ouRec === "over") pill = <Pill color="green">Over</Pill>;
    else if (ouRec === "under") pill = <Pill color="red">Under</Pill>;
    else if (ouRec === "push") pill = <Pill color="gray">Pass</Pill>;
    else return <span style={{ color: COL.textMuted }}>—</span>;
    const showMark = gameFinished && ouResult != null && ouRec !== "push";
    if (!showMark) return pill;
    const isHit = ouResult === "hit";
    const markBg = isHit ? "rgba(34,197,94,0.18)" : "rgba(239,68,68,0.18)";
    const markColor = isHit ? COL.positive : COL.negative;
    const markBorder = isHit ? "rgba(22,163,74,0.45)" : "rgba(220,38,38,0.45)";
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: 6, justifyContent: "center", flexWrap: "nowrap" }}>
        {pill}
        <span
          aria-label={isHit ? "Hit" : "Miss"}
          title={isHit ? "Recommendation hit" : "Recommendation missed"}
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: markBg,
            color: markColor,
            border: `1.5px solid ${markBorder}`,
            fontSize: 12,
            fontWeight: 900,
            lineHeight: 1,
            flex: "0 0 auto",
          }}
        >
          {isHit ? "✓" : "✕"}
        </span>
      </span>
    );
  };

  const arrowCell = () => (
    <span
      aria-hidden
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 26,
        height: 26,
        borderRadius: "50%",
        background: hover ? COL.model : COL.controlBg,
        color: hover ? "#fff" : COL.model,
        fontSize: 13,
        fontWeight: 800,
        transition: "all 0.12s ease",
      }}
    >
      →
    </span>
  );

  const topCellShared = { ...cellBase, ...thickTop, ...noBottom };
  const bottomCellShared = { ...cellBase, ...midBetweenTeams };

  // For rowspan cells, span both rows.
  const rowSpan2Style = {
    ...cellBase,
    ...thickTop,
    borderBottom: isLast ? "none" : undefined,
    textAlign: "center",
    verticalAlign: "middle",
  };

  const onHoverOn = () => setHover(true);
  const onHoverOff = () => setHover(false);

  const headerRowSpacerTop = isFirst ? "none" : `8px solid ${COL.cardBg || "#F1F5F9"}`;
  return (
    <>
      <tr>
        {GAMES_TABLE_COLS.map((c) => (
          <th
            key={c.key}
            style={{
              ...c.style,
              borderTop: headerRowSpacerTop,
              borderBottom: `1px solid ${COL.border}`,
            }}
          >
            {c.label}
          </th>
        ))}
      </tr>
      <tr
        style={baseRowStyle}
        onClick={onRowClick}
        onMouseEnter={onHoverOn}
        onMouseLeave={onHoverOff}
      >
        <td style={{ ...rowSpan2Style, ...finishedAccent, textAlign: "left" }} rowSpan={2}>
          {timeCell()}
        </td>
        <td style={{ ...topCellShared, textAlign: "center" }}>{scoreCell(true)}</td>
        <td style={topCellShared}>{teamRowCell(true)}</td>
        <td style={topCellShared}>{pitcherCell(g.away_sp_name)}</td>
        <td style={{ ...topCellShared, textAlign: "center" }}>{modelPctCell(true)}</td>
        <td style={{ ...topCellShared, textAlign: "center" }}>{pctCell(marketPAway)}</td>
        <td style={{ ...topCellShared, textAlign: "center" }}>{mlCell(awayML)}</td>
        <td style={{ ...topCellShared, textAlign: "center" }}>{edgeCell(edgeAway)}</td>
        <td style={{ ...topCellShared, textAlign: "center" }}>{runsCell(awayRunsPred)}</td>
        <td style={{ ...rowSpan2Style }} rowSpan={2}>{totalCell()}</td>
        <td style={{ ...rowSpan2Style }} rowSpan={2}>{ouLineCell()}</td>
        <td style={{ ...rowSpan2Style }} rowSpan={2}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
            {recCell()}
            <button
              onClick={(e) => { e.stopPropagation(); onOpenDetail?.(g.game_id); }}
              style={{
                fontSize: 10.5,
                fontWeight: 700,
                color: hover ? "#fff" : COL.model,
                background: hover ? COL.model : "transparent",
                border: `1.5px solid ${COL.model}`,
                borderRadius: 999,
                padding: "2px 10px",
                cursor: "pointer",
                letterSpacing: 0.3,
                transition: "all 0.12s ease",
                whiteSpace: "nowrap",
              }}
            >
              Details →
            </button>
          </div>
        </td>
        <td
          style={{
            ...rowSpan2Style,
            verticalAlign: "bottom",
            paddingBottom: 12,
          }}
          rowSpan={2}
        >
          {arrowCell()}
        </td>
      </tr>
      <tr
        style={baseRowStyle}
        onClick={onRowClick}
        onMouseEnter={onHoverOn}
        onMouseLeave={onHoverOff}
      >
        <td style={{ ...bottomCellShared, textAlign: "center" }}>{scoreCell(false)}</td>
        <td style={bottomCellShared}>{teamRowCell(false)}</td>
        <td style={bottomCellShared}>{pitcherCell(g.home_sp_name)}</td>
        <td style={{ ...bottomCellShared, textAlign: "center" }}>{modelPctCell(false)}</td>
        <td style={{ ...bottomCellShared, textAlign: "center" }}>{pctCell(marketPHome)}</td>
        <td style={{ ...bottomCellShared, textAlign: "center" }}>{mlCell(homeML)}</td>
        <td style={{ ...bottomCellShared, textAlign: "center" }}>{edgeCell(edgeHome)}</td>
        <td style={{ ...bottomCellShared, textAlign: "center" }}>{runsCell(homeRunsPred)}</td>
      </tr>
    </>
  );
}

function GameCard({ g, live, enrich, onOpenDetail, seasonYear }) {
  const openDetail = onOpenDetail ?? (() => {});
  const liveRow = live[g.game_id];
  const detailed = liveRow?.status ?? g.status ?? "";
  const abstract = liveRow?.abstractGameState ?? null;
  const coded = liveRow?.codedGameState ?? null;
  const mlbStatus = detailed;
  const gameFinished = isMlbGameFinished(detailed, abstract, coded);
  const gamePostponed = isPostponedOrCancelled(detailed, abstract);
  const gameLive = !gameFinished && !gamePostponed && (isLiveStatus(mlbStatus) || isLiveStatus(g.status));
  const liveMoneylines = useLiveMoneylines(g.away_team, g.home_team, gameLive);
  const feed = useMlbGameFeed(g.game_id, gameFinished);
  const atBat = useLiveAtBat(g.game_id, gameLive);
  const standings = useTeamStandingsForGame(g.away_team, g.home_team, seasonYear, gameFinished);
  const themeAway = getTeamTheme(g.away_team);
  const themeHome = getTeamTheme(g.home_team);

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
  const awayDiv = standings?.away?.divisionLabel ?? null;
  const homeDiv = standings?.home?.divisionLabel ?? null;

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

  const firstPitchParts = formatFirstPitchParts(g.first_pitch_utc);
  const venueLabel = liveRow?.venueName || e?.venueName || null;

  return (
    <div style={{
      background: COL.card,
      border: gamePostponed
        ? "2px solid #D97706"
        : gameFinished
          ? `2px solid #0f172a`
          : `1px solid ${COL.border}`,
      borderRadius: 16,
      marginBottom: 32,
      overflow: "hidden",
      boxShadow: gameFinished || gamePostponed
        ? `0 10px 28px rgba(0,0,0,0.14), 0 0 0 1px ${themeAway.stroke}`
        : "0 6px 22px rgba(15,23,42,0.1), 0 1px 3px rgba(15,23,42,0.05)",
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
        <>
          <GameFinalScoreboardCard
            awayTeamName={g.away_team}
            homeTeamName={g.home_team}
            awayAbbr={teamAbbr(g.away_team)}
            homeAbbr={teamAbbr(g.home_team)}
            awayRuns={arFinal}
            homeRuns={hrFinal}
            awayRec={liveRow?.awayRecord ?? null}
            homeRec={liveRow?.homeRecord ?? null}
            awayDiv={awayDiv}
            homeDiv={homeDiv}
            feed={feed}
            themeAway={themeAway}
            themeHome={themeHome}
            marginBottom={10}
          />
          {mlbStatus && String(mlbStatus).toLowerCase().includes("completed early") && (
            <div style={{
              fontSize: 10,
              fontWeight: 600,
              color: COL.textSecondary,
              textAlign: "center",
              padding: "4px 12px 8px",
              letterSpacing: "0.04em",
            }}
            >
              {mlbStatus}
            </div>
          )}
        </>
      )}
      {gameLive && (
        <div style={{
          padding: "12px 14px",
          background: "linear-gradient(180deg, rgba(220,38,38,0.1) 0%, rgba(220,38,38,0.04) 100%)",
          borderBottom: "1px solid rgba(220,38,38,0.22)",
        }}
        >
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 10,
            flexWrap: "wrap",
          }}
          >
            <span style={{
              fontSize: 11,
              fontWeight: 900,
              color: "#fff",
              background: COL.negative,
              padding: "3px 10px",
              borderRadius: 999,
              letterSpacing: "0.14em",
              boxShadow: "0 2px 6px rgba(220,38,38,0.35)",
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
            }}
            >
              <span style={{
                width: 6,
                height: 6,
                borderRadius: 999,
                background: "#fff",
                boxShadow: "0 0 0 2px rgba(255,255,255,0.4)",
                display: "inline-block",
              }}
              />
              LIVE
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: 14, flex: 1, justifyContent: "center", minWidth: 0 }}>
              <Logo team={g.away_team} size={32} />
              <span style={{
                fontSize: 30,
                fontWeight: 900,
                color: COL.text,
                fontVariantNumeric: "tabular-nums",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
              >{awayRunsLive ?? "—"}</span>
              <span style={{ color: COL.textMuted, fontWeight: 700, fontSize: 20 }}>—</span>
              <span style={{
                fontSize: 30,
                fontWeight: 900,
                color: COL.text,
                fontVariantNumeric: "tabular-nums",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
              >{homeRunsLive ?? "—"}</span>
              <Logo team={g.home_team} size={32} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", flexShrink: 0, fontSize: 11, lineHeight: 1.3, minWidth: 100 }}>
              {inningLabel && (
                <span style={{ fontWeight: 800, color: COL.text, fontSize: 13 }}>{inningLabel}</span>
              )}
              {venueLabel && (
                <span style={{ color: COL.textSecondary, fontWeight: 600, marginTop: 2 }}>
                  📍 {venueLabel}
                </span>
              )}
            </div>
          </div>

          {liveRow && (
            <div style={{
              marginTop: 10,
              paddingTop: 10,
              borderTop: "1px solid rgba(220,38,38,0.18)",
              display: "flex",
              alignItems: "stretch",
              gap: 12,
              flexWrap: "wrap",
            }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 220 }}>
                {(atBat?.pitches?.length > 0 || atBat?.batter) ? (
                  <StrikeZone
                    pitches={atBat?.pitches || []}
                    zoneTop={atBat?.pitches?.find((p) => p.zoneTop != null)?.zoneTop}
                    zoneBottom={atBat?.pitches?.find((p) => p.zoneBottom != null)?.zoneBottom}
                    batSide={atBat?.batSide || null}
                  />
                ) : null}
                <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0, flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <BaseDiamond
                      onFirst={!!liveRow.onFirst}
                      onSecond={!!liveRow.onSecond}
                      onThird={!!liveRow.onThird}
                    />
                    <div style={{
                      background: "#FFFFFF",
                      border: "1px solid rgba(220,38,38,0.25)",
                      borderRadius: 8,
                      padding: "4px 10px",
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      boxShadow: "0 1px 3px rgba(15,23,42,0.04)",
                    }}
                    >
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 8, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em" }}>COUNT</span>
                        <span style={{ fontSize: 16, fontWeight: 900, color: COL.text, fontVariantNumeric: "tabular-nums", lineHeight: 1.1 }}>
                          {formatPitchCount(liveRow.balls, liveRow.strikes)}
                        </span>
                      </div>
                      <div style={{ width: 1, alignSelf: "stretch", background: COL.border }} />
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 8, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em" }}>OUTS</span>
                        <span style={{ fontSize: 16, fontWeight: 900, color: COL.text, fontVariantNumeric: "tabular-nums", lineHeight: 1.1 }}>
                          {liveRow.outs ?? "—"}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 4,
                    fontSize: 12,
                    lineHeight: 1.3,
                  }}
                  >
                    <div style={{ color: COL.text }}>
                      <span style={{ fontSize: 9, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginRight: 6 }}>AT BAT</span>
                      <span style={{ fontWeight: 800 }}>{liveRow.batterName ?? atBat?.batter ?? "—"}</span>
                      {atBat?.batSide && (
                        <span style={{ color: COL.textMuted, fontWeight: 700, marginLeft: 6, fontSize: 10 }}>({atBat.batSide}HB)</span>
                      )}
                    </div>
                    <div style={{ color: COL.text }}>
                      <span style={{ fontSize: 9, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginRight: 6 }}>PITCHING</span>
                      <span style={{ fontWeight: 800 }}>{liveRow.pitcherName ?? atBat?.pitcher ?? "—"}</span>
                      {atBat?.pitchHand && (
                        <span style={{ color: COL.textMuted, fontWeight: 700, marginLeft: 6, fontSize: 10 }}>({atBat.pitchHand}HP)</span>
                      )}
                    </div>
                    {atBat?.pitches?.length > 0 && (() => {
                      const last = atBat.pitches[atBat.pitches.length - 1];
                      const parts = [];
                      if (last.pitchNumber != null) parts.push(`#${last.pitchNumber}`);
                      if (last.typeDesc) parts.push(last.typeDesc);
                      if (last.startSpeed != null) parts.push(`${Math.round(last.startSpeed)} mph`);
                      if (last.callDesc) parts.push(last.callDesc);
                      if (!parts.length) return null;
                      return (
                        <div style={{ color: COL.textSecondary, fontSize: 11 }}>
                          <span style={{ fontSize: 9, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginRight: 6 }}>LAST PITCH</span>
                          <span style={{ fontWeight: 700, color: COL.text }}>{parts.join(" · ")}</span>
                        </div>
                      );
                    })()}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {firstPitchParts && !gameFinished && !gameLive && !gamePostponed && (
        <div style={{
          padding: "10px 16px 8px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 10,
          flexWrap: "wrap",
          borderBottom: `1px solid ${COL.border}`,
        }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <span style={{ fontSize: 10, color: COL.textMuted, fontWeight: 700, letterSpacing: "0.08em" }}>⚾ FIRST PITCH</span>
            <span style={{ fontSize: 11, color: COL.text }}>
              <span style={{ fontWeight: 800, color: "#1e3a5f" }}>{firstPitchParts.et} ET</span>
              <span style={{ color: COL.textMuted, margin: "0 5px", fontWeight: 500 }}>/</span>
              <span style={{ fontWeight: 800, color: "#1e3a5f" }}>{firstPitchParts.pt} PT</span>
            </span>
          </div>
          {venueLabel && (
            <span style={{ fontSize: 11, color: COL.textSecondary, fontWeight: 700 }}>📍 {venueLabel}</span>
          )}
        </div>
      )}

      {gameLive && (
        <div style={{ padding: "3px 12px 5px", fontSize: 9, color: COL.textMuted, lineHeight: 1.35 }}>
          Model P and market P are pre–first pitch. Odds are pregame (closing). Live shows current book moneylines.
        </div>
      )}

      <div style={{ padding: "10px 12px 8px" }}>
        {teams.map((r, i) => {
          const th = i === 0 ? themeAway : themeHome;
          const rec = i === 0 ? liveRow?.awayRecord : liveRow?.homeRecord;
          return (
            <div
              key={i}
              style={{
                marginBottom: i === 0 ? 8 : 0,
              }}
            >
              <div style={{
                borderRadius: 10,
                border: `1px solid ${COL.border}`,
                background: COL.card,
                boxShadow: `0 2px 10px rgba(15,23,42,0.07), 0 0 0 1px ${th.stroke}`,
                overflow: "hidden",
              }}
              >
                <div style={{ height: 3, background: th.primary }} />
                <div style={{
                  padding: "8px 12px",
                  background: `linear-gradient(90deg, ${th.primary} 0%, ${th.soft} 65%, #FFFFFF 100%)`,
                  borderBottom: `1px solid ${th.stroke}`,
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                }}
                >
                  <div style={{
                    width: 32,
                    height: 32,
                    borderRadius: 999,
                    background: "#FFFFFF",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    boxShadow: `0 1px 3px rgba(15,23,42,0.18), 0 0 0 2px rgba(255,255,255,0.9)`,
                    flexShrink: 0,
                  }}
                  >
                    <Logo team={r.team} size={24} />
                  </div>
                  <div style={{ minWidth: 0, flex: 1, display: "flex", alignItems: "baseline", gap: 8, lineHeight: 1.1 }}>
                    <span style={{
                      fontSize: 9,
                      fontWeight: 900,
                      color: th.onPrimary,
                      background: "rgba(255,255,255,0.25)",
                      padding: "2px 6px",
                      borderRadius: 4,
                      letterSpacing: "0.1em",
                      flexShrink: 0,
                    }}
                    >{i === 0 ? "AWAY" : "HOME"}
                    </span>
                    <span style={{
                      fontSize: 14,
                      fontWeight: 900,
                      color: th.onPrimary,
                      letterSpacing: "-0.01em",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      minWidth: 0,
                      textShadow: "0 1px 2px rgba(0,0,0,0.15)",
                    }}
                    >{r.team}</span>
                  </div>
                  {rec != null && (
                    <span style={{
                      fontSize: 11,
                      fontWeight: 900,
                      color: th.primary,
                      background: "#FFFFFF",
                      padding: "3px 10px",
                      borderRadius: 999,
                      letterSpacing: "0.04em",
                      fontVariantNumeric: "tabular-nums",
                      boxShadow: `0 1px 3px rgba(15,23,42,0.18)`,
                      whiteSpace: "nowrap",
                      border: `1px solid ${th.stroke}`,
                    }}
                    >{rec}</span>
                  )}
                </div>
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(0, 1.1fr) minmax(0, 1fr)",
                  gap: 8,
                  padding: 8,
                  alignItems: "start",
                }}
                >
                  <PitcherStarterCard
                    spName={r.spEnrich?.spName || r.sp}
                    stats={r.spEnrich?.stats}
                    theme={th}
                  />
                  <TeamMetricsColumns
                    r={r}
                    teamIndex={i}
                    gameLive={gameLive}
                    gameFinished={gameFinished}
                    awayRunsLive={awayRunsLive}
                    homeRunsLive={homeRunsLive}
                    theme={th}
                  />
                </div>
              </div>
            </div>
          );
        })}

        {!gamePostponed && (
          <div style={{
            marginTop: 8,
            paddingTop: 10,
            borderTop: `1px solid ${COL.border}`,
          }}
          >
              <div style={{
                fontSize: 11,
                fontWeight: 700,
                color: COL.textMuted,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                marginBottom: 8,
                textAlign: "left",
              }}
              >
                Game projection
              </div>
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
                gap: 8,
                alignItems: "stretch",
              }}
              >
                <div style={GP_STAT_WRAP}>
                  <div style={GP_STAT_HEADER}>Pred runs</div>
                  <div style={{ ...GP_STAT_BODY, alignItems: "stretch" }}>
                    <PredRunsBars
                      awayTeam={g.away_team}
                      homeTeam={g.home_team}
                      awayPred={g.away_runs_pred}
                      homePred={g.home_runs_pred}
                    />
                  </div>
                </div>
                <div style={GP_STAT_WRAP}>
                  <div style={GP_STAT_HEADER}>Total</div>
                  <div style={GP_STAT_BODY}>
                    <div style={{ fontSize: 18, color: COL.model, fontWeight: 800, lineHeight: 1.15, fontVariantNumeric: "tabular-nums", textAlign: "center", width: "100%" }}>{g.total_runs_pred ?? "—"}</div>
                    <div style={{ fontSize: 10, fontWeight: 600, color: COL.textMuted, marginTop: 6, textAlign: "center" }}>Model total</div>
                  </div>
                </div>
                <div style={GP_STAT_WRAP}>
                  <div style={GP_STAT_HEADER}>Market line</div>
                  <div style={GP_STAT_BODY}>
                    <div style={{ fontSize: 18, color: COL.text, fontWeight: 800, lineHeight: 1.15, fontVariantNumeric: "tabular-nums", textAlign: "center", width: "100%" }}>
                      {ouDisplay != null ? Number(ouDisplay).toFixed(1) : "—"}
                    </div>
                  </div>
                </div>
                <div style={GP_STAT_WRAP}>
                  <div style={GP_STAT_HEADER}>O/U</div>
                  <div style={GP_STAT_BODY}>
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
        )}
      </div>

      {!gamePostponed && (
        <div style={{
          padding: "8px 14px 10px",
          borderTop: `1px solid ${COL.border}`,
          textAlign: "right",
          background: COL.card,
        }}
        >
          <button
            type="button"
            onClick={() => openDetail(g.game_id)}
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: COL.textSecondary,
              background: "#FFFFFF",
              border: `1px solid ${COL.border}`,
              borderRadius: 8,
              padding: "6px 14px",
              cursor: "pointer",
              transition: "background 0.15s ease, border-color 0.15s ease, color 0.15s ease, box-shadow 0.15s ease",
            }}
            onMouseEnter={(ev) => {
              ev.currentTarget.style.background = COL.modelTint;
              ev.currentTarget.style.borderColor = COL.model;
              ev.currentTarget.style.color = COL.model;
            }}
            onMouseLeave={(ev) => {
              ev.currentTarget.style.background = "#FFFFFF";
              ev.currentTarget.style.borderColor = COL.border;
              ev.currentTarget.style.color = COL.textSecondary;
            }}
          >
            Show details
          </button>
        </div>
      )}
    </div>
  );
}

function GameDetailRoute({ g, live, enrich, seasonYear, onBack, lastUpdatedAt }) {
  const liveRow = live[g.game_id];
  const detailed = liveRow?.status ?? g.status ?? "";
  const abstract = liveRow?.abstractGameState ?? null;
  const coded = liveRow?.codedGameState ?? null;
  const mlbStatus = detailed;
  const gameFinished = isMlbGameFinished(detailed, abstract, coded);
  const gamePostponed = isPostponedOrCancelled(detailed, abstract);
  const gameLive = !gameFinished && !gamePostponed && (isLiveStatus(mlbStatus) || isLiveStatus(g.status));
  return (
    <GameDetailPage
      g={g}
      liveRow={liveRow}
      enrich={enrich[g.game_id]}
      seasonYear={seasonYear}
      onBack={onBack}
      mlbStatus={mlbStatus}
      gameLive={gameLive}
      gameFinished={gameFinished}
      lastUpdatedAt={lastUpdatedAt}
    />
  );
}

const ACCURACY_POLL_MS = 90_000; // while Model Accuracy tab is open

function parseAccuracyResponse(r) {
  return r.text().then((bodyText) => {
    let parsed = null;
    try { parsed = bodyText ? JSON.parse(bodyText) : null; } catch { /* ignore */ }
    if (!r.ok) {
      const msg = (parsed && (parsed.message || parsed.error)) || bodyText?.slice(0, 200) || `HTTP ${r.status}`;
      throw new Error(msg);
    }
    return parsed;
  });
}

function useAccuracyData(enabled, refreshKey) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setError(null);
    setLoading(true);
    fetch(`${API}?view=accuracy`)
      .then((r) => parseAccuracyResponse(r))
      .then((j) => { if (!cancelled) { setData(j); setLoading(false); } })
      .catch((e) => { if (!cancelled) { setError(String(e?.message || e)); setLoading(false); } });
    return () => { cancelled = true; };
  }, [enabled, refreshKey]);

  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => {
      fetch(`${API}?view=accuracy`)
        .then((r) => parseAccuracyResponse(r))
        .then((j) => { setData(j); })
        .catch((e) => { setError(String(e?.message || e)); });
    }, ACCURACY_POLL_MS);
    return () => clearInterval(id);
  }, [enabled, refreshKey]);

  return { data, error, loading };
}

function fmtMoney(n) {
  if (n == null || !Number.isFinite(Number(n))) return "—";
  const v = Number(n);
  const sign = v >= 0 ? "+" : "−";
  return `${sign}$${Math.abs(v).toFixed(2)}`;
}

function fmtPct(n, digits = 1) {
  if (n == null || !Number.isFinite(Number(n))) return "—";
  return `${Number(n).toFixed(digits)}%`;
}

function KpiCard({ label, value, sub, accent, mono = true }) {
  return (
    <div style={{
      background: COL.card,
      border: `1px solid ${COL.border}`,
      borderRadius: 14,
      padding: "14px 16px",
      boxShadow: "0 1px 2px rgba(15,23,42,0.04)",
      minWidth: 0,
    }}>
      <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.1em", textTransform: "uppercase", color: COL.textMuted }}>
        {label}
      </div>
      <div style={{
        fontSize: 24,
        fontWeight: 800,
        marginTop: 4,
        color: accent || COL.text,
        fontVariantNumeric: mono ? "tabular-nums" : undefined,
        letterSpacing: "-0.01em",
      }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: COL.textMuted, marginTop: 2, fontVariantNumeric: mono ? "tabular-nums" : undefined }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function PnlLineChart({ rows, height = 220, accent = COL.model }) {
  const data = (rows || []).filter((r) => r && r.date != null && r.cumulative_dollars != null);
  if (data.length < 2) {
    return (
      <div style={{
        height,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: COL.textMuted,
        fontSize: 13,
        background: COL.cardInner,
        border: `1px dashed ${COL.border}`,
        borderRadius: 12,
      }}>
        Not enough graded days to plot P&amp;L yet.
      </div>
    );
  }
  const W = 900, H = height, padL = 46, padR = 14, padT = 14, padB = 28;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const ys = data.map((d) => Number(d.cumulative_dollars));
  let yMin = Math.min(...ys, 0);
  let yMax = Math.max(...ys, 0);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const span = yMax - yMin;
  const pad = span * 0.08;
  yMin -= pad; yMax += pad;

  const xFor = (i) => padL + (i / (data.length - 1)) * innerW;
  const yFor = (v) => padT + innerH - ((v - yMin) / (yMax - yMin)) * innerH;
  const y0 = yFor(0);

  const path = data.map((d, i) => `${i === 0 ? "M" : "L"} ${xFor(i).toFixed(1)} ${yFor(Number(d.cumulative_dollars)).toFixed(1)}`).join(" ");

  const lastY = Number(data[data.length - 1].cumulative_dollars);
  const overallPositive = lastY >= 0;
  const lineColor = overallPositive ? COL.positive : COL.negative;
  const fillTop = overallPositive ? "rgba(22,163,74,0.18)" : "rgba(220,38,38,0.18)";
  const fillBottom = overallPositive ? "rgba(22,163,74,0.01)" : "rgba(220,38,38,0.01)";

  const areaPath = `${path} L ${xFor(data.length - 1).toFixed(1)} ${y0.toFixed(1)} L ${xFor(0).toFixed(1)} ${y0.toFixed(1)} Z`;

  const gridValues = [yMin, (yMin + yMax) / 2, yMax];
  const fmt$ = (v) => `${v >= 0 ? "$" : "−$"}${Math.abs(v).toFixed(0)}`;

  const labelIdxs = data.length <= 8
    ? data.map((_, i) => i)
    : [0, Math.floor(data.length * 0.25), Math.floor(data.length * 0.5), Math.floor(data.length * 0.75), data.length - 1];

  return (
    <div style={{
      background: COL.card,
      border: `1px solid ${COL.border}`,
      borderRadius: 14,
      padding: "10px 10px 4px",
      boxShadow: "0 1px 2px rgba(15,23,42,0.04)",
      overflow: "hidden",
    }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" aria-label="P&L chart" role="img">
        <defs>
          <linearGradient id="pnlFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={fillTop} />
            <stop offset="100%" stopColor={fillBottom} />
          </linearGradient>
        </defs>
        {gridValues.map((g, i) => (
          <g key={i}>
            <line x1={padL} y1={yFor(g)} x2={W - padR} y2={yFor(g)} stroke={COL.border} strokeDasharray={g === 0 ? "0" : "3 4"} strokeWidth={g === 0 ? 1.25 : 1} />
            <text x={padL - 6} y={yFor(g) + 3} fontSize="10" fontFamily="inherit" textAnchor="end" fill={COL.textMuted} fontWeight="600">
              {fmt$(g)}
            </text>
          </g>
        ))}
        <path d={areaPath} fill="url(#pnlFill)" />
        <path d={path} fill="none" stroke={lineColor} strokeWidth={2.25} strokeLinejoin="round" strokeLinecap="round" />
        {data.map((d, i) => (
          <circle key={i} cx={xFor(i)} cy={yFor(Number(d.cumulative_dollars))} r={i === data.length - 1 ? 3.5 : 1.5} fill={lineColor} />
        ))}
        {labelIdxs.map((i) => (
          <text key={i} x={xFor(i)} y={H - 8} fontSize="10" fontFamily="inherit" textAnchor="middle" fill={COL.textMuted}>
            {String(data[i].date).slice(5)}
          </text>
        ))}
      </svg>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "2px 8px 6px" }}>
        <div style={{ fontSize: 11, color: COL.textMuted }}>
          {data.length} graded days · $10 flat stake per pick
        </div>
        <div style={{ fontSize: 12, fontWeight: 800, color: lineColor, fontVariantNumeric: "tabular-nums" }}>
          {fmtMoney(lastY)} total
        </div>
      </div>
    </div>
  );
}

function AccuracyBucketTable({ buckets }) {
  if (!buckets || !buckets.length) {
    return <div style={{ color: COL.textMuted, fontSize: 13 }}>No bucket data yet.</div>;
  }
  const cellStyle = {
    padding: "10px 14px",
    fontSize: 13,
    color: COL.text,
    borderBottom: `1px solid ${COL.border}`,
    fontVariantNumeric: "tabular-nums",
  };
  const thStyle = {
    ...cellStyle,
    fontSize: 10,
    fontWeight: 800,
    color: COL.textMuted,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    background: COL.cardInner,
  };
  return (
    <div style={{
      background: COL.card,
      border: `1px solid ${COL.border}`,
      borderRadius: 14,
      overflow: "hidden",
      boxShadow: "0 1px 2px rgba(15,23,42,0.04)",
    }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ ...thStyle, textAlign: "left" }}>Confidence</th>
            <th style={{ ...thStyle, textAlign: "right" }}>Picks</th>
            <th style={{ ...thStyle, textAlign: "right" }}>W-L</th>
            <th style={{ ...thStyle, textAlign: "right" }}>Win %</th>
            <th style={{ ...thStyle, textAlign: "right" }}>Net $</th>
            <th style={{ ...thStyle, textAlign: "right" }}>ROI</th>
          </tr>
        </thead>
        <tbody>
          {buckets.map((b, i) => {
            const isLast = i === buckets.length - 1;
            const losses = (b.bets || 0) - (b.wins || 0);
            const roiColor = b.roi_pct == null ? COL.textMuted : (b.roi_pct >= 0 ? COL.positive : COL.negative);
            const netColor = b.net_dollars == null ? COL.textMuted : (b.net_dollars >= 0 ? COL.positive : COL.negative);
            return (
              <tr key={b.bucket}>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, fontWeight: 700 }}>
                  {b.bucket}
                </td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right" }}>{b.bets ?? 0}</td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right", color: COL.textSecondary }}>
                  {b.wins ?? 0}-{losses}
                </td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right", fontWeight: 700 }}>
                  {fmtPct(b.win_pct)}
                </td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right", color: netColor, fontWeight: 700 }}>
                  {fmtMoney(b.net_dollars)}
                </td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right", color: roiColor, fontWeight: 700 }}>
                  {fmtPct(b.roi_pct)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RecentDailyPnlTable({ rows, limit = 14 }) {
  if (!rows || !rows.length) {
    return <div style={{ color: COL.textMuted, fontSize: 13 }}>No daily P&amp;L yet.</div>;
  }
  const tail = rows.slice(Math.max(0, rows.length - limit)).reverse();
  const cellStyle = { padding: "9px 14px", fontSize: 12.5, color: COL.text, borderBottom: `1px solid ${COL.border}`, fontVariantNumeric: "tabular-nums" };
  const thStyle = { ...cellStyle, fontSize: 10, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.1em", textTransform: "uppercase", background: COL.cardInner };
  return (
    <div style={{
      background: COL.card,
      border: `1px solid ${COL.border}`,
      borderRadius: 14,
      overflow: "hidden",
      boxShadow: "0 1px 2px rgba(15,23,42,0.04)",
    }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ ...thStyle, textAlign: "left" }}>Date</th>
            <th style={{ ...thStyle, textAlign: "right" }}>Picks</th>
            <th style={{ ...thStyle, textAlign: "right" }}>W-L</th>
            <th style={{ ...thStyle, textAlign: "right" }}>Win %</th>
            <th style={{ ...thStyle, textAlign: "right" }}>Net</th>
            <th style={{ ...thStyle, textAlign: "right" }}>Cumulative</th>
          </tr>
        </thead>
        <tbody>
          {tail.map((d, i) => {
            const isLast = i === tail.length - 1;
            const losses = (d.bets || 0) - (d.wins || 0);
            const netColor = d.net_dollars == null ? COL.textMuted : (d.net_dollars >= 0 ? COL.positive : COL.negative);
            const cumColor = d.cumulative_dollars == null ? COL.textMuted : (d.cumulative_dollars >= 0 ? COL.positive : COL.negative);
            return (
              <tr key={d.date}>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, fontWeight: 700 }}>{String(d.date)}</td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right" }}>{d.bets ?? 0}</td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right", color: COL.textSecondary }}>
                  {d.wins ?? 0}-{losses}
                </td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right" }}>{fmtPct(d.win_pct)}</td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right", color: netColor, fontWeight: 700 }}>
                  {fmtMoney(d.net_dollars)}
                </td>
                <td style={{ ...cellStyle, borderBottom: isLast ? "none" : cellStyle.borderBottom, textAlign: "right", color: cumColor, fontWeight: 700 }}>
                  {fmtMoney(d.cumulative_dollars)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ModelAccuracyPanel({ enabled, refreshKey }) {
  const { data, error, loading } = useAccuracyData(enabled, refreshKey);

  const sectionTitle = (label) => (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: 10,
      margin: "24px 0 14px",
    }}>
      <div style={{ width: 5, height: 26, borderRadius: 3, background: COL.model, flexShrink: 0 }} />
      <span style={{
        fontSize: 18,
        fontWeight: 800,
        color: COL.textPrimary,
        letterSpacing: "0.06em",
        textTransform: "uppercase",
      }}>{label}</span>
      <div style={{ flex: 1, height: 1, background: COL.border }} />
    </div>
  );

  if (loading && !data) {
    return <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "2rem" }}>Loading model accuracy…</p>;
  }
  if (error && !data) {
    return (
      <div style={{ maxWidth: 640, margin: "24px auto", padding: 16, background: COL.card, border: `1px solid ${COL.border}`, borderRadius: 12, color: COL.textSecondary, fontSize: 13 }}>
        Could not load model accuracy data: {error}
      </div>
    );
  }
  if (!data) return null;

  const overall = data.overall || {};
  const ouOverall = data.ou_overall || {};
  const buckets = data.buckets || [];
  const mlDaily = data.daily || [];
  const ouDaily = data.ou_daily || [];
  const mlCum = data.daily_cumulative || [];
  const ouCum = data.ou_daily_cumulative || [];
  const ouPicks = data.ou_pick_counts || { over: 0, under: 0 };

  const mlNetColor = overall.net_dollars != null && overall.net_dollars >= 0 ? COL.positive : COL.negative;
  const ouNetColor = ouOverall.net_dollars != null && ouOverall.net_dollars >= 0 ? COL.positive : COL.negative;
  const mlRoiColor = overall.roi_pct != null && overall.roi_pct >= 0 ? COL.positive : COL.negative;
  const ouRoiColor = ouOverall.roi_pct != null && ouOverall.roi_pct >= 0 ? COL.positive : COL.negative;

  return (
    <div>
      <div style={{
        background: COL.card,
        border: `1px solid ${COL.border}`,
        borderRadius: 16,
        padding: "16px 18px",
        marginTop: 4,
        boxShadow: "0 1px 2px rgba(15,23,42,0.04)",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 800, color: COL.text, letterSpacing: "-0.01em" }}>Model Accuracy</div>
            <div style={{ fontSize: 12, color: COL.textMuted, marginTop: 2 }}>
              Graded games from {data.meta?.min_game_date || "season start"} · flat $10 stake per pick
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, fontSize: 11, color: COL.textMuted }}>
            <span>ML @ actual book odds</span>
            <span>·</span>
            <span>O/U @ -110 assumed</span>
          </div>
        </div>
      </div>

      {sectionTitle("Moneyline — Overall")}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 12 }}>
        <KpiCard label="Picks" value={overall.bets != null ? String(overall.bets) : "—"} sub={overall.wins != null ? `${overall.wins} wins · ${(overall.bets ?? 0) - (overall.wins ?? 0)} losses` : null} />
        <KpiCard label="Win %" value={fmtPct(overall.win_pct)} accent={overall.win_pct != null && overall.win_pct >= 50 ? COL.positive : COL.text} />
        <KpiCard label="Net P&amp;L" value={fmtMoney(overall.net_dollars)} accent={mlNetColor} sub={overall.bets ? `$10 × ${overall.bets} staked` : null} />
        <KpiCard label="ROI" value={fmtPct(overall.roi_pct)} accent={mlRoiColor} />
      </div>

      {sectionTitle("Moneyline — Accuracy by Confidence Bucket")}
      <AccuracyBucketTable buckets={buckets} />

      {sectionTitle("If You Bet $10 per Pick (Moneyline)")}
      <PnlLineChart rows={mlCum} />

      {sectionTitle("Over / Under — Overall")}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 12 }}>
        <KpiCard label="Picks" value={ouOverall.bets != null ? String(ouOverall.bets) : "—"} sub={`${ouPicks.over || 0} overs · ${ouPicks.under || 0} unders`} />
        <KpiCard label="Win %" value={fmtPct(ouOverall.win_pct)} accent={ouOverall.win_pct != null && ouOverall.win_pct >= 52.38 ? COL.positive : COL.text} sub="break-even ≈ 52.4%" />
        <KpiCard label="Net P&amp;L" value={fmtMoney(ouOverall.net_dollars)} accent={ouNetColor} sub={ouOverall.bets ? `$10 × ${ouOverall.bets} staked` : null} />
        <KpiCard label="ROI" value={fmtPct(ouOverall.roi_pct)} accent={ouRoiColor} />
      </div>

      {ouCum.length >= 2 && (
        <>
          {sectionTitle("If You Bet $10 per Pick (Over / Under)")}
          <PnlLineChart rows={ouCum} />
        </>
      )}

      {sectionTitle("Recent Daily P&L (Moneyline)")}
      <RecentDailyPnlTable rows={mlDaily} />

      {ouDaily.length > 0 && (
        <>
          {sectionTitle("Recent Daily P&L (Over / Under)")}
          <RecentDailyPnlTable rows={ouDaily} />
        </>
      )}
    </div>
  );
}

function AboutUsPanel() {
  const p = {
    fontSize: 15,
    lineHeight: 1.7,
    color: "#334155",
    margin: "0 0 14px",
  };
  const h2 = {
    fontSize: 18,
    fontWeight: 800,
    color: COL.text,
    margin: "28px 0 12px",
    letterSpacing: "-0.02em",
  };
  const li = { margin: "0 0 10px" };
  return (
    <div
      style={{
        maxWidth: 800,
        margin: "4px auto 0",
        background: COL.card,
        border: `1px solid ${COL.pageBorder}`,
        borderRadius: 16,
        padding: "28px 28px 36px",
        boxShadow: "0 1px 3px rgba(15,23,42,0.06)",
      }}
    >
      <h1 style={{ fontSize: 26, fontWeight: 800, color: COL.text, margin: "0 0 16px", letterSpacing: "-0.03em" }}>
        About MLB Predictor
      </h1>
      <p style={p}>
        MLB Predictor is an independent research project that forecasts Major League Baseball game outcomes using machine learning. For each matchup, the model estimates
        expected runs for both teams and converts them into a win probability, a projected total, and a predicted score. Forecasts are generated daily from a production
        pipeline that ingests fresh data on pitchers, bullpens, lineups, weather, and betting markets.
      </p>
      <p style={p}>
        This page explains what the model does, what it learns from, and how it&apos;s built. It is not a betting service and does not offer picks or financial advice.
      </p>
      <hr style={{ border: "none", borderTop: `1px solid ${COL.pageBorder}`, margin: "24px 0" }} />

      <h2 style={h2}>What the model predicts</h2>
      <p style={p}>
        For every scheduled game, MLB Predictor produces three things:
      </p>
      <ul style={{ ...p, paddingLeft: 22, marginTop: 0 }}>
        <li style={li}>
          <strong style={{ color: COL.text }}>Expected runs for each team.</strong> The core output — a real-valued estimate of how many runs the home and away sides are
          likely to score, given everything known about the matchup a few hours before first pitch.
        </li>
        <li style={li}>
          <strong style={{ color: COL.text }}>Win probability.</strong> Derived from the expected-runs estimates using a statistical model of how run totals translate into wins.
        </li>
        <li style={li}>
          <strong style={{ color: COL.text }}>Projected total and score.</strong> A human-readable summary that sports fans can compare against the sportsbook line or their
          own intuition.
        </li>
      </ul>
      <p style={p}>
        Predictions update as new information arrives throughout the day, most notably when starting lineups are announced and when closing odds settle.
      </p>

      <h2 style={h2}>What the model sees</h2>
      <p style={p}>
        The model is trained on several seasons of historical games and scores them using features grouped into six categories. Each category captures a different source
        of signal that baseball analysts and sharps have long known to matter.
      </p>
      <figure style={{ margin: "6px 0 22px" }}>
        <div
          style={{
            background: "#0B1020",
            borderRadius: 12,
            padding: 12,
            border: `1px solid ${COL.pageBorder}`,
          }}
        >
          <img
            src="/about-feature-groups.png"
            alt="Diagram: six feature groups (starting pitching, bullpen, lineups, weather, umpire, market) feed into the expected-runs model for home and away estimates."
            style={{ width: "100%", height: "auto", display: "block", borderRadius: 8 }}
          />
        </div>
        <figcaption style={{ fontSize: 12, color: COL.textMuted, marginTop: 10, lineHeight: 1.4 }}>
          How feature groups combine into home and away run estimates.
        </figcaption>
      </figure>
      <p style={p}>
        <strong style={{ color: COL.text }}>Starting pitching.</strong> How good is each team&apos;s starter, and how are they trending? The model uses season-long ERA, plus
        rolling 30-day, 60-day, and last-three-start windows to capture both baseline skill and recent form. Pitchers who haven&apos;t thrown enough innings to stabilize
        get handled with league-average fallbacks so the model isn&apos;t misled by noisy small-sample numbers.
      </p>
      <p style={p}>
        <strong style={{ color: COL.text }}>Bullpen fatigue.</strong> Relief corps that have been overworked in the last few days pitch worse. The model tracks bullpen outs
        recorded over 1-, 3-, and 5-day windows for each team and compares them to opponent usage. A team whose pen threw three innings yesterday against one that
        didn&apos;t is meaningfully disadvantaged.
      </p>
      <p style={p}>
        <strong style={{ color: COL.text }}>Lineups and matchups.</strong> Once official lineups are posted, the model incorporates lineup-level context — who&apos;s batting,
        what handedness they bring, and how that maps against the opposing starter. Matchup features capture platoon effects and known performance splits.
      </p>
      <p style={p}>
        <strong style={{ color: COL.text }}>Weather.</strong> Wind direction, temperature, and humidity at game time all affect how the ball carries. Weather is pulled from
        Open-Meteo for each stadium and merged into the feature set.
      </p>
      <p style={p}>
        <strong style={{ color: COL.text }}>Umpire tendencies.</strong> Home-plate umpires vary in how they call the strike zone, and some games run systematically higher
        or lower than average. The model uses per-umpire run tendencies backfilled across 25,000+ historical games.
      </p>
      <p style={p}>
        <strong style={{ color: COL.text }}>Market signals.</strong> Closing betting lines are one of the most accurate aggregators of public information available. The model
        uses market movement and closing odds as features, and calibrates its own probabilities against the market using isotonic regression — more on that below.
      </p>

      <h2 style={h2}>How it&apos;s trained</h2>
      <p style={p}>
        MLB Predictor uses gradient-boosted regression to predict runs, trained on historical game outcomes with target leakage carefully prevented (no feature uses
        information the model wouldn&apos;t have had before first pitch).
      </p>
      <p style={p}>
        A few design choices are worth calling out for readers interested in the methodology.
      </p>
      <p style={p}>
        <strong style={{ color: COL.text }}>Residual target formulation.</strong> Early versions of the model consistently under-predicted run totals by about one full run
        per game — a persistent bias that resisted simple corrections. The current version trains against the residual between actual runs and a rolling league-average
        baseline, which anchors predictions to the current run-scoring environment and effectively eliminates the bias. This also makes the model robust to year-over-year
        shifts in offensive environment (rule changes, ball changes, and so on).
      </p>
      <p style={p}>
        <strong style={{ color: COL.text }}>Calibration against the market.</strong> Raw model probabilities can be overconfident in the tails. Rather than trusting a
        Skellam or Poisson transformation blindly, the current model applies isotonic regression calibration against historical closing market odds. The market
        isn&apos;t always right, but it&apos;s a well-calibrated baseline, and regressing toward it in the tails produces probabilities that are both accurate and
        usefully sharp.
      </p>
      <p style={p}>
        <strong style={{ color: COL.text }}>Inference-time feature parity.</strong> A major class of model failure isn&apos;t in training — it&apos;s in production, when
        a feature available at training time silently becomes unavailable at inference. The v9 pipeline explicitly verifies that every feature used by the model is
        populated at prediction time, including weather (fetched with a fallback mapping), pitcher workload histories, lineup-dependent features, and umpire assignments.
      </p>

      <h2 style={h2}>How it&apos;s deployed</h2>
      <figure style={{ margin: "6px 0 22px" }}>
        <div
          style={{
            background: "#0B1020",
            borderRadius: 12,
            padding: 12,
            border: `1px solid ${COL.pageBorder}`,
          }}
        >
          <img
            src="/about-pipeline-architecture.png"
            alt="Diagram: data sources (MLB API, Open-Meteo, odds, Statcast) flow through Cloud Run jobs into Cloud SQL, then training and daily inference, to the React dashboard."
            style={{ width: "100%", height: "auto", display: "block", borderRadius: 8 }}
          />
        </div>
        <figcaption style={{ fontSize: 12, color: COL.textMuted, marginTop: 10, lineHeight: 1.4 }}>
          End-to-end flow from sources to Cloud Run, Cloud SQL, model training and inference, and the live dashboard.
        </figcaption>
      </figure>
      <p style={p}>
        The full system runs on Google Cloud Platform. Daily ingestion, feature engineering, and inference are handled by containerized jobs on Cloud Run, scheduled
        to produce morning forecasts before first pitch and refresh them as lineups and odds finalize. Model artifacts and feature state live in Cloud SQL (PostgreSQL). The
        dashboard is a React frontend that reads from the same database and polls live scores during games.
      </p>
      <p style={p}>
        The codebase is versioned, and each model release is tracked — the current production model is v9, which supersedes v8&apos;s run-estimation approach with
        calibrated probabilities and a cleaner feature set.
      </p>

      <h2 style={h2}>How accuracy is measured</h2>
      <p style={p}>
        Model performance is tracked on held-out games the model has never seen during training. The dashboard&apos;s Model accuracy tab shows rolling performance
        over recent windows, including how often win-probability calls were correct at various confidence levels and how model-projected totals compared to actual
        outcomes. Results are reported honestly, including days and weeks where the model performs worse than the market.
      </p>
      <p style={p}>
        Evaluating a prediction system fairly requires time. A model that looks sharp over a few weeks can look average over a full season, and baseball is a high-variance
        sport where even the strongest edge takes thousands of games to show up cleanly in the results.
      </p>

      <h2 style={h2}>Honest limitations</h2>
      <p style={p}>
        A few things this model is not, and does not claim to be:
      </p>
      <ul style={{ ...p, paddingLeft: 22, marginTop: 0 }}>
        <li style={li}>
          It is not a betting service. It makes no recommendations about which games to wager on, and no result on this site should be treated as investment advice.
        </li>
        <li style={li}>
          It does not model in-game state. All predictions are pre-game; the model does not update mid-game based on live play-by-play.
        </li>
        <li style={li}>
          It does not incorporate injuries or late scratches beyond what shows up in the announced lineup. A last-minute position-player change after the lineup posts is
          not reflected.
        </li>
        <li style={li}>
          It is a single model&apos;s opinion. Baseball is noisy, and even a well-built model is often wrong on individual games — the goal is accuracy across many games, not
          certainty on any one.
        </li>
      </ul>

      <h2 style={h2}>About the project</h2>
      <p style={p}>
        MLB Predictor is an independent research project built and maintained by one person as a serious exercise in end-to-end machine learning engineering: data
        ingestion, feature engineering, model training and evaluation, cloud deployment, and production monitoring. It is not affiliated with Major League Baseball, any
        team, or any sportsbook.
      </p>
      <p style={p}>
        The project exists because baseball is a uniquely good testbed for ML — it&apos;s data-rich, it has a long season that generates many independent trials, it
        has a strong public baseline (the betting market) to measure against, and it&apos;s genuinely fun to work on.
      </p>
      <p style={{ ...p, marginBottom: 0 }}>
        For questions, feedback, or technical discussion, feel free to reach out at{" "}
        <a
          href="mailto:contact@mlbpredictor.com"
          style={{ color: COL.model, fontWeight: 700, textDecoration: "none" }}
        >
          contact@mlbpredictor.com
        </a>
        .
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Agent Chat Widget (Claude-powered assistant, bottom-right popup)  */
/* ------------------------------------------------------------------ */

const CHAT_SUGGESTIONS = [
  "Who does the model like today?",
  "How is the model doing this week?",
  "Explain this game",
  "Which games have the biggest edges?",
];

/** Tiny inline markdown renderer: **bold**, *italic*, `code`. Safe: no HTML, no links. */
function renderChatMarkdown(text) {
  if (!text) return null;
  // Split on paragraph boundaries to preserve line structure; keep trailing empty lines.
  const lines = String(text).split(/\n/);
  const nodes = [];
  lines.forEach((line, i) => {
    nodes.push(...renderInline(line, i));
    if (i < lines.length - 1) nodes.push(<br key={`br-${i}`} />);
  });
  return nodes;
}

function renderInline(line, lineIndex) {
  // Regex matches **bold**, *italic* (not preceded by alnum), or `code`.
  const re = /(\*\*([^*\n]+)\*\*)|(?:(?<![A-Za-z0-9])\*([^*\n]+)\*(?![A-Za-z0-9]))|(`([^`\n]+)`)/g;
  const out = [];
  let lastIdx = 0;
  let key = 0;
  let m;
  while ((m = re.exec(line)) !== null) {
    if (m.index > lastIdx) out.push(line.slice(lastIdx, m.index));
    if (m[2] != null) {
      out.push(<strong key={`l${lineIndex}-b-${key++}`} style={{ fontWeight: 800 }}>{m[2]}</strong>);
    } else if (m[3] != null) {
      out.push(<em key={`l${lineIndex}-i-${key++}`}>{m[3]}</em>);
    } else if (m[5] != null) {
      out.push(
        <code key={`l${lineIndex}-c-${key++}`} style={{
          background: "rgba(15,23,42,0.06)",
          padding: "1px 5px",
          borderRadius: 4,
          fontSize: "0.92em",
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        }}>{m[5]}</code>
      );
    }
    lastIdx = m.index + m[0].length;
  }
  if (lastIdx < line.length) out.push(line.slice(lastIdx));
  return out;
}

function AgentChatWidget({ context }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState(() => [
    {
      role: "assistant",
      content: "Hi! I'm the MLB Predictor assistant. Ask me about today's slate, why the model favors a team, or how it's been doing lately. ⚾\n\n*These are model outputs, not guaranteed picks — please wager responsibly.*",
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);

  const listRef = useCallback((node) => {
    if (node) node.scrollTop = node.scrollHeight;
  }, []);

  const sendMessage = async (textOverride) => {
    const text = (textOverride ?? input).trim();
    if (!text || sending) return;
    const nextMessages = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setInput("");
    setSending(true);
    setError(null);
    try {
      const body = {
        messages: nextMessages.map(m => ({ role: m.role, content: m.content })),
        context,
      };
      const r = await fetch(CHAT_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (r.status === 429) {
        setError("You've hit today's message limit. Please try again tomorrow.");
        setSending(false);
        return;
      }
      if (!r.ok) {
        let detail = "";
        try { detail = (await r.json())?.message || ""; } catch {}
        throw new Error(`HTTP ${r.status}${detail ? `: ${detail}` : ""}`);
      }
      const data = await r.json();
      setMessages([...nextMessages, { role: "assistant", content: data.reply || "(no reply)" }]);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      {/* Floating launcher */}
      <button
        type="button"
        aria-label={open ? "Close MLB assistant" : "Open MLB assistant"}
        onClick={() => setOpen(v => !v)}
        style={{
          position: "fixed",
          bottom: 20,
          right: 20,
          zIndex: 2147483000,
          width: 58,
          height: 58,
          borderRadius: "50%",
          border: "none",
          background: "linear-gradient(135deg, #ea580c 0%, #c2410c 100%)",
          color: "#fff",
          boxShadow: "0 8px 24px rgba(234,88,12,0.45)",
          cursor: "pointer",
          fontSize: 26,
          fontFamily: "inherit",
          transition: "transform 0.15s ease",
          transform: open ? "scale(0.95)" : "scale(1)",
        }}
      >
        {open ? "×" : "⚾"}
      </button>

      {open && (
        <div
          style={{
            position: "fixed",
            bottom: 90,
            right: 20,
            zIndex: 2147482999,
            width: 380,
            maxWidth: "calc(100vw - 32px)",
            height: 560,
            maxHeight: "calc(100vh - 120px)",
            background: "#fff",
            borderRadius: 16,
            boxShadow: "0 24px 60px rgba(15,23,42,0.25)",
            border: `1px solid ${COL.border}`,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            fontFamily: "inherit",
          }}
        >
          <div style={{
            background: "linear-gradient(135deg, #1a202c 0%, #2d3748 100%)",
            padding: "14px 18px",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
          >
            <img
              src="/mlb-predictor-mark-64.png"
              alt=""
              width={34}
              height={34}
              style={{ display: "block", borderRadius: 8, flexShrink: 0, mixBlendMode: "screen" }}
            />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 14, fontWeight: 800, letterSpacing: 0.2 }}>MLB Predictor</div>
              <div style={{ fontSize: 11, fontWeight: 500, color: "rgba(255,255,255,0.65)" }}>
                Powered by Claude · {context?.game_id ? "Viewing game" : (context?.date ? `Slate ${context.date}` : "Today")}
              </div>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close"
              style={{
                background: "transparent",
                border: "none",
                color: "rgba(255,255,255,0.75)",
                fontSize: 22,
                cursor: "pointer",
                padding: "0 4px",
              }}
            >×</button>
          </div>

          <div
            ref={listRef}
            style={{
              flex: 1,
              overflowY: "auto",
              padding: 14,
              background: "#F8FAFC",
              display: "flex",
              flexDirection: "column",
              gap: 10,
            }}
          >
            {messages.map((m, i) => (
              <div
                key={i}
                style={{
                  alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                  maxWidth: "88%",
                  padding: "10px 13px",
                  borderRadius: m.role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                  background: m.role === "user" ? COL.model : "#fff",
                  color: m.role === "user" ? "#fff" : COL.text,
                  border: m.role === "user" ? "none" : `1px solid ${COL.border}`,
                  fontSize: 13.5,
                  lineHeight: 1.5,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  boxShadow: m.role === "assistant" ? "0 1px 2px rgba(15,23,42,0.04)" : "none",
                }}
              >
                {m.role === "assistant" ? renderChatMarkdown(m.content) : m.content}
              </div>
            ))}
            {sending && (
              <div style={{
                alignSelf: "flex-start",
                padding: "10px 13px",
                borderRadius: "14px 14px 14px 4px",
                background: "#fff",
                border: `1px solid ${COL.border}`,
                fontSize: 13,
                color: COL.textMuted,
                fontStyle: "italic",
              }}>
                Thinking…
              </div>
            )}
            {error && (
              <div style={{
                alignSelf: "stretch",
                padding: "8px 10px",
                borderRadius: 8,
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.25)",
                fontSize: 12,
                color: "#b91c1c",
              }}>
                {error}
              </div>
            )}
            {messages.length <= 1 && !sending && (
              <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
                {CHAT_SUGGESTIONS.map(s => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => sendMessage(s)}
                    style={{
                      textAlign: "left",
                      padding: "8px 12px",
                      borderRadius: 10,
                      border: `1px solid ${COL.border}`,
                      background: "#fff",
                      color: COL.text,
                      fontSize: 12.5,
                      fontWeight: 600,
                      cursor: "pointer",
                      fontFamily: "inherit",
                    }}
                  >{s}</button>
                ))}
              </div>
            )}
          </div>

          <form
            onSubmit={(e) => { e.preventDefault(); sendMessage(); }}
            style={{
              display: "flex",
              gap: 8,
              padding: 10,
              borderTop: `1px solid ${COL.border}`,
              background: "#fff",
            }}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about the model…"
              disabled={sending}
              style={{
                flex: 1,
                border: `1px solid ${COL.border}`,
                borderRadius: 10,
                padding: "9px 12px",
                fontSize: 13.5,
                outline: "none",
                fontFamily: "inherit",
                color: COL.text,
              }}
            />
            <button
              type="submit"
              disabled={sending || !input.trim()}
              style={{
                background: COL.model,
                color: "#fff",
                border: "none",
                borderRadius: 10,
                padding: "0 14px",
                fontWeight: 700,
                fontSize: 13,
                cursor: sending || !input.trim() ? "not-allowed" : "pointer",
                opacity: sending || !input.trim() ? 0.55 : 1,
                fontFamily: "inherit",
              }}
            >Send</button>
          </form>
          <div style={{ padding: "4px 12px 8px", background: "#fff", fontSize: 10, color: COL.textMuted, textAlign: "center" }}>
            Replies are model outputs, not betting advice.
          </div>
        </div>
      )}
    </>
  );
}

function ScheduleDateStrip({ date, todayStr, onPick, refreshing, onManualRefresh, statusLine }) {
  const navBtn = (label, ch, delta) => (
    <button
      type="button"
      aria-label={label}
      onClick={() => onPick(addDaysToYmd(date, delta))}
      style={{
        width: 32,
        height: 32,
        borderRadius: 8,
        border: `1px solid ${COL.pageBorder}`,
        background: "#fff",
        color: "#1e293b",
        fontSize: 17,
        fontWeight: 700,
        lineHeight: 1,
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "inherit",
        flexShrink: 0,
      }}
    >{ch}</button>
  );
  const days = [-2, -1, 0, 1, 2];
  return (
    <div
      style={{
        width: "100%",
        background: "#fff",
        border: `1px solid ${COL.pageBorder}`,
        borderRadius: 12,
        boxShadow: "0 1px 3px rgba(15,23,42,0.06)",
        padding: "10px 12px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-start", gap: 5, flex: "1 1 280px", minWidth: 0 }}>
          {navBtn("Previous day", "‹", -1)}
          {days.map((off) => {
            const d = addDaysToYmd(date, off);
            const isSel = off === 0;
            return (
              <button
                key={d}
                type="button"
                onClick={() => onPick(d)}
                style={{
                  minWidth: 58,
                  padding: "5px 4px 6px",
                  borderRadius: 9,
                  border: isSel ? `1.5px solid ${COL.model}` : `1px solid ${COL.pageBorder}`,
                  background: isSel ? "rgba(37,99,235,0.08)" : "#fff",
                  color: isSel ? COL.model : "#1e293b",
                  fontFamily: "inherit",
                  cursor: "pointer",
                  textAlign: "center",
                  flex: "0 0 auto",
                }}
              >
                <div style={{ fontSize: 8.5, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: isSel ? COL.model : "#64748b" }}>
                  {weekdayShort(d)}
                </div>
                <div style={{ fontSize: 12.5, fontWeight: 800, fontVariantNumeric: "tabular-nums", marginTop: 1 }}>{shortDateLabel(d)}</div>
              </button>
            );
          })}
          {navBtn("Next day", "›", 1)}
        </div>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
          flexWrap: "wrap",
          marginTop: 8,
          paddingTop: 8,
          borderTop: `1px solid ${COL.pageBorder}`,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {date !== todayStr && (
            <button
              type="button"
              onClick={() => onPick(todayStr)}
              style={{
                fontFamily: "inherit",
                fontSize: 11.5,
                fontWeight: 700,
                padding: "5px 10px",
                borderRadius: 6,
                border: "none",
                background: "#2563EB",
                color: "#fff",
                cursor: "pointer",
              }}
            >Today</button>
          )}
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 4,
              background: "#f8fafc",
              border: `1px solid ${COL.pageBorder}`,
              borderRadius: 8,
              padding: "0 8px 0 10px",
              fontSize: 12,
              color: "#64748B",
              fontWeight: 600,
            }}
          >
            <span>Calendar</span>
            <input
              type="date"
              value={date}
              onChange={(e) => onPick(e.target.value)}
              style={{
                border: "none",
                background: "transparent",
                color: "#1e293b",
                fontWeight: 700,
                fontSize: 12.5,
                fontFamily: "inherit",
                padding: "5px 0",
                cursor: "pointer",
                maxWidth: 132,
              }}
            />
          </div>
          <button
            type="button"
            onClick={onManualRefresh}
            disabled={!!refreshing}
            title="Refresh games"
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              border: `1px solid ${COL.pageBorder}`,
              background: "#fff",
              color: "#64748b",
              fontSize: 15,
            cursor: refreshing ? "wait" : "pointer",
            opacity: refreshing ? 0.6 : 1,
            fontFamily: "inherit",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            }}
          >{refreshing ? "…" : "↻"}</button>
        </div>
        {statusLine && (
          <span style={{ fontSize: 11, color: "#64748B", fontWeight: 500, marginLeft: "auto", textAlign: "right", whiteSpace: "nowrap" }}>
            {statusLine}
          </span>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [date, setDate] = useState(getInitialDateFromUrl);
  const [detailGameId, setDetailGameId] = useState(readHashGameId);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [mainTab, setMainTab] = useState("games");
  const [accuracyRefreshKey, setAccuracyRefreshKey] = useState(0);
  const prevMainTabRef = useRef(null);
  const prevFinalGameIdsRef = useRef(null);

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
  const allStandings = useAllTeamStandings(seasonYear, games.length > 0);

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

  // eslint-disable-next-line react-hooks/set-state-in-effect -- reset loading then async fetch
  useEffect(() => { setLoading(true); fetchGames(); }, [date, fetchGames]);

  // Reset “new final” detection when the displayed date changes
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { prevFinalGameIdsRef.current = null; }, [date]);

  // Re-fetch model accuracy when any slate game is newly final (MLB live feed or game status in API)
  useEffect(() => {
    if (!games.length) return;
    const nowFinal = new Set();
    for (const g of games) {
      if (g == null || g.game_id == null) continue;
      const lr = live?.[g.game_id];
      const detailed = lr?.status ?? g.status ?? "";
      const abstract = lr?.abstractGameState ?? null;
      const coded = lr?.codedGameState ?? null;
      if (isMlbGameFinished(detailed, abstract, coded)) {
        nowFinal.add(String(g.game_id));
      }
    }
    if (prevFinalGameIdsRef.current == null) {
      prevFinalGameIdsRef.current = nowFinal;
      return;
    }
    for (const id of nowFinal) {
      if (!prevFinalGameIdsRef.current.has(id)) {
        setAccuracyRefreshKey((k) => k + 1);
        break;
      }
    }
    prevFinalGameIdsRef.current = nowFinal;
  }, [live, games]);

  // Fresh accuracy pull when user opens the Model Accuracy tab
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => {
    if (mainTab === "accuracy" && prevMainTabRef.current !== "accuracy" && !detailGameId) {
      setAccuracyRefreshKey((k) => k + 1);
    }
    prevMainTabRef.current = mainTab;
  }, [mainTab, detailGameId]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (isGameHours()) fetchGames(true);
    }, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchGames]);

  const detailGame = useMemo(
    () => sortedGames.find((x) => String(x.game_id) === String(detailGameId)),
    [sortedGames, detailGameId],
  );

  const openGameDetail = useCallback((pk) => {
    const url = new URL(window.location.href);
    url.searchParams.set("date", date);
    url.hash = `game=${pk}`;
    window.history.pushState({}, "", url);
    setDetailGameId(String(pk));
  }, [date]);

  const closeGameDetail = useCallback(() => {
    const url = new URL(window.location.href);
    url.hash = "";
    window.history.pushState({}, "", url);
    setDetailGameId(null);
  }, []);

  useEffect(() => {
    const onHash = () => setDetailGameId(readHashGameId());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.gtag !== "function") return;
    let title = "MLB Predictor — Games";
    let path = "/";
    if (detailGameId) {
      const label = detailGame
        ? `${detailGame.away_team} @ ${detailGame.home_team}`
        : `Game ${detailGameId}`;
      title = `MLB Predictor — ${label}`;
      path = `/game/${detailGameId}`;
    } else if (mainTab === "accuracy") {
      title = "MLB Predictor — Model Accuracy";
      path = "/accuracy";
    } else if (mainTab === "about") {
      title = "MLB Predictor — About";
      path = "/about";
    }
    window.gtag("event", "page_view", {
      page_title: title,
      page_location: window.location.href,
      page_path: path,
    });
  }, [mainTab, detailGameId, detailGame]);

  useEffect(() => {
    const onPop = () => {
      setDetailGameId(readHashGameId());
      try {
        const p = new URLSearchParams(window.location.search).get("date");
        if (p && /^\d{4}-\d{2}-\d{2}$/.test(p)) setDate(p);
      } catch {
        /* ignore */
      }
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const applyDate = useCallback((d) => {
    setDate(d);
    setDetailGameId(null);
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("date", d);
      url.hash = "";
      window.history.replaceState({}, "", url);
    } catch {
      /* ignore */
    }
  }, []);

  return (
    <div style={{ background: COL.bg, minHeight: "100vh", width: "100%", fontFamily: "'SF Pro Display',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" }}>
      <style>{`
        * { box-sizing: border-box; }
        input[type="date"]::-webkit-calendar-picker-indicator { filter: invert(0.35); opacity: 0.7; }
        ::-webkit-scrollbar { height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 2px; }
      `}</style>

      <div style={{ maxWidth: detailGameId ? 1160 : 1200, margin: "0 auto", padding: "0 12px 24px" }}>

        <div
          style={{
            position: "sticky",
            top: 0,
            zIndex: 200,
            background: COL.bg,
            margin: "0 -12px 0",
            padding: "0 12px 8px",
            borderBottom: `1px solid ${COL.pageBorder}`,
            boxShadow: "0 1px 0 rgba(15,23,42,0.04)",
          }}
        >
          <div
            style={{
              background: "#0F172A",
              borderRadius: "0 0 10px 10px",
              border: `1px solid ${COL.pageBorder}`,
              borderTop: "none",
              maxWidth: detailGameId ? 1160 : 1200,
              margin: "0 auto",
              padding: "14px 16px 16px",
              boxShadow: "0 2px 8px rgba(15,23,42,0.12)",
            }}
          >
            <div
              style={{
                height: 1.5,
                borderRadius: 1,
                background: "linear-gradient(90deg, rgba(234,88,12,0.7), rgba(37,99,235,0.75), rgba(34,197,94,0.5))",
                marginBottom: 12,
                opacity: 0.9,
              }}
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 16,
                flexWrap: "wrap",
                rowGap: 12,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", flex: "0 0 auto" }}>
                <h1 style={{ margin: 0, lineHeight: 0 }}>
                  <img
                    src="/mlb-predictor-logo.jpg"
                    alt="MLB Predictor"
                    style={{
                      display: "block",
                      width: "auto",
                      height: "clamp(76px, 11vw, 110px)",
                      maxWidth: "min(calc(100vw - 40px), 900px)",
                      objectFit: "contain",
                      objectPosition: "left center",
                      mixBlendMode: "screen",
                    }}
                  />
                </h1>
              </div>
              {(!detailGameId) && (
                <div
                  style={{
                    display: "inline-flex",
                    padding: 2,
                    gap: 2,
                    background: "rgba(15,23,42,0.5)",
                    border: "1px solid rgba(148,163,184,0.2)",
                    borderRadius: 8,
                    flexShrink: 0,
                  }}
                >
                  {[
                    { id: "games", label: "Games" },
                    { id: "accuracy", label: "Model accuracy" },
                    { id: "about", label: "About us" },
                  ].map((t) => {
                    const active = mainTab === t.id;
                    return (
                      <button
                        key={t.id}
                        type="button"
                        onClick={() => setMainTab(t.id)}
                        style={{
                          fontFamily: "inherit",
                          fontSize: 12,
                          fontWeight: 700,
                          padding: "6px 14px",
                          borderRadius: 6,
                          border: "none",
                          cursor: "pointer",
                          background: active ? "#2563EB" : "transparent",
                          color: active ? "#fff" : "rgba(226,232,240,0.8)",
                          transition: "background 0.15s, color 0.15s",
                        }}
                      >
                        {t.label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
          {(detailGameId || mainTab === "games") && (
            <div style={{ maxWidth: detailGameId ? 1160 : 1200, margin: "8px auto 0", paddingTop: 2 }}>
              <ScheduleDateStrip
                date={date}
                todayStr={today}
                onPick={applyDate}
                refreshing={refreshing}
                onManualRefresh={() => fetchGames(true)}
                statusLine={
                  lastUpdated
                    ? `Updated ${formatTime(lastUpdated)}${isGameHours() ? " · auto-refreshing" : ""}`
                    : null
                }
              />
            </div>
          )}
        </div>

        {detailGameId && loading && (
          <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "2rem" }}>Loading…</p>
        )}
        {detailGameId && !loading && !detailGame && (
          <div style={{ padding: "24px 0 8px", textAlign: "center" }}>
            <p style={{ color: COL.textSecondary, fontSize: 14, marginBottom: 16 }}>No game <code style={{ fontSize: 13 }}>{detailGameId}</code> on {date}. Try another date.</p>
            <button
              type="button"
              onClick={closeGameDetail}
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: COL.model,
                background: COL.controlBg,
                border: `1px solid ${COL.controlBorder}`,
                borderRadius: 10,
                padding: "8px 16px",
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              ← Back to schedule
            </button>
          </div>
        )}
        {detailGameId && !loading && detailGame && (
          <GameDetailRoute
            g={detailGame}
            live={live}
            enrich={enrich}
            seasonYear={seasonYear}
            onBack={closeGameDetail}
            lastUpdatedAt={lastUpdated}
          />
        )}
        {!detailGameId && mainTab === "games" && loading && (
          <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "3rem" }}>Loading games...</p>
        )}
        {!detailGameId && mainTab === "games" && !loading && games.length === 0 && (
          <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "3rem" }}>
            Lineups are not yet confirmed for these games.
          </p>
        )}
        {!detailGameId && mainTab === "games" && !loading && games.length > 0 && (
          <GamesTable
            sortedGames={sortedGames}
            live={live}
            onOpenDetail={openGameDetail}
            standingsMap={allStandings}
          />
        )}
        {!detailGameId && mainTab === "accuracy" && (
          <ModelAccuracyPanel enabled refreshKey={accuracyRefreshKey} />
        )}
        {!detailGameId && mainTab === "about" && (
          <AboutUsPanel />
        )}

      </div>
      <AgentChatWidget
        context={{
          date,
          game_id: detailGameId ? Number(detailGameId) : undefined,
          away_team: detailGame?.away_team,
          home_team: detailGame?.home_team,
        }}
      />
    </div>
  );
}
