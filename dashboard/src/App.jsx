import { useEffect, useState, useCallback, useMemo, useRef, Fragment } from "react";
import {
  BatterPropLineupPair,
  displayBattingOrder,
  filterBattersForDisplay,
  prepareGameBatterLineups,
} from "./BatterPropTable.jsx";

const API = "https://us-central1-mlb-model-491223.cloudfunctions.net/get-daily-predictions";

function pacificTodayStr() {
  return new Date().toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });
}

function useIsMobile(breakpoint = 768) {
  const [isMobile, setIsMobile] = useState(
    typeof window !== "undefined"
      ? window.matchMedia(`(max-width: ${breakpoint}px)`).matches
      : false
  );
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia(`(max-width: ${breakpoint}px)`);
    const onChange = (e) => setIsMobile(e.matches);
    mq.addEventListener("change", onChange);
    setIsMobile(mq.matches);
    return () => mq.removeEventListener("change", onChange);
  }, [breakpoint]);
  return isMobile;
}

const MOBILE_HEADER_H = 56;
const PAGE_HEADER_TOP_PADDING = 36;

function pageShellStyle(maxWidth, { top = PAGE_HEADER_TOP_PADDING, bottom = 48, horizontal = 0 } = {}) {
  const h = horizontal ? ` ${horizontal}px` : "";
  return {
    maxWidth,
    margin: "0 auto",
    padding: `${top}px${h} ${bottom}px${h}`,
  };
}

const NAV_TABS = [
  { id: "games", label: "Games" },
  { id: "edges", label: "Top Edges" },
  { id: "players", label: "Players" },
  { id: "trends", label: "Trends" },
  { id: "standings", label: "Standings" },
  { id: "transactions", label: "Transactions" },
  { id: "accuracy", label: "Model performance" },
  { id: "about", label: "About" },
];

let trendsCache = null;
let trendsCacheDate = null;
let trendsFetchPromise = null;

function fetchTrendsData({ force = false, date = null } = {}) {
  const d = date || pacificTodayStr();
  if (!force && trendsCache && trendsCacheDate === d) {
    return Promise.resolve(trendsCache);
  }
  if (trendsFetchPromise && !force) {
    return trendsFetchPromise;
  }
  trendsFetchPromise = fetch(`${API}?view=trends&date=${encodeURIComponent(d)}`)
    .then((r) => parseAccuracyResponse(r))
    .then((data) => {
      trendsCache = data;
      trendsCacheDate = d;
      trendsFetchPromise = null;
      return data;
    })
    .catch((err) => {
      trendsFetchPromise = null;
      throw err;
    });
  return trendsFetchPromise;
}

function prefetchTrends() {
  const today = pacificTodayStr();
  if (trendsCache && trendsCacheDate === today) return;
  fetchTrendsData().catch(() => {});
}

let edgesCache = null;
let edgesCacheDate = null;
let edgesFetchPromise = null;

function fetchEdgesData(date, { force = false } = {}) {
  if (!date) return Promise.reject(new Error("date required"));
  if (!force && edgesCache && edgesCacheDate === date) {
    return Promise.resolve(edgesCache);
  }
  if (edgesFetchPromise && !force && edgesCacheDate === date) {
    return edgesFetchPromise;
  }
  edgesFetchPromise = fetch(`${API}?view=edges&date=${encodeURIComponent(date)}`)
    .then((r) => parseAccuracyResponse(r))
    .then((data) => {
      edgesCache = data;
      edgesCacheDate = date;
      edgesFetchPromise = null;
      return data;
    })
    .catch((err) => {
      edgesFetchPromise = null;
      throw err;
    });
  return edgesFetchPromise;
}

function prefetchEdges(date) {
  if (!date) return;
  if (edgesCache && edgesCacheDate === date) return;
  fetchEdgesData(date).catch(() => {});
}

let standingsCache = null;
let standingsCacheDate = null;
let standingsFetchPromise = null;

function fetchStandingsData(date, { force = false } = {}) {
  const d = date || pacificTodayStr();
  if (!force && standingsCache && standingsCacheDate === d) {
    return Promise.resolve(standingsCache);
  }
  if (standingsFetchPromise && !force && standingsCacheDate === d) {
    return standingsFetchPromise;
  }
  standingsFetchPromise = fetch(`${API}?view=standings&date=${encodeURIComponent(d)}`)
    .then((r) => parseAccuracyResponse(r))
    .then((data) => {
      standingsCache = data;
      standingsCacheDate = d;
      standingsFetchPromise = null;
      return data;
    })
    .catch((err) => {
      standingsFetchPromise = null;
      throw err;
    });
  return standingsFetchPromise;
}

function prefetchStandings(date) {
  const d = date || pacificTodayStr();
  if (standingsCache && standingsCacheDate === d) return;
  fetchStandingsData(d).catch(() => {});
}

const transactionsCache = new Map();
let transactionsFetchPromise = null;

function fetchTransactionsData(date, days = 14, { force = false } = {}) {
  const d = date || pacificTodayStr();
  const safeDays = Math.max(1, Math.min(60, Number(days) || 14));
  const key = `${d}:${safeDays}`;
  if (!force && transactionsCache.has(key)) {
    return Promise.resolve(transactionsCache.get(key));
  }
  if (transactionsFetchPromise && !force) {
    return transactionsFetchPromise;
  }
  transactionsFetchPromise = fetch(`${API}?view=transactions&date=${encodeURIComponent(d)}&days=${safeDays}`)
    .then((r) => parseAccuracyResponse(r))
    .then((data) => {
      transactionsCache.set(key, data);
      transactionsFetchPromise = null;
      return data;
    })
    .catch((err) => {
      transactionsFetchPromise = null;
      throw err;
    });
  return transactionsFetchPromise;
}

function prefetchTransactions(date) {
  const d = date || pacificTodayStr();
  if (transactionsCache.has(`${d}:14`)) return;
  fetchTransactionsData(d, 14).catch(() => {});
}

const playersCache = new Map();
let playersFetchPromise = null;

function fetchPlayersData(date, { force = false } = {}) {
  const d = date || pacificTodayStr();
  if (!force && playersCache.has(d)) {
    return Promise.resolve(playersCache.get(d));
  }
  if (playersFetchPromise && !force) {
    return playersFetchPromise;
  }
  playersFetchPromise = fetch(`${API}?view=players&date=${encodeURIComponent(d)}`)
    .then((r) => parseAccuracyResponse(r))
    .then((data) => {
      playersCache.set(d, data);
      playersFetchPromise = null;
      return data;
    })
    .catch((err) => {
      playersFetchPromise = null;
      throw err;
    });
  return playersFetchPromise;
}

function prefetchPlayers(date) {
  const d = date || pacificTodayStr();
  if (playersCache.has(d)) return;
  fetchPlayersData(d).catch(() => {});
}
const CHAT_API = "https://us-central1-mlb-model-491223.cloudfunctions.net/mlb-agent-chat";
const MLB_API = "https://statsapi.mlb.com/api/v1.1/game";

const MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule";
const MLB_BOX = "https://statsapi.mlb.com/api/v1/game";
const MLB_PEOPLE = "https://statsapi.mlb.com/api/v1/people";
const today = new Date().toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });
const REFRESH_INTERVAL = 45000;

/** Min |market O/U line − predicted total| to show Over/Under vs Pass (aligned with inference + grading). */
const OU_LINE_GAP_THRESHOLD = 0.5;

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
  return new Date(Y, (M || 1) - 1, D || 1).toLocaleDateString("en-US", { weekday: "short", timeZone: "America/Los_Angeles" }).toUpperCase();
}

/** Day number for date strip; includes month when crossing a month boundary. */
function dayStripNumberLabel(ymd, prevYmd) {
  const [y, m, d] = ymd.split("-").map((n) => parseInt(n, 10));
  if (!y) return ymd;
  if (prevYmd) {
    const [, pm] = prevYmd.split("-").map((n) => parseInt(n, 10));
    if (pm !== m) {
      return new Date(y, m - 1, d).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/Los_Angeles" });
    }
  }
  return String(d);
}

function formatSectionScheduleDate(ymd) {
  const [y, m, d] = ymd.split("-").map((n) => parseInt(n, 10));
  if (!y) return ymd;
  return new Date(y, m - 1, d).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/Los_Angeles" });
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

/** UI palette — The Hot Corner design tokens */
const COL = {
  bg: "#111827",
  pageBorder: "#374151",
  card: "#1F2937",
  cardInner: "#172032",
  border: "#374151",
  model: "#F59E0B",
  modelTint: "rgba(245,158,11,0.12)",
  accentGold: "#D97706",
  market: "#3B82F6",
  marketMuted: "#6B7280",
  positive: "#16A34A",
  negative: "#DC2626",
  text: "#F9FAFB",
  textSecondary: "#9CA3AF",
  textMuted: "#6B7280",
  controlBg: "#1F2937",
  controlBorder: "#374151",
  logoBg: "#374151",
};
const FONT_BODY = "'Inter', system-ui, sans-serif";
const FONT_DISPLAY = "'Bebas Neue', sans-serif";
const FONT_MONO = "'JetBrains Mono', ui-monospace, monospace";
const GAMES_REC_STACK_WIDTH = 72;
const GAMES_TIME_MIN_HEIGHT = 40;
const PANEL_BG = "#0D1420";
const PANEL_BORDER = "#1F2937";
const CSS_TEXT = "var(--color-text-primary)";
const CSS_CARD = "var(--color-bg-card)";
const CSS_CARD_ALT = "var(--color-bg-primary)";

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

const MLB_TEAM_ID_TO_FULL = Object.fromEntries(
  Object.entries(TEAM_IDS).map(([name, id]) => [id, name]),
);

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

const TEAM_BADGE_COLORS = {
  "Arizona Diamondbacks": "#A71930",
  "Atlanta Braves": "#CE1141",
  "Baltimore Orioles": "#DF4601",
  "Boston Red Sox": "#BD3039",
  "Chicago Cubs": "#0E3386",
  "Chicago White Sox": "#27251F",
  "Cincinnati Reds": "#C6011F",
  "Cleveland Guardians": "#E31937",
  "Colorado Rockies": "#33006F",
  "Detroit Tigers": "#0C2340",
  "Houston Astros": "#EB6E1F",
  "Kansas City Royals": "#004687",
  "Los Angeles Angels": "#BA0021",
  "Los Angeles Dodgers": "#005A9C",
  "Miami Marlins": "#00A3E0",
  "Milwaukee Brewers": "#FFC52F",
  "Minnesota Twins": "#002B5C",
  "New York Mets": "#FF5910",
  "New York Yankees": "#0C2340",
  "Oakland Athletics": "#003831",
  "Athletics": "#003831",
  "Philadelphia Phillies": "#E81828",
  "Pittsburgh Pirates": "#FDB827",
  "San Diego Padres": "#2F241D",
  "San Francisco Giants": "#FD5A1E",
  "Seattle Mariners": "#005C5C",
  "St. Louis Cardinals": "#C41E3A",
  "Tampa Bay Rays": "#092C5C",
  "Texas Rangers": "#003278",
  "Toronto Blue Jays": "#134A8E",
  "Washington Nationals": "#AB0003",
};

function teamAbbr(full) {
  if (!full) return "—";
  return TEAM_ABBR[full] || full.split(/\s+/).pop()?.slice(0, 3).toUpperCase() || "—";
}

/** Team nickname for compact layouts (logo carries city identity). */
function teamNickname(full) {
  if (!full) return "—";
  if (full === "Chicago White Sox" || full === "Boston Red Sox") {
    return full.split(/\s+/).slice(-2).join(" ");
  }
  if (full === "Toronto Blue Jays") return "Blue Jays";
  if (full === "Los Angeles Angels") return "Angels";
  if (full === "Los Angeles Dodgers") return "Dodgers";
  if (full === "New York Yankees") return "Yankees";
  if (full === "New York Mets") return "Mets";
  if (full === "San Francisco Giants") return "Giants";
  if (full === "San Diego Padres") return "Padres";
  if (full === "St. Louis Cardinals") return "Cardinals";
  if (full === "Tampa Bay Rays") return "Rays";
  if (full === "Kansas City Royals") return "Royals";
  if (full === "Colorado Rockies") return "Rockies";
  const parts = full.split(/\s+/);
  return parts[parts.length - 1] || full;
}

function teamBadgeColor(full) {
  return TEAM_BADGE_COLORS[full] || getTeamTheme(full).primary || COL.logoBg;
}

/** Relative luminance 0–1; below ~0.45 reads as dark on our card backgrounds. */
function hexLuminance(hex) {
  const h = (hex || "").replace("#", "");
  if (h.length !== 6) return 1;
  const r = parseInt(h.slice(0, 2), 16) / 255;
  const g = parseInt(h.slice(2, 4), 16) / 255;
  const b = parseInt(h.slice(4, 6), 16) / 255;
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/** Text accent on dark cards — white when team primary is too dark (e.g. Rockies purple). */
function teamAccentText(theme) {
  if (theme?.accentText) return theme.accentText;
  const primary = theme?.primary;
  if (primary && hexLuminance(primary) < 0.45) return COL.text;
  return primary ?? COL.model;
}

/** Dark-theme tint for team-colored table/card backgrounds (readable with light text). */
function teamSoftTint(primary) {
  const hex = (primary || COL.model).replace("#", "");
  if (hex.length !== 6) return COL.modelTint;
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, 0.22)`;
}

/** Primary / soft background for lineup cards & pred-run bars */
function getTeamTheme(fullTeamName) {
  const n = (fullTeamName || "").toLowerCase();
  const fallback = { primary: COL.model, soft: COL.modelTint, stroke: "rgba(245,158,11,0.35)", onPrimary: CSS_TEXT };
  const mk = (primary, stroke) => ({
    primary,
    soft: teamSoftTint(primary),
    stroke,
    onPrimary: CSS_TEXT,
    accentText: hexLuminance(primary) < 0.45 ? COL.text : primary,
  });
  if (n.includes("astros")) return mk("#EB6E1F", "rgba(235,110,31,0.38)");
  if (n.includes("guardians")) return mk("#5B7DB1", "rgba(91,125,177,0.35)");
  if (n.includes("yankees")) return mk("#5B7DB1", "rgba(91,125,177,0.35)");
  if (n.includes("dodgers")) return mk("#005A9C", "rgba(0,90,156,0.35)");
  if (n.includes("red sox")) return mk("#BD3039", "rgba(189,48,57,0.35)");
  if (n.includes("cubs")) return mk("#0E3386", "rgba(14,51,134,0.35)");
  if (n.includes("tigers")) return mk("#5B7DB1", "rgba(91,125,177,0.35)");
  if (n.includes("rockies")) return mk("#33006F", "rgba(51,0,111,0.35)");
  if (n.includes("braves")) return mk("#CE1141", "rgba(206,17,65,0.35)");
  if (n.includes("mets")) return mk("#FF5910", "rgba(255,89,16,0.35)");
  if (n.includes("phillies")) return mk("#E81828", "rgba(232,24,40,0.35)");
  if (n.includes("cardinals")) return mk("#C41E3A", "rgba(196,30,58,0.35)");
  if (n.includes("twins")) return mk("#5B7DB1", "rgba(91,125,177,0.35)");
  if (n.includes("rangers")) return mk("#003278", "rgba(0,50,120,0.35)");
  if (n.includes("orioles")) return mk("#DF4601", "rgba(223,70,1,0.35)");
  if (n.includes("reds")) return mk("#C6011F", "rgba(198,1,31,0.35)");
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
            <span style={{ fontSize: 13, fontWeight: 800, color: COL.text, fontVariantNumeric: "tabular-nums", fontFamily: FONT_MONO, width: 40, textAlign: "right", flexShrink: 0 }}>
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
  background: COL.bg,
  color: COL.text,
  fontSize: 10,
  fontWeight: 800,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  padding: "8px 8px",
  textAlign: "center",
};
const GP_STAT_BODY = {
  background: COL.cardInner,
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

/** Reject live/in-game ML pollution stored in morning columns (e.g. -15000). */
function saneMlPrice(price) {
  if (price === null || price === undefined) return null;
  const n = Number(price);
  if (!Number.isFinite(n) || n === 0) return null;
  if (Math.abs(n) > 500) return null;
  return n;
}

/** Pre-game ML from API overlay (first inference snapshot) or morning features. */
function pickPregameMlPrices(g) {
  const pregameHome = saneMlPrice(g?.pregame_home_price);
  const pregameAway = saneMlPrice(g?.pregame_away_price);
  if (pregameHome != null && pregameAway != null) {
    return { home: pregameHome, away: pregameAway };
  }

  const morningHome = saneMlPrice(g?.morning_home_price);
  const morningAway = saneMlPrice(g?.morning_away_price);
  if (morningHome != null && morningAway != null) {
    return { home: morningHome, away: morningAway };
  }

  const closingHome = saneMlPrice(g?.closing_home_price);
  const closingAway = saneMlPrice(g?.closing_away_price);
  if (closingHome != null && closingAway != null) {
    return { home: closingHome, away: closingAway };
  }

  const home = pregameHome ?? morningHome ?? closingHome;
  const away = pregameAway ?? morningAway ?? closingAway;
  return { home, away };
}

function pickPregameMarketPct(g) {
  const homeRaw = g?.market_p_home;
  const awayRaw = g?.market_p_away;
  const homeFromApi = homeRaw != null && homeRaw >= 8 && homeRaw <= 92 ? Number(homeRaw) : null;
  const awayFromApi = awayRaw != null && awayRaw >= 8 && awayRaw <= 92 ? Number(awayRaw) : null;
  const mornPh = g?.morning_p_home;
  const mornPa = mornPh != null ? 1 - Number(mornPh) : null;
  const sanePh = mornPh != null && mornPh >= 0.08 && mornPh <= 0.92 ? Number(mornPh) * 100 : null;
  const sanePa = mornPa != null && mornPa >= 0.08 && mornPa <= 0.92 ? mornPa * 100 : null;
  return {
    home: homeFromApi ?? sanePh,
    away: awayFromApi ?? sanePa,
  };
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
 * Pass (no directional bet) when abs(line minus predicted total) is below OU_LINE_GAP_THRESHOLD.
 */
function computeOU(totalPred, line) {
  if (line == null || totalPred == null) return null;
  const L = Number(line);
  const T = Number(totalPred);
  if (!Number.isFinite(L) || !Number.isFinite(T)) return null;
  if (Math.abs(L - T) < OU_LINE_GAP_THRESHOLD) return "push";
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

function formatGameDateLong(utcStr) {
  if (!utcStr) return null;
  const d = new Date(utcStr);
  return d.toLocaleDateString("en-US", {
    timeZone: "America/New_York",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

function formatFirstPitchET(utcStr) {
  if (!utcStr) return null;
  const d = new Date(utcStr);
  const t = d.toLocaleTimeString("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
  return `${t} ET`;
}

function formatMatchupAt(awayTeam, homeTeam) {
  return `${teamNickname(awayTeam)} at ${teamNickname(homeTeam)}`;
}

function formatLiveSituation(liveRow) {
  if (!liveRow) return null;
  const parts = [];
  if (liveRow.currentInning && liveRow.inningState) {
    parts.push(`${liveRow.inningState} ${ordinal(liveRow.currentInning)}`);
  }
  if (liveRow.outs != null) {
    parts.push(`${liveRow.outs} out${liveRow.outs === 1 ? "" : "s"}`);
  }
  const bases = [];
  if (liveRow.onFirst) bases.push("1st");
  if (liveRow.onSecond) bases.push("2nd");
  if (liveRow.onThird) bases.push("3rd");
  if (bases.length === 3) parts.push("Bases loaded");
  else if (bases.length === 2) parts.push(`Runners on ${bases.join(" & ")}`);
  else if (bases.length === 1) parts.push(`Runner on ${bases[0]}`);
  return parts.length ? parts.join(" · ") : null;
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

function parseBatterGameLine(line) {
  if (!line || typeof line !== "string") return null;
  const parts = line.split(",").map((p) => p.trim()).filter(Boolean);
  const first = parts[0] || "";
  const hitMatch = first.match(/^(\d+)\s*-\s*(\d+)/);
  if (!hitMatch) return null;

  const hits = Number(hitMatch[1]);
  const atBats = Number(hitMatch[2]);
  if (!Number.isFinite(hits) || !Number.isFinite(atBats)) return null;

  const countToken = (labelRe) => {
    for (const raw of parts.slice(1)) {
      const normalized = raw.replace(/[×x]/gi, "x").trim();
      let n = 1;
      let label = normalized;
      const counted = normalized.match(/^(\d+)\s+(2B|3B|HR|K|SO|BB)$/i)
        || normalized.match(/^(\d+)x(2B|3B|HR|K|SO|BB)$/i);
      if (counted) {
        n = Number(counted[1]);
        label = counted[2];
      }
      if (labelRe.test(label)) return Number.isFinite(n) ? n : 1;
    }
    return 0;
  };

  const doubles = countToken(/^2B$/i);
  const triples = countToken(/^3B$/i);
  const homeRuns = countToken(/^HR$/i);
  const strikeouts = countToken(/^(K|SO)$/i);
  const walks = countToken(/^BB$/i);
  const singles = Math.max(0, hits - doubles - triples - homeRuns);
  const totalBases = singles + (2 * doubles) + (3 * triples) + (4 * homeRuns);

  return { hits, atBats, doubles, triples, homeRuns, strikeouts, walks, totalBases };
}

function Logo({ team, size = 44 }) {
  const url = logoUrl(team);
  const ini = team ? team.split(" ").map(w => w[0]).slice(-2).join("") : "?";
  if (!url) return (
    <div
      className="team-logo-chip"
      aria-label={team || "Team"}
      style={{ width: size, height: size, fontSize: size * 0.28, fontWeight: 900, color: "#111827" }}
    >
      {ini}
    </div>
  );
  return (
    <div className="team-logo-chip" style={{ width: size, height: size }} aria-label={team || "Team"}>
      <img src={url} alt={team} onError={e => { e.target.style.display = "none"; }} />
    </div>
  );
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

/** Fixed-width O/U rec pill for the Games table REC column. */
function GamesOuRecPill({ children, color }) {
  const c = {
    green: { bg: "rgba(34,197,94,0.15)", text: COL.positive, bd: "rgba(34,197,94,0.35)" },
    red: { bg: "rgba(239,68,68,0.15)", text: COL.negative, bd: "rgba(239,68,68,0.35)" },
    gray: { bg: "rgba(107,114,128,0.12)", text: COL.textMuted, bd: "rgba(107,114,128,0.25)" },
  }[color] || { bg: "rgba(107,114,128,0.12)", text: COL.textMuted, bd: "rgba(107,114,128,0.25)" };
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        minWidth: GAMES_REC_STACK_WIDTH,
        width: GAMES_REC_STACK_WIDTH,
        boxSizing: "border-box",
        background: c.bg,
        color: c.text,
        border: `1px solid ${c.bd}`,
        padding: "4px 0",
        borderRadius: 100,
        fontSize: 11,
        fontWeight: 700,
        whiteSpace: "nowrap",
        letterSpacing: "0.02em",
        lineHeight: 1.2,
      }}
    >
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
    background: CSS_CARD,
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
          background: CSS_CARD,
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
function PitcherStarterCard({ spName, stats, theme, spStatus }) {
  const name = spName || "SP TBD";
  const gs = stats?.gs != null && stats.gs !== "—" ? String(stats.gs) : null;
  const th = theme || { primary: COL.model, soft: COL.cardInner, stroke: COL.border, onPrimary: "#FFFFFF" };
  const hasNamedStarter = !!(spName && spName !== "SP TBD");
  const isProbable = spStatus === "probable";

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
      background: CSS_CARD,
      border: `1px solid ${COL.border}`,
      boxShadow: "0 2px 6px rgba(15, 23, 42, 0.06)",
      minWidth: 0,
    }}
    >
      <div style={{
        padding: hdrPad,
        background: `linear-gradient(135deg, ${th.soft} 0%, ${CSS_CARD} 100%)`,
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
          {isProbable && (
            <div style={{
              fontSize: compact ? 8 : 9,
              fontWeight: 800,
              color: th.primary,
              letterSpacing: "0.06em",
              marginTop: 2,
            }}
            >Probable
            </div>
          )}
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
      ) : hasNamedStarter && isProbable ? (
        <div style={{ fontSize: 11, color: COL.textMuted, padding: compact ? "8px 12px" : "12px 14px" }}>
          Probable starter — stats unavailable
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

/** No browser Odds API — server morning pull only (see market_movement job). */
const PER_BOOK_ODDS_UNAVAILABLE_MSG = "Per-book sportsbook lines are temporarily unavailable.";

function useLiveMoneylines() {
  return null;
}

function useAllBookMoneylines() {
  return { books: null, quotaExhausted: false };
}

function useAllBookRunlines() {
  return { books: null, quotaExhausted: false };
}

function useAllBookTotals() {
  return { books: null, quotaExhausted: false };
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
  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState([]);

  useEffect(() => {
    if (!personId || !seasonYear) {
      setLoading(false);
      setRows([]);
      return;
    }
    let cancelled = false;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    setLoading(true);
    setRows([]);
    (async () => {
      try {
        const url = `${MLB_PEOPLE}/${personId}/stats?stats=gameLog&group=pitching&season=${seasonYear}&sportId=1`;
        const r = await fetch(url, { signal: ctrl.signal });
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
        if (!cancelled) {
          setRows(top);
          setLoading(false);
        }
      } catch {
        if (!cancelled) {
          setRows([]);
          setLoading(false);
        }
      } finally {
        clearTimeout(timer);
      }
    })();
    return () => {
      cancelled = true;
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [personId, seasonYear, n]);
  return { loading, rows };
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
            const tid = tr.team?.id;
            const packed = packStandingTeam(tr, divShort);
            const fullName = tid != null ? MLB_TEAM_ID_TO_FULL[tid] : null;
            if (fullName) out[fullName] = packed;
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

function useMlbGameFeed(gamePk, enabled, poll = true) {
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
    if (!poll) return () => { cancelled = true; };
    const id = setInterval(load, 30000);
    return () => { cancelled = true; clearInterval(id); };
  }, [gamePk, enabled, poll]);
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
    if (p.isBall) return COL.market;
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
      <rect x={0} y={0} width={W} height={H} rx={8} fill={PANEL_BG} stroke={PANEL_BORDER} />

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
      hr: gb.homeRuns,
      d: gb.doubles,
      t: gb.triples,
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

function parsePitchersSide(teamBox) {
  const pitchers = teamBox?.pitchers || [];
  const players = teamBox?.players || {};
  const rows = [];
  for (const id of pitchers) {
    const p = players[`ID${id}`];
    if (!p) continue;
    const gp = p.stats?.pitching || {};
    rows.push({
      id,
      name: p.person?.fullName || "",
      k: Number(gp.strikeOuts ?? 0),
      bb: Number(gp.baseOnBalls ?? 0),
      h: Number(gp.hits ?? 0),
      er: gp.earnedRuns != null ? Number(gp.earnedRuns) : Number(gp.runs ?? 0),
      ip: gp.inningsPitched ?? null,
    });
  }
  return rows;
}

function batterFeedToGameLine(row) {
  if (!row || row.isTotal) return null;
  return formatGameBattingLine({
    atBats: row.ab,
    hits: row.h,
    homeRuns: row.hr,
    doubles: row.d,
    triples: row.t,
    baseOnBalls: row.bb,
    strikeOuts: row.k,
    rbi: row.rbi,
    runs: row.r,
  });
}

function findFeedBatterGameLine(feedBatters, batterName) {
  if (!feedBatters?.length || !batterName) return null;
  const row = feedBatters.find((r) => !r.isTotal && lineupBatterMatches(r.name, batterName));
  return row ? batterFeedToGameLine(row) : null;
}

function findFeedPitcher(feedPitchers, pitcherName, pitcherId) {
  if (!feedPitchers?.length) return null;
  let row = null;
  if (pitcherId != null) {
    row = feedPitchers.find((p) => String(p.id) === String(pitcherId));
  }
  if (!row && pitcherName) {
    row = feedPitchers.find((p) => lineupBatterMatches(p.name, pitcherName));
  }
  return row || null;
}

function findFeedPitcherKs(feedPitchers, pitcherName, pitcherId) {
  const row = findFeedPitcher(feedPitchers, pitcherName, pitcherId);
  if (!row) return null;
  return Number.isFinite(row.k) ? row.k : null;
}

function findFeedPitcherActuals(feedPitchers, pitcherName, pitcherId) {
  const row = findFeedPitcher(feedPitchers, pitcherName, pitcherId);
  if (!row) return null;
  return {
    k: Number.isFinite(row.k) ? row.k : null,
    bb: Number.isFinite(row.bb) ? row.bb : null,
    hits: Number.isFinite(row.h) ? row.h : null,
    er: Number.isFinite(row.er) ? row.er : null,
    ip: parsePitcherIp(row.ip),
  };
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
    awayPitchers: parsePitchersSide(awayBox),
    homePitchers: parsePitchersSide(homeBox),
    pitching: parsePitchingDecisions(decisions, awayBox, homeBox),
  };
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

function useGameEnrichment(gameIds, seasonYear, refreshKey = 0, slateDate = null) {
  const [data, setData] = useState({});

  useEffect(() => {
    if (!gameIds.length) return;
    let cancelled = false;

    async function fetchProbablePitchersByGame(dateStr) {
      if (!dateStr) return {};
      const url = `${MLB_SCHEDULE}?sportId=1&date=${dateStr}&gameTypes=R&hydrate=probablePitcher`;
      try {
        const r = await fetch(url);
        if (!r.ok) return {};
        const j = await r.json();
        const out = {};
        for (const d of j.dates || []) {
          for (const g of d.games || []) {
            const pk = g.gamePk;
            if (!pk) continue;
            out[pk] = {
              away: g.teams?.away?.probablePitcher || null,
              home: g.teams?.home?.probablePitcher || null,
            };
          }
        }
        return out;
      } catch {
        return {};
      }
    }

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
      if (!t) return { lineup: [], spId: null, spName: null, spStatus: null };
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
      return { lineup, spId, spName, spStatus: spId ? "confirmed" : null };
    }

    function resolveStarter(parsedSide, probableSide) {
      if (parsedSide.spId) {
        return {
          ...parsedSide,
          spStatus: "confirmed",
        };
      }
      const prob = probableSide;
      if (prob?.id) {
        return {
          ...parsedSide,
          spId: prob.id,
          spName: prob.fullName || parsedSide.spName,
          spStatus: "probable",
        };
      }
      return { ...parsedSide, spStatus: null };
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
      const probableByPk = await fetchProbablePitchersByGame(slateDate);
      if (cancelled) return;
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
        const prob = probableByPk[pk];
        if (!box && !prob) continue;
        const awayRaw = box ? parseLineup("away", box) : { lineup: [], spId: null, spName: null, spStatus: null };
        const homeRaw = box ? parseLineup("home", box) : { lineup: [], spId: null, spName: null, spStatus: null };
        const away = resolveStarter(awayRaw, prob?.away);
        const home = resolveStarter(homeRaw, prob?.home);
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
          venueName: box ? venueNameFromBox(box) : null,
          umpireName: box ? homePlateUmpireFromBox(box) : null,
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
  }, [gameIds.join(","), seasonYear, refreshKey, slateDate]);

  return data;
}

/** Actual total equals betting line (integer lines; .5 lines rarely push). */
function isTotalsPush(totalRuns, line) {
  const t = Number(totalRuns);
  const L = Number(line);
  if (!Number.isFinite(t) || !Number.isFinite(L)) return false;
  return Math.abs(t - L) < 0.001;
}

/** Finished game: 'hit' | 'miss' | 'push' (market push: show dash, exclude from hit/miss) | null */
function gradeOuResult(ouRec, totalRuns, line) {
  if (ouRec == null || line == null || totalRuns == null) return null;
  const r = String(ouRec).toLowerCase();
  if (r === "push") return null;
  const t = Number(totalRuns);
  const L = Number(line);
  if (!Number.isFinite(t) || !Number.isFinite(L)) return null;
  const linePush = isTotalsPush(t, L);
  if (r === "over") {
    if (linePush) return "push";
    return t > L ? "hit" : "miss";
  }
  if (r === "under") {
    if (linePush) return "push";
    return t < L ? "hit" : "miss";
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
      background: CSS_CARD,
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

function PitcherLastStartsTable({ loading, rows }) {
  if (loading) {
    return <div style={{ fontSize: 12, color: COL.textMuted, marginTop: 8 }}>Loading last starts…</div>;
  }
  if (!rows?.length) {
    return <div style={{ fontSize: 12, color: COL.textMuted, marginTop: 8 }}>No recent starts available</div>;
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
  { id: "props", label: "Player Props" },
  { id: "last10", label: "Last 10" },
  { id: "weather", label: "Weather" },
  { id: "umpire", label: "Umpire" },
  { id: "moneyline", label: "Money Line" },
  { id: "runline", label: "Run Line" },
  { id: "totals", label: "O/U Totals" },
];

function GameDetailHeader({
  g,
  liveRow,
  venueLabel,
  gameLive,
  gameFinished,
  gamePostponed,
  hasLineupData,
  lastUpdatedAt,
  onBack,
}) {
  const abA = teamAbbr(g.away_team);
  const abH = teamAbbr(g.home_team);
  const arFinal = pickFinishedGameRuns(liveRow?.awayRuns, g.away_runs);
  const hrFinal = pickFinishedGameRuns(liveRow?.homeRuns, g.home_runs);
  const dateLong = formatGameDateLong(g.first_pitch_utc);
  const timeEt = formatFirstPitchET(g.first_pitch_utc);
  const metaLine = [dateLong, venueLabel].filter(Boolean).join(" · ");
  const liveSituation = formatLiveSituation(liveRow);

  let statusBadge = null;
  let statusDetail = null;
  if (gamePostponed) {
    statusBadge = (
      <span style={{
        display: "inline-flex",
        alignItems: "center",
        fontSize: 10,
        fontWeight: 900,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        padding: "4px 12px",
        borderRadius: 999,
        background: "#F59E0B",
        color: "#111827",
      }}
      >
        Postponed
      </span>
    );
  } else if (gameLive) {
    statusBadge = <LiveBadge />;
    statusDetail = liveSituation;
  } else if (gameFinished) {
    statusDetail = `${abA} ${arFinal ?? "—"} · ${abH} ${hrFinal ?? "—"} · Final`;
  } else {
    statusBadge = (
      <span style={{
        display: "inline-flex",
        alignItems: "center",
        fontSize: 10,
        fontWeight: 900,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        padding: "4px 12px",
        borderRadius: 999,
        background: "transparent",
        color: COL.model,
        border: `1.5px solid ${COL.model}`,
      }}
      >
        Upcoming
      </span>
    );
  }

  const rightSub = gameFinished
    ? null
    : !gameLive && !hasLineupData
      ? "Lineups pending"
      : lastUpdatedAt
        ? `Updated ${formatRelativeAgo(lastUpdatedAt)}`
        : null;

  return (
    <div style={{
      borderRadius: 14,
      border: `1px solid ${COL.border}`,
      background: COL.card,
      boxShadow: "0 4px 18px rgba(0,0,0,0.22)",
      overflow: "hidden",
      marginBottom: 20,
    }}
    >
      <div style={{ padding: "18px 20px 16px" }}>
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 16,
          flexWrap: "wrap",
        }}
        >
          <div style={{ display: "flex", alignItems: "flex-start", gap: 12, minWidth: 0, flex: "1 1 240px" }}>
            <button
              type="button"
              onClick={onBack}
              style={{
                marginTop: 4,
                fontSize: 20,
                fontWeight: 700,
                color: COL.textSecondary,
                background: "transparent",
                border: "none",
                padding: 0,
                cursor: "pointer",
                fontFamily: "inherit",
                lineHeight: 1,
                flexShrink: 0,
              }}
              aria-label="Back to schedule"
            >
              ←
            </button>
            <div style={{ minWidth: 0 }}>
              <h2 style={{
                fontSize: "clamp(22px, 4vw, 28px)",
                fontWeight: 800,
                color: COL.text,
                margin: 0,
                letterSpacing: "-0.02em",
                lineHeight: 1.15,
              }}
              >
                {formatMatchupAt(g.away_team, g.home_team)}
              </h2>
              {metaLine && (
                <div style={{ fontSize: 13, color: COL.textSecondary, fontWeight: 500, marginTop: 6, lineHeight: 1.4 }}>
                  {metaLine}
                </div>
              )}
            </div>
          </div>
          <div style={{ textAlign: "right", flexShrink: 0 }}>
            {timeEt && (
              <div style={{ fontSize: 15, fontWeight: 700, color: COL.text, letterSpacing: "-0.01em" }}>
                {timeEt}
              </div>
            )}
            {gameFinished ? (
              <div style={{ fontSize: 13, fontWeight: 800, color: COL.positive, marginTop: 4, fontVariantNumeric: "tabular-nums" }}>
                {`${abA} ${arFinal ?? "—"} · ${abH} ${hrFinal ?? "—"} · Final`}
              </div>
            ) : rightSub ? (
              <div style={{ fontSize: 12, color: COL.textMuted, marginTop: 4, fontWeight: 500 }}>
                {rightSub}
              </div>
            ) : null}
          </div>
        </div>
      </div>
      {(statusBadge || (statusDetail && !gameFinished)) && (
        <div style={{
          padding: "12px 20px",
          borderTop: `1px solid ${COL.border}`,
          background: gameLive ? "rgba(220,38,38,0.06)" : COL.cardInner,
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
        >
          {statusBadge}
          {statusDetail && !gameFinished && (
            <span style={{
              fontSize: 13,
              fontWeight: 600,
              color: COL.textSecondary,
            }}
            >
              {statusDetail}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

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
        background: `linear-gradient(135deg, ${themeAway.soft} 0%, ${CSS_CARD} 45%, ${CSS_CARD} 55%, ${themeHome.soft} 100%)`,
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
        background: `linear-gradient(135deg, ${themeAway.soft} 0%, ${CSS_CARD} 45%, ${CSS_CARD} 55%, ${themeHome.soft} 100%)`,
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
          background: `linear-gradient(135deg, ${th.soft} 0%, ${CSS_CARD} 100%)`,
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
                color: CSS_TEXT,
                letterSpacing: "0.06em",
              }}
              >Player</th>
              {cols.map((c) => (
                <th
                  key={c.key}
                  style={{
                    ...BATTER_H,
                    color: CSS_TEXT,
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
                        ? CSS_CARD_ALT
                        : CSS_CARD,
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

const TAB_ACCENT = COL.model;

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
        background: `linear-gradient(135deg, ${th.soft} 0%, ${CSS_CARD} 100%)`,
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
                  background: i % 2 === 1 ? CSS_CARD_ALT : CSS_CARD,
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
      background: `linear-gradient(${alignRight ? "225deg" : "135deg"}, ${theme.soft} 0%, ${CSS_CARD} 70%)`,
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
  const isMobile = useIsMobile();
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
      border: `1px solid ${PANEL_BORDER}`,
      background: PANEL_BG,
    }}
    >
      <div style={{
        display: "grid",
        gridTemplateColumns: isMobile ? "1fr" : "1fr minmax(200px, 240px) 1fr",
        minHeight: isMobile ? undefined : 220,
      }}
      >
        <div style={{
          padding: "18px 16px",
          borderRight: isMobile ? "none" : `1px solid ${PANEL_BORDER}`,
          borderBottom: isMobile ? `1px solid ${PANEL_BORDER}` : "none",
        }}
        >
          {isMobile ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4, minWidth: 0 }}>
              <Logo team={g.away_team} size={24} />
              <div style={{ fontSize: 14, fontWeight: 800, color: CSS_TEXT, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{g.away_team}</div>
            </div>
          ) : (
            <div style={{ fontSize: 14, fontWeight: 800, color: CSS_TEXT, marginBottom: 4 }}>{g.away_team}</div>
          )}
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.5)", marginBottom: 14, paddingLeft: isMobile ? 34 : 0 }}>{awayRecStr ? `(${awayRecStr})` : ""}</div>
          {statSide(sa, false)}
        </div>
        <div style={{ padding: "16px 12px", display: "flex", flexDirection: "column", justifyContent: "center", gap: 14, borderBottom: isMobile ? `1px solid ${PANEL_BORDER}` : "none" }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 11, fontWeight: 800 }}>
              <span style={{ color: "#DC2626" }}>{aPct != null ? `${aPct.toFixed(1)}%` : "—"}</span>
              <span style={{ color: "#22C55E" }}>{hPct != null ? `${hPct.toFixed(1)}%` : "—"}</span>
            </div>
            <div style={{ display: "flex", height: 12, borderRadius: 6, overflow: "hidden", background: PANEL_BORDER }}>
              <div style={{ width: `${aW}%`, background: "#DC2626" }} />
              <div style={{ width: `${hW}%`, background: "#22C55E" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, fontSize: 11, fontWeight: 700 }}>
              <span style={{ color: "#DC2626" }}>{awayShort} win</span>
              <span style={{ color: "#22C55E" }}>{homeShort} win</span>
            </div>
          </div>
          <div
            style={{
              background: PANEL_BG,
              border: `1px solid ${PANEL_BORDER}`,
              borderRadius: 8,
              padding: "10px 12px",
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}
          >
            <div style={{ flex: 1, textAlign: "center", color: CSS_TEXT, fontWeight: 800, fontSize: 15, fontVariantNumeric: "tabular-nums", fontFamily: FONT_MONO }}>{ar}</div>
            <div style={{ fontSize: 10, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.12em", textAlign: "center", flex: 1.2, whiteSpace: "nowrap" }}>PROJECTED RUNS</div>
            <div style={{ flex: 1, textAlign: "center", color: CSS_TEXT, fontWeight: 800, fontSize: 15, fontVariantNumeric: "tabular-nums", fontFamily: FONT_MONO }}>{hr}</div>
          </div>
        </div>
        <div style={{
          padding: "18px 16px",
          borderLeft: isMobile ? "none" : `1px solid ${PANEL_BORDER}`,
        }}
        >
          {isMobile ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4, minWidth: 0 }}>
              <Logo team={g.home_team} size={24} />
              <div style={{ fontSize: 14, fontWeight: 800, color: CSS_TEXT, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{g.home_team}</div>
            </div>
          ) : (
            <div style={{ fontSize: 14, fontWeight: 800, color: CSS_TEXT, marginBottom: 4, textAlign: "right" }}>{g.home_team}</div>
          )}
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.5)", marginBottom: 14, textAlign: isMobile ? "left" : "right", paddingLeft: isMobile ? 34 : 0 }}>{homeRecStr ? `(${homeRecStr})` : ""}</div>
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
  gamePostponed,
  lastUpdatedAt,
  propsBatters,
  propsPitchers,
  propsLoading,
}) {
  const isMobile = useIsMobile();
  const [activeNav, setActiveNav] = useState("pitching");
  const [expandedBatterId, setExpandedBatterId] = useState(null);
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
  const clientOddsDisabled = true;
  const oddsKeyMissing = true;
  const safeBookRows = Array.isArray(bookRows) ? bookRows : [];
  const safeRunlineRows = Array.isArray(runlineRows) ? runlineRows : [];
  const safeTotalsRows = Array.isArray(totalsRows) ? totalsRows : [];
  const abA = teamAbbr(g.away_team);
  const abH = teamAbbr(g.home_team);
  const awayRecStr = liveRow?.awayRecord ?? null;
  const homeRecStr = liveRow?.homeRecord ?? null;
  const awayDiv = standings?.away?.divisionLabel ?? null;
  const homeDiv = standings?.home?.divisionLabel ?? null;

  const { home: homeBatters, away: awayBatters, all: gameBatters } = useMemo(
    () => prepareGameBatterLineups(propsBatters, g.game_id),
    [propsBatters, g.game_id],
  );
  const gamePitchers = useMemo(
    () => (propsPitchers || []).filter((p) => String(p.game_id) === String(g.game_id)),
    [propsPitchers, g.game_id],
  );
  const homePitcher = resolveGamePitcher(gamePitchers, g, "home");
  const awayPitcher = resolveGamePitcher(gamePitchers, g, "away");
  const showPitcherLiveStats = gameLive || gameFinished;
  const homePitcherLive = showPitcherLiveStats
    ? findFeedPitcherActuals(feed?.homePitchers, homePitcher?.pitcher_name, homePitcher?.pitcher_id)
    : null;
  const awayPitcherLive = showPitcherLiveStats
    ? findFeedPitcherActuals(feed?.awayPitchers, awayPitcher?.pitcher_name, awayPitcher?.pitcher_id)
    : null;

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

  const sectionStyle = { scrollMarginTop: isMobile ? MOBILE_HEADER_H + 56 : 72, marginBottom: 32 };
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
    <div style={{ maxWidth: 1120, margin: "0 auto", padding: isMobile ? "0 0 48px" : "0 12px 48px" }}>
      <GameDetailHeader
        g={g}
        liveRow={liveRow}
        venueLabel={venueLabel}
        gameLive={gameLive}
        gameFinished={gameFinished}
        gamePostponed={gamePostponed}
        hasLineupData={hasLineupData}
        lastUpdatedAt={lastUpdatedAt}
        onBack={onBack}
      />

      <div style={{ display: "flex", gap: 0, alignItems: "stretch", flexDirection: isMobile ? "column" : "row", flexWrap: isMobile ? "nowrap" : "wrap" }}>
        <nav
          aria-label="Game sections"
          style={isMobile ? {
            position: "sticky",
            top: MOBILE_HEADER_H,
            zIndex: 50,
            display: "flex",
            gap: 8,
            overflowX: "auto",
            padding: "10px 0",
            background: COL.bg,
            borderBottom: `1px solid ${COL.border}`,
            marginBottom: 12,
          } : {
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
                style={isMobile ? {
                  flexShrink: 0,
                  whiteSpace: "nowrap",
                  padding: "8px 14px",
                  borderRadius: 999,
                  border: `1px solid ${active ? COL.model : COL.border}`,
                  background: active ? "rgba(234,88,0,0.12)" : "transparent",
                  color: active ? COL.text : COL.textSecondary,
                  fontWeight: active ? 800 : 600,
                  fontSize: 13,
                  cursor: "pointer",
                  fontFamily: "inherit",
                } : {
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
                {!isMobile && (
                  <span style={{
                    width: 6,
                    height: 6,
                    borderRadius: 2,
                    background: active ? TAB_ACCENT : "transparent",
                    flexShrink: 0,
                  }}
                  />
                )}
                {item.label}
              </button>
            );
          })}
        </nav>
        <div style={{ flex: 1, minWidth: isMobile ? 0 : 280, paddingBottom: 48 }}>
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
                { side: "away", team: g.away_team, theme: themeAway, sp: e?.away?.spName || g.away_sp_name, stats: e?.away?.stats, spStatus: e?.away?.spStatus, starts: awayStarts },
                { side: "home", team: g.home_team, theme: themeHome, sp: e?.home?.spName || g.home_sp_name, stats: e?.home?.stats, spStatus: e?.home?.spStatus, starts: homeStarts },
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
                    background: `linear-gradient(135deg, ${col.theme.soft} 0%, ${CSS_CARD} 100%)`,
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
                    <PitcherStarterCard spName={col.sp} stats={col.stats} theme={col.theme} spStatus={col.spStatus} />
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
                    <PitcherLastStartsTable loading={col.starts.loading} rows={col.starts.rows} />
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
                  const BATTER_H_PREGAME = { padding: "7px 10px", fontWeight: 800, fontSize: 10.5, letterSpacing: "0.06em", color: CSS_TEXT, textTransform: "uppercase", whiteSpace: "nowrap", background: th.soft };
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
                      <div style={{ padding: "12px 14px", background: `linear-gradient(135deg, ${th.soft} 0%, ${CSS_CARD} 100%)`, display: "flex", alignItems: "center", gap: 10, borderBottom: `1px solid ${COL.border}` }}>
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
                            <tr key={i} style={{ background: i % 2 === 0 ? CSS_CARD : CSS_CARD_ALT, borderBottom: `1px solid ${COL.border}` }}>
                              <td style={{ padding: "7px 10px", color: COL.textMuted, fontVariantNumeric: "tabular-nums", fontSize: 11, fontWeight: 700 }}>
                                {row.order != null ? row.order : ""}
                              </td>
                              <td style={{ padding: "7px 10px", color: CSS_TEXT, fontWeight: row.isSub ? 500 : 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 180 }}>
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

          <section id="detail-section-props" style={sectionStyle}>
            {sectionTitle("PLAYER PROPS")}
            {propsLoading ? (
              placeholder("Loading player prop predictions…")
            ) : !gameBatters.length && !gamePitchers.length ? (
              placeholder(
                hasLineupData
                  ? "Prop predictions for this game are not in the database yet. Run morning inference after lineups are confirmed."
                  : "Player props appear once lineups are confirmed and morning inference has run.",
              )
            ) : (
              <>
                <div style={{ fontSize: 11, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginBottom: 10 }}>STARTING PITCHERS</div>
                <div style={{ display: "flex", flexDirection: isMobile ? "column" : "row", gap: 16, flexWrap: isMobile ? "nowrap" : "wrap", marginBottom: 24 }}>
                  <div style={{ flex: isMobile ? "none" : 1, minWidth: 0, width: isMobile ? "100%" : undefined }}>
                    <PitcherPropCard pitcher={homePitcher} teamName={g.home_team} theme={themeHome} oppSpName={awayPitcher?.pitcher_name} liveActuals={homePitcherLive} />
                  </div>
                  <div style={{ flex: isMobile ? "none" : 1, minWidth: 0, width: isMobile ? "100%" : undefined }}>
                    <PitcherPropCard pitcher={awayPitcher} teamName={g.away_team} theme={themeAway} oppSpName={homePitcher?.pitcher_name} liveActuals={awayPitcherLive} />
                  </div>
                </div>
                <div style={{ fontSize: 11, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginBottom: 10 }}>BATTING LINEUPS</div>
                <BatterPropLineupPair
                  homeTeam={g.home_team}
                  awayTeam={g.away_team}
                  homeBatters={homeBatters}
                  awayBatters={awayBatters}
                  homePitcherName={awayPitcher?.pitcher_name}
                  awayPitcherName={homePitcher?.pitcher_name}
                  themeHome={themeHome}
                  themeAway={themeAway}
                  expandedBatterId={expandedBatterId}
                  onToggleBatter={(id) => setExpandedBatterId((cur) => (cur === id ? null : id))}
                  findGameLine={findFeedBatterGameLine}
                  feedHomeBatters={feed?.homeBatters}
                  feedAwayBatters={feed?.awayBatters}
                  showBoxScore={gameLive || gameFinished}
                  gameFinished={gameFinished}
                  renderExpanded={(batter, teamTitle) => (
                    <ExpandedBatterCard batter={batter} teamTitle={teamTitle} />
                  )}
                />
              </>
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
              <div style={{ padding: "16px 18px", background: CSS_CARD }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: COL.text, letterSpacing: "-0.02em" }}>{e?.umpireName ?? "—"}</div>
              </div>
            </div>
          </section>

          <section id="detail-section-moneyline" style={sectionStyle}>
            {sectionTitle("MONEY LINE")}
            <SteamMoveCallout g={g} themeAway={themeAway} themeHome={themeHome} />
            {!oddsKeyMissing && safeBookRows.length > 0 && (
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
                  <span style={{ fontSize: 10, fontWeight: 800, color: CSS_TEXT, letterSpacing: "0.1em", textTransform: "uppercase" }}>Per-book lines</span>
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
                    {safeBookRows.map((b, i) => (
                      <tr
                        key={b.key}
                        style={{
                          borderTop: `1px solid ${COL.border}`,
                          background: i % 2 === 1 ? CSS_CARD_ALT : CSS_CARD,
                        }}
                      >
                        <td style={{ padding: "11px 14px", fontWeight: 700, color: COL.text }}>{b.title}</td>
                        <td style={{
                          padding: "11px 14px",
                          textAlign: "right",
                          fontVariantNumeric: "tabular-nums",
                          fontWeight: 800,
                          color: CSS_TEXT,
                          background: themeAway.soft,
                        }}
                        >{b.away != null ? fmt(b.away) : "—"}</td>
                        <td style={{
                          padding: "11px 14px",
                          textAlign: "right",
                          fontVariantNumeric: "tabular-nums",
                          fontWeight: 800,
                          color: CSS_TEXT,
                          background: themeHome.soft,
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
                {clientOddsDisabled
                  ? PER_BOOK_ODDS_UNAVAILABLE_MSG
                  : oddsQuotaExhausted
                  ? "Odds API monthly quota exhausted — per-book lines unavailable until the quota resets. Upgrade at the-odds-api.com if needed."
                  : <>Set <code style={{ fontSize: 12 }}>VITE_ODDS_API_KEY</code> to load sportsbook moneylines.</>
                }
              </div>
            )}
            {!oddsKeyMissing && !oddsQuotaExhausted && safeBookRows.length === 0 && !gameLive && !gameFinished && (
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
                    <span style={{ fontSize: 10, fontWeight: 800, color: CSS_TEXT, letterSpacing: "0.1em", textTransform: "uppercase" }}>Closing consensus lines</span>
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
                        <tr style={{ borderTop: `1px solid ${COL.border}`, background: CSS_CARD }}>
                          <td style={{ padding: "11px 14px", fontWeight: 700, color: COL.text }}>Opening</td>
                          <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: CSS_TEXT, background: themeAway.soft }}>{fmt(g.morning_away_price)}</td>
                          <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: CSS_TEXT, background: themeHome.soft }}>{fmt(g.morning_home_price)}</td>
                        </tr>
                      )}
                      {g.closing_away_price != null && (
                        <tr style={{ borderTop: `1px solid ${COL.border}`, background: "rgba(15,23,42,0.015)" }}>
                          <td style={{ padding: "11px 14px", fontWeight: 700, color: COL.text }}>Closing</td>
                          <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: CSS_TEXT, background: themeAway.soft }}>{fmt(g.closing_away_price)}</td>
                          <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: CSS_TEXT, background: themeHome.soft }}>{fmt(g.closing_home_price)}</td>
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
            {!gameLive && !gameFinished && safeRunlineRows.length > 0 && (() => {
              // Compute consensus line (average of first-book point) for move display
              const firstPoint = safeRunlineRows[0]?.awayPoint;
              return (
                <div style={{ border: `1px solid ${COL.border}`, borderRadius: 12, overflow: "hidden", background: COL.card, boxShadow: `0 4px 14px rgba(15,23,42,0.07)` }}>
                  <div style={{ height: 4, background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)` }} />
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: "#1a202c" }}>
                    <span style={{ fontSize: 10, fontWeight: 800, color: CSS_TEXT, letterSpacing: "0.1em", textTransform: "uppercase" }}>Per-book run lines</span>
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
                      {safeRunlineRows.map((b, i) => {
                        const awayLineStr = b.awayPoint != null ? (b.awayPoint > 0 ? `+${b.awayPoint}` : String(b.awayPoint)) : "—";
                        const homeLineStr = b.homePoint != null ? (b.homePoint > 0 ? `+${b.homePoint}` : String(b.homePoint)) : "—";
                        return (
                          <tr key={b.key} style={{ borderTop: `1px solid ${COL.border}`, background: i % 2 === 1 ? CSS_CARD_ALT : CSS_CARD }}>
                            <td style={{ padding: "11px 14px", fontWeight: 700, color: COL.text }}>{b.title}</td>
                            <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: CSS_TEXT, background: themeAway.soft }}>{awayLineStr}</td>
                            <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 700, color: CSS_TEXT, background: themeAway.soft }}>{b.awayPrice != null ? fmt(b.awayPrice) : "—"}</td>
                            <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 800, color: CSS_TEXT, background: themeHome.soft }}>{homeLineStr}</td>
                            <td style={{ padding: "11px 14px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 700, color: CSS_TEXT, background: themeHome.soft }}>{b.homePrice != null ? fmt(b.homePrice) : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              );
            })()}
            {clientOddsDisabled && !gameLive && !gameFinished && (
              <div style={{ marginTop: 4, border: `1px solid ${COL.border}`, borderRadius: 12, background: COL.card, padding: "14px 16px", fontSize: 13, color: COL.textMuted }}>
                {PER_BOOK_ODDS_UNAVAILABLE_MSG}
              </div>
            )}
            {!clientOddsDisabled && !gameLive && !gameFinished && !rlQuotaExhausted && safeRunlineRows.length === 0 && (
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
                    <span style={{ fontSize: 10, fontWeight: 800, color: CSS_TEXT, letterSpacing: "0.1em", textTransform: "uppercase" }}>Run line not tracked live</span>
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
            {!gameLive && !gameFinished && safeTotalsRows.length > 0 && (
              <div style={{ border: `1px solid ${COL.border}`, borderRadius: 12, overflow: "hidden", background: COL.card, boxShadow: `0 4px 14px rgba(15,23,42,0.07)` }}>
                <div style={{ height: 4, background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)` }} />
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: "#1a202c" }}>
                  <span style={{ fontSize: 10, fontWeight: 800, color: CSS_TEXT, letterSpacing: "0.1em", textTransform: "uppercase" }}>Per-book O/U totals</span>
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
                    {safeTotalsRows.map((b, i) => (
                      <tr key={b.key} style={{ borderTop: `1px solid ${COL.border}`, background: i % 2 === 1 ? CSS_CARD_ALT : CSS_CARD }}>
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
            {clientOddsDisabled && !gameLive && !gameFinished && (
              <div style={{ marginTop: 4, border: `1px solid ${COL.border}`, borderRadius: 12, background: COL.card, padding: "14px 16px", fontSize: 13, color: COL.textMuted }}>
                {PER_BOOK_ODDS_UNAVAILABLE_MSG}
              </div>
            )}
            {!clientOddsDisabled && !gameLive && !gameFinished && !totalsQuotaExhausted && safeTotalsRows.length === 0 && (
              <div style={{ marginTop: 4, border: `1px solid ${COL.border}`, borderRadius: 12, background: COL.card, padding: "14px 16px", fontSize: 13, color: COL.textMuted }}>
                No O/U totals data available yet for this matchup.
              </div>
            )}
            {(gameLive || gameFinished) && (
              <div style={{ border: `1px solid ${COL.border}`, borderRadius: 12, overflow: "hidden", background: COL.card, boxShadow: `0 4px 14px rgba(15,23,42,0.07)` }}>
                <div style={{ height: 4, background: `linear-gradient(90deg, ${themeAway.primary} 0%, ${themeAway.primary} 50%, ${themeHome.primary} 50%, ${themeHome.primary} 100%)` }} />
                <div style={{ padding: "10px 14px", background: "#1a202c" }}>
                  <span style={{ fontSize: 10, fontWeight: 800, color: CSS_TEXT, letterSpacing: "0.1em", textTransform: "uppercase" }}>O/U totals not tracked live</span>
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
        background: CSS_CARD,
        boxShadow: `0 4px 18px rgba(15,23,42,0.08), 0 0 0 1px ${themeAway?.stroke || COL.border}`,
      }}
    >
      <div
        style={{
          background: "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)",
          padding: "18px 20px",
          color: CSS_TEXT,
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
            <div style={{ fontSize: 14, fontWeight: 800, color: CSS_TEXT, marginTop: 6 }}>
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
                    background: CSS_CARD,
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
                      background: CSS_CARD,
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
                    background: rowIdx % 2 === 0 ? CSS_CARD : CSS_CARD_ALT,
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

  const inningCell = (n, isAway) => {
    const raw = isAway ? awayInningRuns(n) : homeInningRuns(n);
    if (currentCol == null) {
      return { text: raw != null ? raw : "·", color: raw != null ? COL.text : COL.textMuted };
    }
    if (n > currentCol) {
      return { text: "·", color: COL.textMuted };
    }
    if (n < currentCol) {
      return { text: raw != null ? raw : 0, color: COL.text };
    }
    // current inning
    if (isAway) {
      if (isTop) return { text: "—", color: COL.textSecondary };
      return { text: raw != null ? raw : 0, color: COL.text };
    }
    if (isTop) return { text: "·", color: COL.textMuted };
    return { text: "—", color: COL.textSecondary };
  };

  const cellStyle = {
    padding: "8px 4px",
    textAlign: "center",
    fontSize: 14,
    fontWeight: 700,
    fontVariantNumeric: "tabular-nums",
    fontFamily: FONT_MONO,
  };
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
        background: PANEL_BG,
        border: `1px solid ${PANEL_BORDER}`,
        borderRadius: 12,
        overflow: "hidden",
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
                    color: currentCol === n ? "#DC2626" : COL.textMuted,
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
                const { text, color } = inningCell(n, true);
                return (
                  <td key={n} style={{ ...cellStyle, color }}>
                    {text}
                  </td>
                );
              })}
              {rheCell(rhe?.away?.r, true)}
              {rheCell(rhe?.away?.h)}
              {rheCell(rhe?.away?.e)}
            </tr>
            <tr style={{ borderTop: `1px solid ${PANEL_BORDER}` }}>
              <td style={{ padding: "6px 10px 6px 0", whiteSpace: "nowrap" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Logo team={homeTeamName} size={20} />
                  <span style={{ fontSize: 13, fontWeight: 800, color: themeHome?.primary || COL.text, letterSpacing: 0.2 }}>
                    {homeAbbr}
                  </span>
                </div>
              </td>
              {innNums.map((n) => {
                const { text, color } = inningCell(n, false);
                return (
                  <td key={n} style={{ ...cellStyle, color }}>
                    {text}
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
  const isMobile = useIsMobile();
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

  const scoreBlock = (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 12, minWidth: 0 }}>
      <Logo team={g.away_team} size={isMobile ? 28 : 36} />
      <span
        style={{
          fontSize: isMobile ? 28 : 34,
          fontWeight: 900,
          color: COL.text,
          fontVariantNumeric: "tabular-nums",
          fontFamily: FONT_MONO,
          letterSpacing: "-0.02em",
          lineHeight: 1,
        }}
      >{awayRunsLive}</span>
      <span style={{ color: COL.textMuted, fontWeight: 700, fontSize: isMobile ? 18 : 22 }}>—</span>
      <span
        style={{
          fontSize: isMobile ? 28 : 34,
          fontWeight: 900,
          color: COL.text,
          fontVariantNumeric: "tabular-nums",
          fontFamily: FONT_MONO,
          letterSpacing: "-0.02em",
          lineHeight: 1,
        }}
      >{homeRunsLive}</span>
      <Logo team={g.home_team} size={isMobile ? 28 : 36} />
    </div>
  );

  return (
    <div
      style={{
        borderRadius: 14,
        overflow: "hidden",
        border: `1px solid ${PANEL_BORDER}`,
        background: PANEL_BG,
        marginBottom: 16,
      }}
    >
      {isMobile ? (
        <div style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: 10, borderBottom: `1px solid ${PANEL_BORDER}` }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
            <LiveBadge />
            {inningLabel && <span style={{ fontWeight: 800, color: COL.text, fontSize: 14 }}>{inningLabel}</span>}
          </div>
          {scoreBlock}
          {venueLabel && (
            <span style={{ textAlign: "center", color: COL.textSecondary, fontWeight: 600, fontSize: 11 }}>{venueLabel}</span>
          )}
        </div>
      ) : (
        <div
          style={{
            padding: "14px 18px",
            display: "grid",
            gridTemplateColumns: "auto 1fr auto",
            alignItems: "center",
            gap: 12,
            borderBottom: `1px solid ${PANEL_BORDER}`,
          }}
        >
          <LiveBadge />
          {scoreBlock}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", flexShrink: 0, fontSize: 11, lineHeight: 1.3, minWidth: 100 }}>
            {inningLabel && (
              <span style={{ fontWeight: 800, color: COL.text, fontSize: 14 }}>{inningLabel}</span>
            )}
            {venueLabel && (
              <span style={{ color: COL.textSecondary, fontWeight: 600, marginTop: 2 }}>
                {venueLabel}
              </span>
            )}
          </div>
        </div>
      )}

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
              <div
                style={{
                  background: PANEL_BG,
                  border: `1px solid ${PANEL_BORDER}`,
                  borderRadius: 10,
                  padding: 4,
                  flexShrink: 0,
                }}
              >
                <StrikeZone
                  pitches={atBat?.pitches || []}
                  zoneTop={atBat?.pitches?.find((p) => p.zoneTop != null)?.zoneTop}
                  zoneBottom={atBat?.pitches?.find((p) => p.zoneBottom != null)?.zoneBottom}
                  batSide={atBat?.batSide || null}
                />
              </div>
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
                    background: PANEL_BG,
                    border: `1px solid ${PANEL_BORDER}`,
                    borderRadius: 10,
                    padding: "6px 12px",
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                  }}
                >
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <span style={{ fontSize: 9, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em" }}>COUNT</span>
                    <span style={{ fontSize: 18, fontWeight: 900, color: COL.text, fontVariantNumeric: "tabular-nums", fontFamily: FONT_MONO, lineHeight: 1.1 }}>
                      {formatPitchCount(liveRow.balls, liveRow.strikes)}
                    </span>
                  </div>
                  <div style={{ width: 1, alignSelf: "stretch", background: PANEL_BORDER }} />
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <span style={{ fontSize: 9, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em" }}>OUTS</span>
                    <span style={{ fontSize: 18, fontWeight: 900, color: COL.text, fontVariantNumeric: "tabular-nums", fontFamily: FONT_MONO, lineHeight: 1.1 }}>
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
          background: COL.card,
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
  color: "#4B5563",
  padding: "10px 10px",
  background: CSS_CARD_ALT,
  whiteSpace: "nowrap",
};
const GAMES_TABLE_NUM_HEADER = { ...GAMES_TABLE_HEADER_STYLE, textAlign: "right" };
const GAMES_NUM_CELL = { textAlign: "right" };

const GAMES_TABLE_COLS = [
  { key: "time", width: 108, label: "Time", style: GAMES_TABLE_HEADER_STYLE },
  { key: "score", width: 52, label: "Score", style: GAMES_TABLE_NUM_HEADER },
  { key: "teams", flex: true, minWidth: 220, label: "Teams", style: GAMES_TABLE_HEADER_STYLE },
  { key: "pitchers", width: 148, label: "Pitchers", style: GAMES_TABLE_HEADER_STYLE },
  { key: "model", width: 66, label: "Model", style: GAMES_TABLE_NUM_HEADER },
  { key: "market", width: 66, label: "Market", style: GAMES_TABLE_NUM_HEADER },
  { key: "odds", width: 60, label: "Odds", style: GAMES_TABLE_NUM_HEADER },
  { key: "edge", width: 60, label: "Edge", style: GAMES_TABLE_NUM_HEADER },
  { key: "runs", width: 54, label: "Runs", style: GAMES_TABLE_NUM_HEADER },
  { key: "total", width: 54, label: "Total", style: GAMES_TABLE_NUM_HEADER },
  { key: "ouline", width: 60, label: "O/U Line", style: GAMES_TABLE_NUM_HEADER },
  { key: "rec", width: 108, label: "Rec", style: GAMES_TABLE_NUM_HEADER },
];
const GAMES_TABLE_MIN_WIDTH =
  GAMES_TABLE_COLS.filter((c) => !c.flex).reduce((sum, c) => sum + (c.width || 0), 0)
  + (GAMES_TABLE_COLS.find((c) => c.flex)?.minWidth || 0);
const GAMES_REC_CELL_PAD = { paddingLeft: 12, paddingRight: 24 };

const GAMES_SECTION_TITLE_STYLE = {
  padding: "14px 16px",
  fontSize: 13,
  fontWeight: 900,
  letterSpacing: "0.14em",
  textTransform: "uppercase",
  color: "#E5E7EB",
  background: "#0D1420",
  borderBottom: `1px solid ${COL.border}`,
  borderTop: `1px solid ${COL.border}`,
};

const GAMES_SECTION_ACCENT = {
  live: { color: "#FCA5A5", borderLeft: "4px solid #EF4444" },
  upcoming: { color: COL.model, borderLeft: `4px solid ${COL.model}` },
  completed: { color: "#9CA3AF", borderLeft: "4px solid #6B7280" },
};

const GAMES_MONO = { fontFamily: FONT_MONO, fontVariantNumeric: "tabular-nums" };

/** API SP name, else MLB schedule/boxscore enrichment from useGameEnrichment. */
function resolveSideStarter(g, enrich, side) {
  const sideEnrich = side === "away" ? enrich?.[g.game_id]?.away : enrich?.[g.game_id]?.home;
  const apiName = side === "away" ? g.away_sp_name : g.home_sp_name;
  return {
    name: sideEnrich?.spName || apiName || null,
    status: sideEnrich?.spStatus || null,
  };
}

function GamesTable({ sortedGames, live, enrich, onOpenDetail, standingsMap, scheduleDate }) {
  // Ensure the pulse keyframe is injected once.
  useEffect(() => {
    const id = "mlb-live-pulse-kf";
    if (typeof document === "undefined" || document.getElementById(id)) return;
    const s = document.createElement("style");
    s.id = id;
    s.textContent = `@keyframes mlbLivePulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }`;
    document.head.appendChild(s);
  }, []);

  const isMobile = useIsMobile();

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
      out.push({ type: "section", k: p.key, title: p.title, count: p.games.length });
      for (const g of p.games) {
        out.push({
          type: "game",
          g,
          sectionKey: p.key,
          isFirst: globalIdx === 0,
          isLast: globalIdx === n - 1,
          isFirstInSection: out.length === 0 || out[out.length - 1].type === "section",
        });
        globalIdx += 1;
      }
    }
    return out;
  }, [sortedGames, live]);

  return (
    <div
      style={
        isMobile
          ? {}
          : {
              background: COL.cardBg,
              border: `1px solid ${COL.border}`,
              borderRadius: 14,
              overflow: "hidden",
              boxShadow: "0 6px 20px rgba(15,23,42,0.06), 0 1px 3px rgba(15,23,42,0.04)",
            }
      }
    >
      {isMobile ? (
        <div>
          {tableRows.map((row, rowIdx) => {
            if (row.type === "section") {
              const isFirstSection = !tableRows.slice(0, rowIdx).some((r) => r.type === "section");
              const isUpcoming = row.k === "upcoming";
              const accent = GAMES_SECTION_ACCENT[row.k] || GAMES_SECTION_ACCENT.completed;
              return (
                <div
                  key={`sec-${row.k}`}
                  style={{
                    ...GAMES_SECTION_TITLE_STYLE,
                    ...accent,
                    borderRadius: 8,
                    margin: "4px 0 10px",
                    border: "none",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 12,
                  }}
                >
                  <span>{row.title}</span>
                  {isUpcoming && scheduleDate && (
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 600,
                        letterSpacing: 0,
                        textTransform: "none",
                        color: "#6B7280",
                        fontFamily: FONT_BODY,
                      }}
                    >
                      {row.count} game{row.count === 1 ? "" : "s"} · {formatSectionScheduleDate(scheduleDate)}
                    </span>
                  )}
                </div>
              );
            }
            return (
              <GamesTableRow
                key={row.g.game_id}
                g={row.g}
                live={live}
                enrich={enrich}
                onOpenDetail={onOpenDetail}
                standingsMap={standingsMap}
                isLast={row.isLast}
                isFirst={row.isFirst}
                isFirstInSection={row.isFirstInSection}
                isMobile
              />
            );
          })}
        </div>
      ) : (
      <div style={{ width: "100%", overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
        <table
          style={{
            width: "100%",
            minWidth: GAMES_TABLE_MIN_WIDTH,
            borderCollapse: "collapse",
            fontFamily: "inherit",
            tableLayout: "fixed",
          }}
        >
          <colgroup>
            {GAMES_TABLE_COLS.map((c) => (
              <col
                key={c.key}
                style={c.flex ? undefined : { width: c.width ? `${c.width}px` : undefined }}
              />
            ))}
          </colgroup>
          <tbody>
            {tableRows.map((row, rowIdx) => {
              if (row.type === "section") {
                const isFirstSection = !tableRows.slice(0, rowIdx).some((r) => r.type === "section");
                const isUpcoming = row.k === "upcoming";
                const accent = GAMES_SECTION_ACCENT[row.k] || GAMES_SECTION_ACCENT.completed;
                return (
                  <Fragment key={`sec-${row.k}`}>
                    <tr>
                      <td
                        colSpan={GAMES_TABLE_COLS.length}
                        style={{
                          ...GAMES_SECTION_TITLE_STYLE,
                          ...accent,
                          borderTop: isFirstSection ? "none" : GAMES_SECTION_TITLE_STYLE.borderTop,
                          paddingLeft: 12,
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                          <span>{row.title}</span>
                          {isUpcoming && scheduleDate && (
                            <span
                              style={{
                                fontSize: 12,
                                fontWeight: 600,
                                letterSpacing: 0,
                                textTransform: "none",
                                color: "#6B7280",
                                fontFamily: FONT_BODY,
                              }}
                            >
                              {row.count} game{row.count === 1 ? "" : "s"} · {formatSectionScheduleDate(scheduleDate)}
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                    <tr>
                      {GAMES_TABLE_COLS.map((c) => (
                        <th
                          key={c.key}
                          style={{
                            ...c.style,
                            borderBottom: `1px solid ${COL.border}`,
                            position: "sticky",
                            top: 0,
                            zIndex: 1,
                          }}
                        >
                          {c.label}
                        </th>
                      ))}
                    </tr>
                  </Fragment>
                );
              }
              return (
                <Fragment key={row.g.game_id}>
                  <GamesTableRow
                    g={row.g}
                    live={live}
                    enrich={enrich}
                    onOpenDetail={onOpenDetail}
                    standingsMap={standingsMap}
                    isLast={row.isLast}
                    isFirst={row.isFirst}
                    isFirstInSection={row.isFirstInSection}
                  />
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      )}
    </div>
  );
}

function GamesTableRow({ g, live, enrich, onOpenDetail, standingsMap, isLast, isFirst, isFirstInSection, isMobile = false }) {
  const [hover, setHover] = useState(false);

  const awayStarter = resolveSideStarter(g, enrich, "away");
  const homeStarter = resolveSideStarter(g, enrich, "home");

  const liveRow = live?.[g.game_id];
  const detailed = liveRow?.status ?? g.status ?? "";
  const abstract = liveRow?.abstractGameState ?? null;
  const coded = liveRow?.codedGameState ?? null;
  const gameFinished = isMlbGameFinished(detailed, abstract, coded);
  const gamePostponed = isPostponedOrCancelled(detailed, abstract);
  const gameLive = !gameFinished && !gamePostponed && (isLiveStatus(detailed) || isLiveStatus(g.status));

  const themeAway = getTeamTheme(g.away_team);
  const themeHome = getTeamTheme(g.home_team);

  const { home: morningH, away: morningA } = pickPregameMlPrices(g);
  const preMarket = pickPregameMarketPct(g);
  const homeML = morningH != null ? fmt(morningH)
    : (preMarket.home != null ? (toAmerican(preMarket.home / 100) ?? "—") : "—");
  const awayML = morningA != null ? fmt(morningA)
    : (preMarket.away != null ? (toAmerican(preMarket.away / 100) ?? "—") : "—");

  const mpDev = deviggedMarketPct(morningH, morningA);
  const marketPHome = preMarket.home ?? mpDev.home ?? null;
  const marketPAway = preMarket.away ?? mpDev.away ?? null;

  const modelPHome = winProbPercent(g.p_win_home);
  const modelPAway = winProbPercent(g.p_win_away);

  const edgeHome = (modelPHome != null && marketPHome != null) ? (modelPHome - marketPHome) : null;
  const edgeAway = (modelPAway != null && marketPAway != null) ? (modelPAway - marketPAway) : null;

  const awayRunsPred = g.away_runs_pred != null ? Number(g.away_runs_pred) : null;
  const homeRunsPred = g.home_runs_pred != null ? Number(g.home_runs_pred) : null;
  const totalPred = g.total_runs_pred != null ? Number(g.total_runs_pred) : null;

  const ouLine = g.morning_ou_line ?? null;
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
  const lineupsPending = g.lineup_pending === true || g.lineup_pending === "true" || g.lineup_pending === 1;
  const awayRec = standingsMap?.[g.away_team];
  const homeRec = standingsMap?.[g.home_team];

  const onRowClick = () => onOpenDetail?.(g.game_id);

  const finishedBg = "rgba(245,158,11,0.05)";
  const finishedHoverBg = "rgba(245,158,11,0.09)";
  const baseRowStyle = {
    cursor: "pointer",
    background: gameFinished
      ? (hover ? finishedHoverBg : finishedBg)
      : (hover ? "#141E2E" : CSS_CARD),
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
  const gameDividerBorder = `1px dashed ${COL.borderSoft || "#374151"}`;

  const teamRowCell = (isAway) => {
    const name = isAway ? g.away_team : g.home_team;
    const displayName = isMobile ? teamNickname(name) : name;
    const rec = isAway ? awayRec : homeRec;
    const liveRecStr = isAway ? liveRow?.awayRecord : liveRow?.homeRecord;
    const recStr = liveRecStr
      ?? (rec?.wins != null && rec?.losses != null ? `${rec.wins}-${rec.losses}` : null);
    const isWinner = gameFinished && awayRunsActual != null && homeRunsActual != null
      && (isAway ? awayRunsActual > homeRunsActual : homeRunsActual > awayRunsActual);
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "nowrap", whiteSpace: "nowrap" }}>
        <Logo team={name} size={22} />
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 5,
            flexWrap: "nowrap",
            whiteSpace: "nowrap",
          }}
        >
          <span
            style={{
              fontSize: 13.5,
              fontWeight: isWinner ? 800 : 700,
              color: CSS_TEXT,
              whiteSpace: "nowrap",
              flexShrink: 0,
            }}
            title={name}
          >
            {displayName}
          </span>
          {recStr && (
            <span
              style={{
                fontSize: 11,
                color: COL.textMuted,
                fontWeight: 600,
                ...GAMES_MONO,
                flexShrink: 0,
                whiteSpace: "nowrap",
              }}
            >
              ({recStr})
            </span>
          )}
        </div>
      </div>
    );
  };

  const scoreCell = (isAway) => {
    const score = isAway ? awayRunsLive : homeRunsLive;
    if (score == null || (!gameLive && !gameFinished)) {
      return <span style={{ color: COL.textMuted, fontWeight: 600, ...GAMES_MONO }}>—</span>;
    }
    const isWinner = gameFinished && awayRunsActual != null && homeRunsActual != null
      && (isAway ? awayRunsActual > homeRunsActual : homeRunsActual > awayRunsActual);
    return (
      <span
        style={{
          fontSize: 16,
          fontWeight: 800,
          color: isWinner ? COL.positive : (gameLive ? COL.negative : COL.textPrimary),
          ...GAMES_MONO,
        }}
      >
        {score}
      </span>
    );
  };

  const pitcherCell = (name, spStatus) => (
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
      title={
        name
          ? (spStatus === "probable" ? `${name} — MLB probable starter` : name)
          : "Starter TBD"
      }
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
        ...GAMES_MONO,
      }}
    >
      {v != null && Number.isFinite(v) ? `${v.toFixed(1)}%` : "—"}
    </span>
  );

  /** Per matchup: higher model win% = green, lower = red, tie = dark text. */
  const modelPctCell = (isAway) => {
    const v = isAway ? modelPAway : modelPHome;
    if (v == null || !Number.isFinite(v)) {
      return <span style={{ color: COL.textMuted, fontWeight: 800, fontSize: 13.5, ...GAMES_MONO }}>—</span>;
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
          ...GAMES_MONO,
        }}
      >
        {v.toFixed(1)}%
      </span>
    );
  };

  const edgeCell = (v) => {
    if (v == null || !Number.isFinite(v)) {
      return <span style={{ color: COL.textMuted, fontWeight: 700, ...GAMES_MONO }}>—</span>;
    }
    const positive = v >= 0;
    return (
      <span
        style={{
          fontSize: 12.5,
          fontWeight: 800,
          color: positive ? COL.positive : COL.negative,
          ...GAMES_MONO,
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
        ...GAMES_MONO,
      }}
    >
      {v}
    </span>
  );

  const runsCell = (v) => (
    <span style={{ fontSize: 12.5, fontWeight: 700, color: COL.textPrimary, ...GAMES_MONO }}>
      {v != null ? v.toFixed(2) : "—"}
    </span>
  );

  const timeCell = () => {
    const timeStackStyle = {
      display: "flex",
      flexDirection: "column",
      gap: 2,
      minHeight: GAMES_TIME_MIN_HEIGHT,
      justifyContent: "center",
    };
    if (gamePostponed) {
      return (
        <div style={{ ...timeStackStyle, alignItems: "flex-start" }}>
          <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: 0.6, color: COL.textMuted, textTransform: "uppercase" }}>Postponed</span>
        </div>
      );
    }
    if (gameFinished) {
      return (
        <div style={{ ...timeStackStyle, alignItems: "flex-start" }}>
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
        <div style={{ ...timeStackStyle, alignItems: "flex-start" }}>
          <LiveBadge size="sm" />
          {inningLabel && (
            <span style={{ fontSize: 9.5, fontWeight: 600, color: COL.textMuted, letterSpacing: 0.1 }}>
              {inningLabel}
            </span>
          )}
        </div>
      );
    }
    return (
      <div style={timeStackStyle}>
        {firstPitchParts?.et ? (
          <>
            <span style={{ fontSize: 12.5, fontWeight: 800, color: COL.textPrimary, letterSpacing: 0.2, whiteSpace: "nowrap", ...GAMES_MONO }}>
              {firstPitchParts.et} ET
            </span>
            {firstPitchParts.pt && (
              <span style={{ fontSize: 10.5, color: COL.textMuted, fontWeight: 600, whiteSpace: "nowrap", ...GAMES_MONO }}>
                {firstPitchParts.pt} PT
              </span>
            )}
            {lineupsPending && (
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  alignSelf: "flex-start",
                  fontSize: 7.5,
                  fontWeight: 800,
                  letterSpacing: 0.35,
                  lineHeight: 1.2,
                  color: "#92400E",
                  textTransform: "uppercase",
                  border: "1px solid rgba(245,158,11,0.3)",
                  padding: "1px 5px",
                  borderRadius: 999,
                  background: "rgba(245,158,11,0.1)",
                  marginTop: 4,
                  whiteSpace: "nowrap",
                }}
                title="Early pre-lineup prediction — updates when lineups are confirmed"
              >
                LINEUPS PENDING
              </span>
            )}
          </>
        ) : (
          <span style={{ fontSize: 12, color: COL.textMuted, fontWeight: 600, ...GAMES_MONO }}>—</span>
        )}
      </div>
    );
  };

  const totalCell = () => (
    <span style={{ fontSize: 13, fontWeight: 800, color: COL.model, ...GAMES_MONO }}>
      {totalPred != null ? totalPred.toFixed(2) : "—"}
    </span>
  );

  const ouLineCell = () => (
    <span style={{ fontSize: 12.5, fontWeight: 700, color: COL.textPrimary, ...GAMES_MONO }}>
      {ouLine != null ? Number(ouLine).toFixed(1) : "—"}
    </span>
  );

  const recCell = () => {
    const emptyRec = (
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: GAMES_REC_STACK_WIDTH,
          color: COL.textMuted,
          fontSize: 12.5,
          fontWeight: 700,
          ...GAMES_MONO,
        }}
      >
        —
      </span>
    );
    if (gamePostponed) return emptyRec;
    let pill = null;
    if (ouRec === "over") pill = <GamesOuRecPill color="green">Over</GamesOuRecPill>;
    else if (ouRec === "under") pill = <GamesOuRecPill color="red">Under</GamesOuRecPill>;
    else if (ouRec === "push") pill = <GamesOuRecPill color="gray">Pass</GamesOuRecPill>;
    else return emptyRec;
    const showGradeMark =
      gameFinished && ouResult != null && ouRec !== "push" && ouResult !== "push";
    if (gameFinished && ouResult === "push" && (ouRec === "over" || ouRec === "under")) {
      return (
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 5,
            width: GAMES_REC_STACK_WIDTH,
            flexWrap: "nowrap",
          }}
        >
          {pill}
          <span
            aria-label="Push"
            title="Push (final total equals market line)"
            style={{ fontSize: 13, fontWeight: 700, color: COL.textMuted, lineHeight: 1, flex: "0 0 auto" }}
          >
            —
          </span>
        </span>
      );
    }
    if (!showGradeMark) return pill;
    const isHit = ouResult === "hit";
    const markBg = isHit ? "rgba(34,197,94,0.18)" : "rgba(239,68,68,0.18)";
    const markColor = isHit ? COL.positive : COL.negative;
    const markBorder = isHit ? "rgba(22,163,74,0.45)" : "rgba(220,38,38,0.45)";
    return (
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 5,
          width: GAMES_REC_STACK_WIDTH,
          flexWrap: "nowrap",
        }}
      >
        {pill}
        <span
          aria-label={isHit ? "Hit" : "Miss"}
          title={isHit ? "Recommendation hit" : "Recommendation missed"}
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 18,
            height: 18,
            borderRadius: "50%",
            background: markBg,
            color: markColor,
            border: `1.5px solid ${markBorder}`,
            fontSize: 11,
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

  const recDetailsLink = (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); onOpenDetail?.(g.game_id); }}
      style={{
        width: GAMES_REC_STACK_WIDTH,
        fontSize: 11,
        fontWeight: 600,
        color: hover ? COL.model : "rgba(234, 88, 12, 0.55)",
        background: "none",
        border: "none",
        padding: 0,
        cursor: "pointer",
        letterSpacing: 0.15,
        transition: "color 0.12s ease",
        whiteSpace: "nowrap",
        textAlign: "center",
        fontFamily: "inherit",
      }}
    >
      Details →
    </button>
  );

  const gameDividerTop = isFirstInSection ? "none" : gameDividerBorder;
  const topCellShared = { ...cellBase, ...thickTop, ...noBottom, borderTop: gameDividerTop };
  const bottomCellShared = { ...cellBase, borderTop: "none" };

  // For rowspan cells, span both rows.
  const rowSpan2Style = {
    ...cellBase,
    ...thickTop,
    borderTop: gameDividerTop,
    borderBottom: isLast ? "none" : undefined,
    verticalAlign: "middle",
  };
  const rowSpanNumStyle = { ...rowSpan2Style, ...GAMES_NUM_CELL };

  const onHoverOn = () => setHover(true);
  const onHoverOff = () => setHover(false);

  if (isMobile) {
    const miniHdr = {
      fontSize: 9,
      fontWeight: 800,
      letterSpacing: 0.5,
      textTransform: "uppercase",
      color: COL.textMuted,
      textAlign: "right",
    };
    const miniAbbr = { fontSize: 11, fontWeight: 800, color: COL.textSecondary, ...GAMES_MONO };
    const cellRight = { textAlign: "right" };
    const awayAbbr = TEAM_ABBR[g.away_team] || g.away_team;
    const homeAbbr = TEAM_ABBR[g.home_team] || g.home_team;
    const rightNum = (isAway) => (gameLive || gameFinished ? scoreCell(isAway) : modelPctCell(isAway));
    const stateAccent = gameLive ? "#EF4444" : gameFinished ? "#374151" : COL.model;

    return (
      <div
        onClick={onRowClick}
        style={{
          cursor: onOpenDetail ? "pointer" : "default",
          padding: "14px 16px",
          marginBottom: 12,
          background: COL.cardBg || "#111827",
          border: `1px solid ${COL.border}`,
          borderLeft: `3px solid ${stateAccent}`,
          borderRadius: 12,
          opacity: gameFinished ? 0.88 : 1,
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
          <div>{timeCell()}</div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 9 }}>
            {recCell()}
            {recDetailsLink}
            {(totalPred != null || ouLine != null) && (
              <span style={{ fontSize: 10.5, color: COL.textMuted, fontWeight: 600, ...GAMES_MONO }}>
                {totalPred != null ? `Tot ${totalPred.toFixed(2)}` : ""}
                {totalPred != null && ouLine != null ? " · " : ""}
                {ouLine != null ? `Line ${Number(ouLine).toFixed(1)}` : ""}
              </span>
            )}
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {[true, false].map((isAway) => (
            <div
              key={isAway ? "away" : "home"}
              style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}
            >
              <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0, flex: 1 }}>
                {teamRowCell(isAway)}
                <span
                  style={{
                    fontSize: 11,
                    color: COL.textMuted,
                    paddingLeft: 30,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {(() => {
                    const st = isAway ? awayStarter : homeStarter;
                    return st.name || "TBD";
                  })()}
                </span>
              </div>
              <div style={{ minWidth: 34, textAlign: "right" }}>{rightNum(isAway)}</div>
            </div>
          ))}
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "auto 1fr 1fr 1fr",
            gap: "4px 10px",
            alignItems: "center",
            paddingTop: 10,
            borderTop: `1px solid ${COL.borderSoft || "#1f2a3a"}`,
          }}
        >
          <span />
          <span style={miniHdr}>Model</span>
          <span style={miniHdr}>Market</span>
          <span style={miniHdr}>Edge</span>
          <span style={miniAbbr}>{awayAbbr}</span>
          <span style={cellRight}>{modelPctCell(true)}</span>
          <span style={cellRight}>{pctCell(marketPAway)}</span>
          <span style={cellRight}>{edgeCell(edgeAway)}</span>
          <span style={miniAbbr}>{homeAbbr}</span>
          <span style={cellRight}>{modelPctCell(false)}</span>
          <span style={cellRight}>{pctCell(marketPHome)}</span>
          <span style={cellRight}>{edgeCell(edgeHome)}</span>
        </div>

        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onOpenDetail?.(g.game_id);
          }}
          style={{
            width: "100%",
            fontSize: 12,
            fontWeight: 600,
            color: "rgba(234, 88, 12, 0.65)",
            background: "transparent",
            border: "none",
            borderTop: `1px solid ${COL.borderSoft || "#1f2a3a"}`,
            padding: "10px 0 2px",
            cursor: "pointer",
            letterSpacing: 0.2,
          }}
        >
          Details →
        </button>
      </div>
    );
  }

  return (
    <>
      <tr
        style={baseRowStyle}
        onClick={onRowClick}
        onMouseEnter={onHoverOn}
        onMouseLeave={onHoverOff}
      >
        <td style={{ ...rowSpan2Style, ...finishedAccent, textAlign: "left", overflow: "visible" }} rowSpan={2}>
          {timeCell()}
        </td>
        <td style={{ ...topCellShared, ...GAMES_NUM_CELL }}>{scoreCell(true)}</td>
        <td style={topCellShared}>{teamRowCell(true)}</td>
        <td style={topCellShared}>{pitcherCell(awayStarter.name, awayStarter.status)}</td>
        <td style={{ ...topCellShared, ...GAMES_NUM_CELL }}>{modelPctCell(true)}</td>
        <td style={{ ...topCellShared, ...GAMES_NUM_CELL }}>{pctCell(marketPAway)}</td>
        <td style={{ ...topCellShared, ...GAMES_NUM_CELL }}>{mlCell(awayML)}</td>
        <td style={{ ...topCellShared, ...GAMES_NUM_CELL }}>{edgeCell(edgeAway)}</td>
        <td style={{ ...topCellShared, ...GAMES_NUM_CELL }}>{runsCell(awayRunsPred)}</td>
        <td style={{ ...rowSpanNumStyle }} rowSpan={2}>
          <div style={{ display: "flex", justifyContent: "flex-end", width: "100%" }}>
            {totalCell()}
          </div>
        </td>
        <td style={{ ...rowSpanNumStyle }} rowSpan={2}>
          <div style={{ display: "flex", justifyContent: "flex-end", width: "100%" }}>
            {ouLineCell()}
          </div>
        </td>
        <td style={{ ...rowSpan2Style, ...GAMES_REC_CELL_PAD, textAlign: "right", overflow: "visible" }} rowSpan={2}>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-end",
              gap: 9,
              width: "100%",
            }}
          >
            {recCell()}
            {recDetailsLink}
          </div>
        </td>
      </tr>
      <tr
        style={baseRowStyle}
        onClick={onRowClick}
        onMouseEnter={onHoverOn}
        onMouseLeave={onHoverOff}
      >
        <td style={{ ...bottomCellShared, ...GAMES_NUM_CELL }}>{scoreCell(false)}</td>
        <td style={bottomCellShared}>{teamRowCell(false)}</td>
        <td style={bottomCellShared}>{pitcherCell(homeStarter.name, homeStarter.status)}</td>
        <td style={{ ...bottomCellShared, ...GAMES_NUM_CELL }}>{modelPctCell(false)}</td>
        <td style={{ ...bottomCellShared, ...GAMES_NUM_CELL }}>{pctCell(marketPHome)}</td>
        <td style={{ ...bottomCellShared, ...GAMES_NUM_CELL }}>{mlCell(homeML)}</td>
        <td style={{ ...bottomCellShared, ...GAMES_NUM_CELL }}>{edgeCell(edgeHome)}</td>
        <td style={{ ...bottomCellShared, ...GAMES_NUM_CELL }}>{runsCell(homeRunsPred)}</td>
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

  const { home: morningH, away: morningA } = pickPregameMlPrices(g);
  const preMarket = pickPregameMarketPct(g);

  const homeML = morningH != null ? fmt(morningH)
    : (preMarket.home != null ? (toAmerican(preMarket.home / 100) ?? "—") : "—");
  const awayML = morningA != null ? fmt(morningA)
    : (preMarket.away != null ? (toAmerican(preMarket.away / 100) ?? "—") : "—");

  const mpDev = deviggedMarketPct(morningH, morningA);
  const marketPHome = preMarket.home ?? mpDev.home;
  const marketPAway = preMarket.away ?? mpDev.away;

  const closPh = g.closing_p_home != null ? Number(g.closing_p_home) : null;
  const deltaHomeProb = mornPh != null && closPh != null ? closPh - mornPh : null;
  const deltaAwayProb = deltaHomeProb != null ? -deltaHomeProb : null;

  const ouLineMorning = g.morning_ou_line;
  const ouDisplay = ouLineMorning;

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
  const ouLineForGrade = ouLineMorning ?? null;
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
                background: COL.card,
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
                      background: CSS_CARD,
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
                  background: `linear-gradient(90deg, ${th.primary} 0%, ${th.soft} 65%, ${CSS_CARD} 100%)`,
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
                    background: CSS_CARD,
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
                      background: CSS_CARD,
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
                    spStatus={r.spEnrich?.spStatus}
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
              background: CSS_CARD,
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
              ev.currentTarget.style.background = CSS_CARD;
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

function GameDetailRoute({ g, live, enrich, seasonYear, onBack, lastUpdatedAt, propsBatters, propsPitchers, propsLoading }) {
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
      gamePostponed={gamePostponed}
      lastUpdatedAt={lastUpdatedAt}
      propsBatters={propsBatters}
      propsPitchers={propsPitchers}
      propsLoading={propsLoading}
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

function useEdgesData(enabled, date, refreshKey) {
  const [data, setData] = useState(() => (
    edgesCache && edgesCacheDate === date ? edgesCache : null
  ));
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(() => !(edgesCache && edgesCacheDate === date));

  useEffect(() => {
    if (!enabled || !date) return;
    let cancelled = false;
    const cached = edgesCache && edgesCacheDate === date;
    if (cached && !refreshKey) {
      setData(edgesCache);
      setError(null);
      setLoading(false);
      return undefined;
    }
    setError(null);
    if (!cached) setLoading(true);
    fetchEdgesData(date, { force: Boolean(refreshKey) })
      .then((j) => { if (!cancelled) { setData(j); setLoading(false); } })
      .catch((e) => { if (!cancelled) { setError(String(e?.message || e)); setLoading(false); } });
    return () => { cancelled = true; };
  }, [enabled, date, refreshKey]);

  return { data, error, loading };
}

function useTrendsData(enabled, date, refreshKey) {
  const d = date || pacificTodayStr();
  const [data, setData] = useState(() => (
    trendsCache && trendsCacheDate === d ? trendsCache : null
  ));
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(() => !(trendsCache && trendsCacheDate === d));

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const cached = trendsCache && trendsCacheDate === d;
    if (cached && !refreshKey) {
      setData(trendsCache);
      setError(null);
      setLoading(false);
      return undefined;
    }
    setError(null);
    if (!cached) setLoading(true);
    fetchTrendsData({ force: Boolean(refreshKey), date: d })
      .then((j) => { if (!cancelled) { setData(j); setLoading(false); } })
      .catch((e) => { if (!cancelled) { setError(String(e?.message || e)); setLoading(false); } });
    return () => { cancelled = true; };
  }, [enabled, refreshKey, d]);

  return { data, error, loading };
}

function useStandingsData(enabled, date, refreshKey) {
  const d = date || pacificTodayStr();
  const [data, setData] = useState(() => (
    standingsCache && standingsCacheDate === d ? standingsCache : null
  ));
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(() => !(standingsCache && standingsCacheDate === d));

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const cached = standingsCache && standingsCacheDate === d;
    if (cached && !refreshKey) {
      setData(standingsCache);
      setError(null);
      setLoading(false);
      return undefined;
    }
    setError(null);
    if (!cached) setLoading(true);
    fetchStandingsData(d, { force: Boolean(refreshKey) })
      .then((j) => { if (!cancelled) { setData(j); setLoading(false); } })
      .catch((e) => { if (!cancelled) { setError(String(e?.message || e)); setLoading(false); } });
    return () => { cancelled = true; };
  }, [enabled, d, refreshKey]);

  return { data, error, loading };
}

function useTransactionsData(enabled, date, days, refreshKey) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const d = date || pacificTodayStr();
    setLoading(true);
    setError(null);
    fetchTransactionsData(d, days, { force: Boolean(refreshKey) })
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || "Failed to load transactions");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, date, days, refreshKey]);

  return { data, error, loading };
}

const EDGE_CARD_BG = "#0D1420";
const EDGE_CARD_BORDER = "#1F2937";

const EDGE_TYPE_STYLE = {
  ml: { bg: "rgba(59,130,246,0.15)", color: "#60A5FA", label: "ML" },
  total: { bg: "rgba(139,92,246,0.15)", color: "#A78BFA", label: "O/U" },
  k: { bg: "rgba(245,158,11,0.15)", color: COL.model, label: "K" },
  prop: { bg: "rgba(245,158,11,0.15)", color: COL.model, label: "PROP" },
};

function EdgeTypePill({ type, subtype }) {
  const s = EDGE_TYPE_STYLE[type] || EDGE_TYPE_STYLE.prop;
  const label = type === "prop" && subtype ? subtype : s.label;
  return (
    <span style={{
      fontSize: 10,
      fontWeight: 800,
      letterSpacing: "0.08em",
      padding: "3px 8px",
      borderRadius: 6,
      background: s.bg,
      color: s.color,
      flexShrink: 0,
    }}>
      {label}
    </span>
  );
}

function EdgeCardSkeleton() {
  return (
    <div
      className="trends-skeleton-row"
      style={{
        display: "grid",
        gridTemplateColumns: "36px auto 1fr auto",
        gap: 14,
        alignItems: "center",
        background: EDGE_CARD_BG,
        border: `1px solid ${EDGE_CARD_BORDER}`,
        borderRadius: 12,
        padding: "14px 16px",
      }}
    >
      <div style={{ height: 28, background: "#1F2937", borderRadius: 6 }} />
      <div style={{ width: 36, height: 22, background: "#1F2937", borderRadius: 6 }} />
      <div>
        <div style={{ height: 14, background: "#1F2937", borderRadius: 4, marginBottom: 8, width: "70%" }} />
        <div style={{ height: 12, background: "#1F2937", borderRadius: 4, width: "55%" }} />
      </div>
      <div style={{ width: 80, height: 32, background: "#1F2937", borderRadius: 6 }} />
    </div>
  );
}

function TopEdgesPanel({ enabled, date, refreshKey }) {
  const isMobile = useIsMobile();
  const { data, error, loading } = useEdgesData(enabled, date, refreshKey);
  const [filter, setFilter] = useState("all");

  const headerDateLabel = new Date(`${date}T12:00:00`).toLocaleDateString("en-US", {
    timeZone: "America/Los_Angeles",
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  const chips = [
    { id: "all", label: "All" },
    { id: "ml", label: "Moneyline" },
    { id: "total", label: "Totals" },
    { id: "k", label: "Pitcher K" },
    { id: "walks", label: "Walks" },
    { id: "hits", label: "Hits" },
    { id: "er", label: "ER" },
    { id: "prop", label: "Batter props" },
  ];

  if (loading && !data) {
    return (
      <div style={pageShellStyle(920, { top: 12, horizontal: isMobile ? -12 : 0 })}>
        <h1 style={{ fontFamily: FONT_DISPLAY, fontSize: 42, fontWeight: 400, color: COL.model, margin: "0 0 8px", letterSpacing: "0.04em", lineHeight: 1 }}>
          Top Edges
        </h1>
        <p style={{ margin: "0 0 24px", fontSize: 14, lineHeight: 1.6, color: "#9CA3AF", maxWidth: 560 }}>
          Where the model diverges most from the market line or league baselines. Not raw probabilities — actual disagreements worth a look.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <EdgeCardSkeleton key={i} />
          ))}
        </div>
      </div>
    );
  }
  if (error && !data) {
    return (
      <div style={{ maxWidth: 640, margin: "24px auto", padding: 16, background: COL.card, border: `1px solid ${COL.border}`, borderRadius: 12, color: COL.textSecondary, fontSize: 13 }}>
        Could not load edges: {error}
      </div>
    );
  }

  const edges = (data?.edges || []).filter((e) => filter === "all" || e.type === filter);
  const edgeSubtitle = (e) => {
    const isPlayerEdge = e.player_id != null && e.team_name;
    return isPlayerEdge ? `${e.team_name} · ${e.detail || ""}` : e.detail;
  };
  const edgeRateHover = (e) => {
    if (!e.rate_detail || !["k", "walks", "hits", "er"].includes(e.type)) return undefined;
    return e.rate_detail;
  };

  return (
    <div style={pageShellStyle(920, { top: 12, horizontal: isMobile ? -12 : 0 })}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, flexWrap: "wrap", marginBottom: 18 }}>
        <div>
          <h1 style={{ fontFamily: FONT_DISPLAY, fontSize: 42, fontWeight: 400, color: COL.model, margin: "0 0 8px", letterSpacing: "0.04em", lineHeight: 1 }}>
            Top Edges
          </h1>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "#9CA3AF", maxWidth: 560 }}>
            Where the model diverges most from the market line or league baselines. Not raw probabilities — actual disagreements worth a look.
          </p>
        </div>
        <div style={{ fontFamily: FONT_MONO, fontSize: 13, color: "#6B7280", flexShrink: 0, paddingTop: 8 }}>
          {headerDateLabel}
        </div>
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
        {chips.map((c) => {
          const active = filter === c.id;
          return (
            <button
              key={c.id}
              type="button"
              onClick={() => setFilter(c.id)}
              style={{
                fontFamily: "inherit",
                fontSize: 12,
                fontWeight: 700,
                padding: "6px 14px",
                borderRadius: 999,
                border: `1px solid ${active ? COL.model : COL.border}`,
                background: active ? "rgba(245,158,11,0.12)" : COL.card,
                color: active ? COL.model : "#9CA3AF",
                cursor: "pointer",
              }}
            >
              {c.label}
            </button>
          );
        })}
      </div>

      {edges.length === 0 ? (
        <div style={{ color: COL.textMuted, fontSize: 14, textAlign: "center", padding: "32px 0" }}>
          No edges above threshold for this filter today.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {edges.map((e) => {
            const edgeColor = e.direction === "under" ? "#60A5FA" : COL.positive;

            if (isMobile) {
              return (
                <div
                  key={`${e.type}-${e.rank}-${e.title}`}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 10,
                    background: EDGE_CARD_BG,
                    border: `1px solid ${EDGE_CARD_BORDER}`,
                    borderRadius: 12,
                    padding: "14px 16px",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontFamily: FONT_DISPLAY, fontSize: 22, color: "#4B5563", lineHeight: 1 }}>{e.rank}</span>
                    <EdgeTypePill type={e.type} subtype={e.subtype} />
                  </div>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                    {e.player_id != null && e.team_name ? <Logo team={e.team_name} size={24} /> : null}
                    <span style={{
                      fontSize: 16,
                      fontWeight: 800,
                      color: CSS_TEXT,
                      lineHeight: 1.3,
                      whiteSpace: "normal",
                      wordBreak: "break-word",
                    }}
                    >
                      {e.title}
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "#6B7280",
                      lineHeight: 1.35,
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                      overflow: "hidden",
                    }}
                    title={edgeRateHover(e)}
                  >
                    {edgeSubtitle(e)}
                  </div>
                  <div style={{ display: "flex", gap: 28, alignItems: "center" }}>
                    <div>
                      <div style={{ fontSize: 9, fontWeight: 800, color: "#6B7280", letterSpacing: "0.08em" }}>{e.stat_label}</div>
                      <div style={{ fontFamily: FONT_MONO, fontSize: 15, fontWeight: 700, color: CSS_TEXT }}>{e.stat_value}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, fontWeight: 800, color: "#6B7280", letterSpacing: "0.08em" }}>{e.edge_label}</div>
                      <div style={{ fontFamily: FONT_MONO, fontSize: 15, fontWeight: 800, color: edgeColor }}>{e.edge_value}</div>
                    </div>
                  </div>
                </div>
              );
            }

            return (
            <div
              key={`${e.type}-${e.rank}-${e.title}`}
              style={{
                display: "grid",
                gridTemplateColumns: "36px auto 1fr auto",
                gap: 10,
                alignItems: "center",
                background: EDGE_CARD_BG,
                border: `1px solid ${EDGE_CARD_BORDER}`,
                borderRadius: 12,
                padding: "14px 16px",
              }}
            >
              <div style={{ fontFamily: FONT_DISPLAY, fontSize: 28, color: "#4B5563", lineHeight: 1, textAlign: "center" }}>
                {e.rank}
              </div>
              <EdgeTypePill type={e.type} subtype={e.subtype} />
              <div style={{ minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15, fontWeight: 800, color: CSS_TEXT, marginBottom: 4, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {e.player_id != null && e.team_name ? <Logo team={e.team_name} size={24} /> : null}
                  <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{e.title}</span>
                </div>
                <div
                  style={{ fontSize: 12, color: "#6B7280", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                  title={edgeRateHover(e)}
                >
                  {edgeSubtitle(e)}
                </div>
              </div>
              <div style={{ display: "flex", gap: 20, alignItems: "center", flexShrink: 0 }}>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 9, fontWeight: 800, color: "#6B7280", letterSpacing: "0.08em" }}>{e.stat_label}</div>
                  <div style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 700, color: CSS_TEXT }}>{e.stat_value}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 9, fontWeight: 800, color: "#6B7280", letterSpacing: "0.08em" }}>{e.edge_label}</div>
                  <div style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: edgeColor }}>{e.edge_value}</div>
                </div>
              </div>
            </div>
            );
          })}
        </div>
      )}

      <p style={{ margin: "28px 0 0", fontSize: 12, color: "#6B7280", lineHeight: 1.55, textAlign: "center" }}>
        Edges are model disagreements with the market or a player&apos;s own baseline — not standalone probability picks. Past performance does not predict future results. Please wager responsibly.
      </p>
    </div>
  );
}

function TrendCardSkeleton({ title, icon }) {
  return (
    <div style={{
      background: EDGE_CARD_BG,
      border: `1px solid ${EDGE_CARD_BORDER}`,
      borderRadius: 12,
      padding: "16px 18px",
      minWidth: 0,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
        <span style={{ color: COL.model, display: "flex" }}>{icon}</span>
        <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.1em", color: "#9CA3AF" }}>{title}</span>
      </div>
      {[0, 1, 2, 3].map((i) => (
        <div
          key={i}
          className="trends-skeleton-row"
          style={{
            height: 44,
            background: "#1F2937",
            borderRadius: 6,
            marginBottom: 8,
          }}
        />
      ))}
    </div>
  );
}

function TrendCard({ icon, title, rows, renderValue, allowWrap, renderNameExtra, onRowClick, wideValue = false, showTeamLogo = false }) {
  const isMobile = useIsMobile();
  return (
    <div style={{
      background: EDGE_CARD_BG,
      border: `1px solid ${EDGE_CARD_BORDER}`,
      borderRadius: 12,
      padding: "16px 18px",
      minWidth: 0,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
        <span style={{ color: COL.model, display: "flex" }}>{icon}</span>
        <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.1em", color: "#9CA3AF" }}>{title}</span>
      </div>
      {!rows?.length ? (
        <div style={{ fontSize: 13, color: COL.textMuted }}>No data yet.</div>
      ) : (
        rows.map((row, i) => {
          const clickable = typeof onRowClick === "function";
          return (
            <div
              key={row.rank ?? i}
              role={clickable ? "button" : undefined}
              tabIndex={clickable ? 0 : undefined}
              onClick={clickable ? () => onRowClick(row) : undefined}
              onKeyDown={clickable ? (e) => { if (e.key === "Enter" || e.key === " ") onRowClick(row); } : undefined}
              style={{
                display: "grid",
                gridTemplateColumns: isMobile
                  ? "28px 1fr"
                  : (wideValue ? "28px minmax(0, 1fr) minmax(152px, auto)" : "28px minmax(0, 1fr) auto"),
                gap: 10,
                alignItems: "center",
                padding: "8px 0",
                borderTop: i > 0 ? `1px solid ${EDGE_CARD_BORDER}` : "none",
                cursor: clickable ? "pointer" : "default",
              }}
            >
              <div style={{ fontFamily: FONT_DISPLAY, fontSize: 22, color: "#4B5563", lineHeight: 1 }}>{row.rank}</div>
              <div style={{ minWidth: 0 }}>
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  fontSize: allowWrap ? 13 : (isMobile ? 13 : 14),
                  fontWeight: 800,
                  color: CSS_TEXT,
                  whiteSpace: allowWrap ? "normal" : "nowrap",
                  overflow: allowWrap ? "visible" : "hidden",
                  textOverflow: allowWrap ? "clip" : "ellipsis",
                  lineHeight: 1.25,
                }}
                >
                  {renderNameExtra ? renderNameExtra(row) : null}
                  {showTeamLogo && row.team_name ? <Logo team={row.team_name} size={24} /> : null}
                  <span style={{ minWidth: 0 }}>{row.name || row.player_name || row.pitcher_name || row.team_name || row.matchup || row.pick}</span>
                </div>
                <div style={{
                  fontSize: 11,
                  color: "#6B7280",
                  marginTop: 2,
                  whiteSpace: allowWrap ? "normal" : "nowrap",
                  overflow: allowWrap ? "visible" : "hidden",
                  textOverflow: allowWrap ? "clip" : "ellipsis",
                  lineHeight: 1.35,
                }}
                >
                  {row.meta || row.description || row.detail}
                </div>
                {isMobile && (
                  <div style={{ marginTop: 4 }}>
                    {renderValue(row)}
                  </div>
                )}
              </div>
              {!isMobile && renderValue(row)}
            </div>
          );
        })
      )}
    </div>
  );
}

function TeamTrendBadge({ teamName }) {
  const theme = getTeamTheme(teamName);
  return (
    <div style={{
      width: 28,
      height: 28,
      borderRadius: "50%",
      background: theme.primary,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 9,
      fontWeight: 900,
      color: "#fff",
      flexShrink: 0,
      letterSpacing: "0.02em",
    }}
    >
      {teamAbbr(teamName)}
    </div>
  );
}

function mapModelEdgeTrendRow(r, i, games, live) {
  const edgeRow = {
    type: r.type,
    title: r.title || r.pick_description,
    pick_description: r.pick_description,
    game_id: r.game_id,
    direction: r.direction,
    comparison_value_num: r.comparison_value_num,
  };
  return {
    rank: r.rank ?? i + 1,
    name: r.title || r.pick_description,
    meta: r.detail || r.detail_line,
    rate_detail: r.rate_detail || r.rate_detail_line,
    stat_label: r.stat_label || (r.type === "total" ? "PROJ" : "MODEL"),
    stat_value: r.stat_value ?? "—",
    edge_label: r.edge_label || "EDGE",
    edge: r.edge_value ?? "—",
    game_id: r.game_id,
    player_id: r.player_id,
    team_id: r.team_id,
    team_abbr: r.team_abbr,
    team_name: r.team_name,
    type: r.type,
    direction: r.direction,
    comparison_value_num: r.comparison_value_num,
    grade: gradeModelEdge(edgeRow, games, live),
  };
}

function TrendsPanel({ enabled, refreshKey, onGoToEdges, games = [], live = {}, date }) {
  const isMobile = useIsMobile();
  const slateDate = date || pacificTodayStr();
  const { data, error, loading } = useTrendsData(enabled, slateDate, refreshKey);
  const { data: edgesData } = useEdgesData(enabled, slateDate, refreshKey);

  const modelEdges = useMemo(() => {
    const raw = (data?.model_edges?.length > 0)
      ? data.model_edges
      : (edgesData?.edges || []).slice(0, 3);
    return raw.map((r, i) => mapModelEdgeTrendRow(r, i, games, live));
  }, [data?.model_edges, edgesData?.edges, games, live]);

  const flameIcon = (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M12 22c4-3 6-6.5 6-10a6 6 0 0 0-12 0c0 3.5 2 7 6 10z" />
    </svg>
  );
  const boltIcon = (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
    </svg>
  );
  const trendIcon = (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M3 17l6-6 4 4 8-8" /><path d="M14 7h7v7" />
    </svg>
  );
  const exchangeIcon = (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M7 7h11l-3-3M17 17H6l3 3" />
    </svg>
  );
  const coldIcon = (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M12 22V2M12 2l3 3M12 2L9 5M12 22l3-3M12 22l-3-3M5 12H2M22 12h-3M7 7L5 5M19 19l-2-2M7 17l-2 2M19 5l-2 2" />
    </svg>
  );
  const edgeIcon = (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M3 17l6-6 4 4 8-8" /><path d="M14 7h7v7" />
    </svg>
  );

  const trendsGridStyle = {
    display: "grid",
    gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr",
    gap: 14,
  };

  if (loading && !data) {
    return (
      <div style={pageShellStyle(1100, { horizontal: isMobile ? -12 : 0 })}>
        <h1 style={{ fontFamily: FONT_DISPLAY, fontSize: 42, fontWeight: 400, color: COL.model, margin: "0 0 8px", letterSpacing: "0.04em", lineHeight: 1 }}>
          Trends
        </h1>
        <p style={{ margin: "0 0 24px", fontSize: 14, lineHeight: 1.6, color: "#9CA3AF", maxWidth: 560 }}>
          What&apos;s heating up across the league — surfaced automatically from rolling Statcast data and recent results.
        </p>
        <div style={trendsGridStyle}>
          <TrendCardSkeleton title="Hottest Hitters · 14d xwOBA" icon={flameIcon} />
          <TrendCardSkeleton title="K Leaders · Last 3 Starts" icon={boltIcon} />
          <TrendCardSkeleton title="Teams Trending Up · 10D" icon={trendIcon} />
          <TrendCardSkeleton title="Biggest Line Moves · Today" icon={exchangeIcon} />
          <TrendCardSkeleton title="Struggling Starters · Last 3" icon={coldIcon} />
          <TrendCardSkeleton title="Model Edges · Today" icon={edgeIcon} />
        </div>
      </div>
    );
  }
  if (error && !data) {
    return (
      <div style={{ maxWidth: 640, margin: "24px auto", padding: 16, background: COL.card, border: `1px solid ${COL.border}`, borderRadius: 12, color: COL.textSecondary, fontSize: 13 }}>
        Could not load trends: {error}
      </div>
    );
  }

  const hottest = (data?.hottest_hitters || []).map((r) => ({
    ...r,
    name: r.player_name,
    meta: r.team_name || "—",
  }));
  const mostHr = (data?.most_hr_last10 || []).map((r) => ({
    ...r,
    name: r.player_name,
    meta: r.team_name || "—",
  }));
  const mostHits = (data?.most_hits_last10 || []).map((r) => ({
    ...r,
    name: r.player_name,
    meta: r.team_name || "—",
  }));
  const hittingStreaks = (data?.hitting_streaks || []).map((r) => ({
    ...r,
    name: r.player_name,
    meta: r.team_name || "—",
  }));
  const coldBats = (data?.cold_bats_last10 || []).map((r) => ({
    ...r,
    name: r.player_name,
    meta: `${r.team_name || "—"} · ${r.hits ?? 0} H`,
  }));
  const kLeaders = (data?.k_leaders || []).map((r) => ({
    ...r,
    name: r.pitcher_name,
    meta: `${r.team_name || "—"} · ${r.k_per_start} K/start`,
  }));
  const bestEra = (data?.best_era_last3 || []).map((r) => ({
    ...r,
    name: r.pitcher_name,
    meta: `${r.team_name || "—"} · ${r.meta || ""}`,
  }));
  const teams = (data?.teams_trending || []).map((r) => ({
    ...r,
    name: r.team_name,
    meta: `${r.wins}-${r.losses} · ${r.run_diff > 0 ? "+" : ""}${r.run_diff} run diff`,
    streak: r.streak,
  }));
  const lineMoves = (data?.line_moves || []).map((r) => ({
    ...r,
    name: r.matchup,
    meta: r.description,
    magnitude: r.magnitude,
    direction: r.direction,
  }));
  const coldPitchers = (data?.cold_pitchers || []).map((r) => ({
    ...r,
    name: r.pitcher_name,
    meta: `${r.team_name || "—"} · ${r.meta || ""}`,
  }));
  const bestBullpens = (data?.best_bullpens_last7 || []).map((r) => ({
    ...r,
    name: r.team_name,
    meta: r.meta || "ERA last 7d",
  }));

  const subHeader = (label) => (
    <div style={{
      gridColumn: "1 / -1",
      margin: "8px 0 -2px",
      display: "flex",
      alignItems: "center",
      gap: 10,
    }}>
      <div style={{ width: 4, height: 18, borderRadius: 3, background: COL.model }} />
      <div style={{ fontSize: 12, fontWeight: 900, color: COL.text, letterSpacing: "0.12em", textTransform: "uppercase" }}>{label}</div>
      <div style={{ flex: 1, height: 1, background: EDGE_CARD_BORDER }} />
    </div>
  );

  return (
    <div style={pageShellStyle(1100, { horizontal: isMobile ? -12 : 0 })}>
      <h1 style={{ fontFamily: FONT_DISPLAY, fontSize: 42, fontWeight: 400, color: COL.model, margin: "0 0 8px", letterSpacing: "0.04em", lineHeight: 1 }}>
        Trends
      </h1>
      <p style={{ margin: "0 0 24px", fontSize: 14, lineHeight: 1.6, color: "#9CA3AF", maxWidth: 560 }}>
        What&apos;s heating up across the league — surfaced automatically from rolling Statcast data and recent results.
      </p>

      <div style={trendsGridStyle}>
        {subHeader("Hitters")}
        <TrendCard
          title="Hottest Hitters · 14d xwOBA"
          icon={flameIcon}
          rows={hottest}
          showTeamLogo
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.positive }}>
              {row.xwoba_14d != null ? `.${Number(row.xwoba_14d).toFixed(3).slice(2)}` : "—"}
            </span>
          )}
        />
        <TrendCard
          title="Most HR · Last 10 Games"
          icon={flameIcon}
          rows={mostHr}
          showTeamLogo
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.positive }}>
              {row.hr ?? "—"} HR
            </span>
          )}
        />
        <TrendCard
          title="Most Hits · Last 10 Games"
          icon={trendIcon}
          rows={mostHits}
          showTeamLogo
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.positive }}>
              {row.hits ?? "—"} H
            </span>
          )}
        />
        <TrendCard
          title="Hottest Streaks"
          icon={boltIcon}
          rows={hittingStreaks}
          showTeamLogo
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.model }}>
              {row.streak ?? "—"} G
            </span>
          )}
        />
        <TrendCard
          title="Cold Bats · Most K Last 10"
          icon={coldIcon}
          rows={coldBats}
          showTeamLogo
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.negative }}>
              {row.strikeouts ?? "—"} K
            </span>
          )}
        />
        {subHeader("Pitching")}
        <TrendCard
          title="K Leaders · Last 3 Starts"
          icon={boltIcon}
          rows={kLeaders}
          showTeamLogo
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.model }}>
              {row.total_k} K
            </span>
          )}
        />
        <TrendCard
          title="Best ERA · Last 3 Starts"
          icon={trendIcon}
          rows={bestEra}
          showTeamLogo
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.positive }}>
              {row.era != null ? Number(row.era).toFixed(2) : "—"}
            </span>
          )}
        />
        <TrendCard
          title="Struggling Starters · Last 3"
          icon={coldIcon}
          rows={coldPitchers}
          showTeamLogo
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.negative }}>
              {row.era != null ? Number(row.era).toFixed(2) : "—"}
            </span>
          )}
        />
        <TrendCard
          title="Best Bullpens · Last 7 Days"
          icon={trendIcon}
          rows={bestBullpens}
          renderNameExtra={(row) => <TeamTrendBadge teamName={row.name || row.team_name} />}
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.positive }}>
              {row.era != null ? Number(row.era).toFixed(2) : "—"}
            </span>
          )}
        />
        {subHeader("Teams & Market")}
        <TrendCard
          title="Teams Trending Up · 10D"
          icon={trendIcon}
          rows={teams}
          renderNameExtra={(row) => <TeamTrendBadge teamName={row.name || row.team_name} />}
          renderValue={(row) => (
            <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color: COL.positive }}>
              {row.streak}
            </span>
          )}
        />
        <TrendCard
          title="Biggest Line Moves · Today"
          icon={exchangeIcon}
          rows={lineMoves}
          allowWrap
          renderValue={(row) => {
            const up = row.direction === "up";
            const color = up ? COL.positive : COL.negative;
            const arrow = up ? "↑" : "↓";
            return (
              <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 800, color }}>
                {arrow}{row.magnitude}
              </span>
            );
          }}
        />
        <TrendCard
          title="Model Edges · Today"
          icon={edgeIcon}
          rows={modelEdges}
          allowWrap
          wideValue
          showTeamLogo
          onRowClick={onGoToEdges ? () => onGoToEdges() : undefined}
          renderValue={(row) => {
            const edgeColor = row.direction === "under" ? "#60A5FA" : COL.positive;
            return (
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 9, fontWeight: 800, color: "#6B7280", letterSpacing: "0.06em" }}>{row.stat_label}</div>
                <div style={{ fontFamily: FONT_MONO, fontSize: 13, fontWeight: 700, color: CSS_TEXT }}>{row.stat_value || "—"}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 9, fontWeight: 800, color: "#6B7280", letterSpacing: "0.06em" }}>{row.edge_label}</div>
                <div style={{ fontFamily: FONT_MONO, fontSize: 13, fontWeight: 800, color: edgeColor }}>{row.edge || "—"}</div>
              </div>
              <EdgeGradeMark result={row.grade} />
            </div>
            );
          }}
        />
      </div>
    </div>
  );
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
      borderRadius: 12,
      padding: "16px 18px",
      minWidth: 0,
    }}>
      <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.1em", textTransform: "uppercase", color: "#6B7280" }}>
        {label}
      </div>
      <div style={{
        fontSize: 28,
        fontWeight: 800,
        marginTop: 6,
        color: accent || CSS_TEXT,
        fontFamily: mono ? FONT_MONO : FONT_BODY,
        fontVariantNumeric: "tabular-nums",
        letterSpacing: "-0.02em",
      }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: "#6B7280", marginTop: 4, fontFamily: mono ? FONT_MONO : FONT_BODY }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function PerfSectionTitle({ label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "28px 0 14px" }}>
      <div style={{ width: 4, height: 22, borderRadius: 2, background: COL.model, flexShrink: 0 }} />
      <span style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.12em", textTransform: "uppercase", color: CSS_TEXT }}>
        {label}
      </span>
    </div>
  );
}

function CalibrationBarRow({ label, predPct, actualPct }) {
  if (predPct == null && actualPct == null) return null;
  const pred = predPct ?? 0;
  const act = actualPct ?? 0;
  const wellCalibrated = predPct != null && actualPct != null && Math.abs(pred - act) <= 3;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, gap: 12 }}>
        <span style={{ color: "#9CA3AF", fontWeight: 600, fontSize: 13 }}>{label}</span>
        <span style={{ color: "#6B7280", fontFamily: FONT_MONO, fontSize: 12, whiteSpace: "nowrap" }}>
          {actualPct != null ? `${actualPct}%` : "—"}
          {" / "}
          {predPct != null ? `${predPct}%` : "—"}
          {wellCalibrated && <span style={{ color: COL.positive, marginLeft: 6 }}>✓</span>}
        </span>
      </div>
      <div style={{ position: "relative", height: 10, borderRadius: 5, background: "#1F2937" }}>
        {actualPct != null && (
          <div
            style={{
              height: "100%",
              width: `${Math.min(100, Math.max(0, act))}%`,
              borderRadius: 5,
              background: COL.model,
            }}
          />
        )}
        {predPct != null && (
          <div
            style={{
              position: "absolute",
              top: -2,
              left: `${Math.min(100, Math.max(0, pred))}%`,
              width: 2,
              height: 14,
              background: CSS_TEXT,
              transform: "translateX(-1px)",
            }}
          />
        )}
      </div>
    </div>
  );
}

function CalibrationBarChart({ rows, caption }) {
  const items = (rows || []).filter((r) => r && (r.pred_pct != null || r.actual_pct != null));
  if (!items.length) {
    return <div style={{ color: COL.textMuted, fontSize: 13 }}>Not enough graded data yet.</div>;
  }
  const labelKey = items[0].bucket != null ? "bucket" : "line";
  return (
    <div
      style={{
        background: COL.card,
        border: `1px solid ${COL.border}`,
        borderRadius: 12,
        padding: "18px 20px",
      }}
    >
      {caption && (
        <p style={{ margin: "0 0 18px", fontSize: 13, lineHeight: 1.6, color: "#9CA3AF" }}>{caption}</p>
      )}
      {items.map((r) => (
        <CalibrationBarRow
          key={r[labelKey]}
          label={r[labelKey]}
          predPct={r.pred_pct}
          actualPct={r.actual_pct}
        />
      ))}
      <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 11, color: "#6B7280" }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: COL.model }} />
          Actual win rate
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 2, height: 10, background: CSS_TEXT }} />
          Model predicted
        </span>
      </div>
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

function ModelPerformancePanel({ enabled, refreshKey }) {
  const [v9Open, setV9Open] = useState(false);
  const { data, error, loading } = useAccuracyData(enabled, refreshKey);

  if (loading && !data) {
    return (
      <div style={pageShellStyle(920)}>
        <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", margin: 0 }}>Loading model performance…</p>
      </div>
    );
  }
  if (error && !data) {
    return (
      <div style={{ maxWidth: 640, margin: "24px auto", padding: 16, background: COL.card, border: `1px solid ${COL.border}`, borderRadius: 12, color: COL.textSecondary, fontSize: 13 }}>
        Could not load model performance data: {error}
      </div>
    );
  }
  if (!data) return null;

  const headline = data.headline || {};
  const meta = data.meta || {};
  const v9History = data.v9_history || null;
  const v9Headline = v9History?.headline || {};
  const buckets = data.buckets || [];
  const mlCalibration = data.ml_calibration || [];
  const pitcherK = data.pitcher_k_calibration || { lines: [], starters_graded: 0 };
  const pitcherWalks = data.pitcher_walks_calibration || { lines: [], starters_graded: 0 };
  const pitcherHits = data.pitcher_hits_calibration || { lines: [], starters_graded: 0 };
  const pitcherEr = data.pitcher_er_calibration || { lines: [], starters_graded: 0 };
  const mlCum = data.daily_cumulative || [];

  const ece = headline.calibration_error_pct;
  const eceSub = ece != null
    ? (ece <= 2 ? "lower is better · excellent" : ece <= 4 ? "lower is better · good" : "lower is better")
    : null;
  const eceColor = ece != null && ece <= 4 ? COL.positive : CSS_TEXT;

  const brier = headline.brier_score;
  const brierMkt = headline.brier_market;
  let brierSub = null;
  if (brier != null && brierMkt != null) {
    brierSub = brier < brierMkt
      ? `market: ${brierMkt} · model sharper`
      : brier > brierMkt
        ? `market: ${brierMkt} · market sharper`
        : `market: ${brierMkt} · tied`;
  } else if (brierMkt != null) {
    brierSub = `market: ${brierMkt}`;
  }
  const brierColor = brier != null && brierMkt != null && brier <= brierMkt ? COL.positive : CSS_TEXT;

  const acc = headline.accuracy_pct;
  const pickHome = headline.pick_home_baseline_pct;
  const accSub = pickHome != null ? `vs ${pickHome}% pick-home` : null;

  return (
    <div style={pageShellStyle(920)}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, flexWrap: "wrap", marginBottom: 8 }}>
        <div>
          <h1 style={{
            fontFamily: FONT_BODY,
            fontSize: "clamp(28px, 4vw, 36px)",
            fontWeight: 800,
            color: CSS_TEXT,
            margin: "0 0 8px",
            letterSpacing: "-0.03em",
          }}>
            Model Performance
          </h1>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "#9CA3AF", maxWidth: 520 }}>
            How well-calibrated are the model&apos;s probabilities? Tracked on games it never saw in training.
          </p>
          <p style={{ margin: "8px 0 0", fontSize: 13, lineHeight: 1.55, color: "#6B7280", maxWidth: 560 }}>
            Grading v10 predictions from May 28, 2026 onward.
          </p>
        </div>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 12px",
            borderRadius: 999,
            border: `1px solid ${COL.border}`,
            background: "rgba(31,41,55,0.6)",
            fontSize: 12,
            fontFamily: FONT_MONO,
            color: "#9CA3AF",
            flexShrink: 0,
          }}
        >
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: COL.positive, flexShrink: 0 }} />
          v10 · May 28 – today
        </div>
      </div>

      <div
        style={{
          marginBottom: 20,
          padding: "12px 14px",
          borderRadius: 10,
          border: "1px solid rgba(245,158,11,0.35)",
          background: "rgba(245,158,11,0.08)",
          fontSize: 13,
          lineHeight: 1.55,
          color: "#FCD34D",
        }}
      >
        {meta.small_sample_note || "v10 launched May 28 — these metrics are based on a limited early sample and will stabilize as more games are graded."}
      </div>

      <PerfSectionTitle label="Moneyline — Headline Metrics" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
        <KpiCard
          label="Games Graded"
          value={headline.games_graded != null ? String(headline.games_graded) : "—"}
          sub="May 28 – today"
        />
        <KpiCard
          label="Calibration Error"
          value={ece != null ? `${ece}%` : "—"}
          sub={eceSub}
          accent={eceColor}
        />
        <KpiCard
          label="Brier Score"
          value={brier != null ? String(brier) : "—"}
          sub={brierSub}
          accent={brierColor}
        />
        <KpiCard
          label="Accuracy"
          value={acc != null ? `${acc}%` : "—"}
          sub={accSub}
        />
      </div>

      <PerfSectionTitle label="Calibration — Predicted vs Actual" />
      <CalibrationBarChart
        rows={mlCalibration}
        caption="When the model says a team has a 60% chance to win, that team should win about 60% of the time. The amber bar is what actually happened; the white line is what the model predicted. Closer together = better calibrated."
      />

      <PerfSectionTitle label="Confidence Bucket Performance" />
      <AccuracyBucketTable buckets={buckets} />

      <PerfSectionTitle label="Pitcher Strikeouts — Calibration" />
      <CalibrationBarChart
        rows={pitcherK.lines || []}
        caption={`For pitcher K props: when the model says a pitcher has a ~54% chance to go over a strikeout line, how often does that actually happen?${pitcherK.starters_graded ? ` (${pitcherK.starters_graded} SP appearances graded)` : ""}`}
      />

      <PerfSectionTitle label="Pitcher Walks — Calibration" />
      <CalibrationBarChart
        rows={pitcherWalks.lines || []}
        caption={`Walks allowed O/U calibration on graded SP starts.${pitcherWalks.starters_graded ? ` (${pitcherWalks.starters_graded} SP appearances graded)` : ""}`}
      />

      <PerfSectionTitle label="Pitcher Hits Allowed — Calibration" />
      <CalibrationBarChart
        rows={pitcherHits.lines || []}
        caption={`Hits allowed O/U calibration on graded SP starts.${pitcherHits.starters_graded ? ` (${pitcherHits.starters_graded} SP appearances graded)` : ""}`}
      />

      <PerfSectionTitle label="Pitcher Earned Runs — Calibration" />
      <CalibrationBarChart
        rows={pitcherEr.lines || []}
        caption={`Earned runs O/U calibration on graded SP starts.${pitcherEr.starters_graded ? ` (${pitcherEr.starters_graded} SP appearances graded)` : ""}`}
      />

      {mlCum.length >= 2 && (
        <>
          <PerfSectionTitle label="Illustrative P&L Simulation" />
          <p style={{ margin: "0 0 12px", fontSize: 12, color: "#6B7280", lineHeight: 1.5 }}>
            Illustrative only — hypothetical $10 flat-stake moneyline simulation at actual book odds. Not a recommended strategy or guaranteed result.
          </p>
          <div style={{ opacity: 0.72 }}>
            <PnlLineChart rows={mlCum} />
          </div>
        </>
      )}

      {v9History && (
        <div style={{ marginTop: 32, borderTop: `1px solid ${COL.border}`, paddingTop: 20 }}>
          <button
            type="button"
            onClick={() => setV9Open((o) => !o)}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              width: "100%",
              gap: 12,
              padding: "12px 14px",
              borderRadius: 10,
              border: `1px solid ${COL.border}`,
              background: "rgba(31,41,55,0.45)",
              color: COL.textSecondary,
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              textAlign: "left",
            }}
          >
            <span>{v9History.label || "Previous model (v9), Apr 14 – May 27"}</span>
            <span style={{ fontFamily: FONT_MONO, fontSize: 12, color: "#9CA3AF", flexShrink: 0 }}>
              {v9Headline.games_graded != null ? `${v9Headline.games_graded} games` : "—"}
              {" · "}
              {v9Open ? "▲" : "▼"}
            </span>
          </button>
          {v9Open && (
            <div style={{ marginTop: 16 }}>
              <p style={{ margin: "0 0 14px", fontSize: 12, lineHeight: 1.55, color: "#6B7280" }}>
                Legacy v9 architecture (pre–May 28). Shown for transparency — not the current production model.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginBottom: 20 }}>
                <KpiCard
                  label="Games Graded"
                  value={v9Headline.games_graded != null ? String(v9Headline.games_graded) : "—"}
                  sub="Apr 14 – May 27"
                />
                <KpiCard
                  label="Calibration Error"
                  value={v9Headline.calibration_error_pct != null ? `${v9Headline.calibration_error_pct}%` : "—"}
                  sub="v9 only"
                />
                <KpiCard
                  label="Brier Score"
                  value={v9Headline.brier_score != null ? String(v9Headline.brier_score) : "—"}
                  sub={v9Headline.brier_market != null ? `market: ${v9Headline.brier_market}` : null}
                />
                <KpiCard
                  label="Accuracy"
                  value={v9Headline.accuracy_pct != null ? `${v9Headline.accuracy_pct}%` : "—"}
                  sub="v9 only"
                />
              </div>
              <CalibrationBarChart
                rows={v9History.ml_calibration || []}
                caption="v9 moneyline calibration (Apr 14 – May 27). This model is no longer in production."
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function divisionDisplayName(name) {
  const s = name || "";
  return s.replace("American League", "AL").replace("National League", "NL");
}

function StandingsTeamCell({ row }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 9, minWidth: 128 }}>
      <Logo team={row.team_name} size={28} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 900, color: COL.text, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {teamNickname(row.team_name)}
        </div>
      </div>
    </div>
  );
}

function StandingsPanel({ enabled, date, refreshKey }) {
  const isMobile = useIsMobile();
  const { data, error, loading } = useStandingsData(enabled, date, refreshKey);
  const [mode, setMode] = useState("current");
  const rows = data?.standings || [];
  const grouped = useMemo(() => {
    const leagues = { 103: { label: "American League", divisions: {} }, 104: { label: "National League", divisions: {} } };
    for (const row of rows) {
      const leagueId = Number(row.league_id);
      if (!leagues[leagueId]) leagues[leagueId] = { label: row.league_name || `League ${leagueId}`, divisions: {} };
      const divKey = String(row.division_id || row.division_name);
      if (!leagues[leagueId].divisions[divKey]) {
        leagues[leagueId].divisions[divKey] = {
          label: divisionDisplayName(row.division_name_short || row.division_name),
          rows: [],
        };
      }
      leagues[leagueId].divisions[divKey].rows.push(row);
    }
    for (const league of Object.values(leagues)) {
      for (const div of Object.values(league.divisions)) {
        div.rows.sort((a, b) => Number(a.rank || 99) - Number(b.rank || 99));
      }
    }
    return leagues;
  }, [rows]);

  const headerDateLabel = new Date(`${date || pacificTodayStr()}T12:00:00`).toLocaleDateString("en-US", {
    timeZone: "America/Los_Angeles",
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  const thStyle = {
    padding: "9px 7px",
    textAlign: "right",
    fontSize: 10,
    fontWeight: 900,
    color: COL.textMuted,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    background: "#111827",
    whiteSpace: "nowrap",
    fontFamily: FONT_MONO,
  };
  const tdNum = {
    padding: "9px 7px",
    textAlign: "right",
    fontSize: 12,
    fontWeight: 800,
    color: COL.text,
    fontVariantNumeric: "tabular-nums",
    fontFamily: FONT_MONO,
    whiteSpace: "nowrap",
  };
  const streakColor = (streak) => {
    const s = String(streak || "").toUpperCase();
    if (s.startsWith("W")) return COL.positive;
    if (s.startsWith("L")) return COL.negative;
    return COL.text;
  };
  const playoffOddsColor = (odds) => {
    const p = Number(odds);
    if (!Number.isFinite(p)) return COL.textMuted;
    if (p > 0.6) return COL.positive;
    if (p >= 0.3) return COL.model;
    return COL.textMuted;
  };
  const formatPlayoffOdds = (odds) => {
    const p = Number(odds);
    if (!Number.isFinite(p)) return "—";
    if (p >= 0.99) return ">99%";
    if (p <= 0.01) return "<1%";
    return `${Math.round(p * 100)}%`;
  };

  return (
    <div style={pageShellStyle(1160, { bottom: isMobile ? 96 : 48, horizontal: isMobile ? -12 : 0 })}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "flex-start", marginBottom: 20, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ fontFamily: FONT_DISPLAY, fontSize: 42, fontWeight: 400, color: COL.model, margin: "0 0 8px", letterSpacing: "0.04em", lineHeight: 1 }}>
            Standings
          </h1>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "#9CA3AF", maxWidth: 620 }}>
            MLB division standings with The Hot Corner projected final records and playoff odds from model win probabilities.
          </p>
          <div style={{ marginTop: 8, fontSize: 11, color: COL.textMuted, fontWeight: 700 }}>
            Snapshot: {headerDateLabel}{data?.meta?.simulations ? ` · ${Number(data.meta.simulations).toLocaleString()} simulations` : ""}
          </div>
        </div>
        <div style={{ display: "inline-flex", border: `1px solid ${COL.border}`, borderRadius: 999, overflow: "hidden", background: COL.card }}>
          {[
            ["current", "Current"],
            ["projected", "Projected"],
          ].map(([id, label]) => {
            const active = mode === id;
            return (
              <button
                key={id}
                type="button"
                onClick={() => setMode(id)}
                style={{
                  border: "none",
                  background: active ? COL.model : "transparent",
                  color: active ? "#111827" : COL.textSecondary,
                  padding: "8px 14px",
                  fontSize: 12,
                  fontWeight: 900,
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      {loading && <p style={{ color: COL.textSecondary, fontSize: 14 }}>Loading standings...</p>}
      {error && <p style={{ color: COL.negative, fontSize: 14 }}>Standings unavailable: {error}</p>}
      {!loading && !error && rows.length === 0 && <p style={{ color: COL.textSecondary, fontSize: 14 }}>No standings snapshot is available yet.</p>}

      {!loading && !error && rows.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {[104, 103].map((leagueId) => {
            const league = grouped[leagueId];
            return (
              <div key={leagueId} style={{ border: `1px solid ${COL.border}`, borderRadius: 14, background: COL.card, overflow: "hidden" }}>
                <div style={{ padding: "14px 16px", borderBottom: `1px solid ${COL.border}`, background: "#1F2937", fontSize: 15, fontWeight: 900, color: COL.text }}>
                  {league.label}
                </div>
                {Object.entries(league.divisions).map(([divId, div]) => (
                  <div key={divId}>
                    <div style={{ padding: "10px 14px", background: "#0D1420", color: COL.model, fontSize: 11, fontWeight: 900, letterSpacing: "0.08em", textTransform: "uppercase", borderTop: `1px solid ${COL.border}` }}>
                      {div.label}
                    </div>
                    <div style={{ width: "100%", overflowX: "auto", WebkitOverflowScrolling: "touch", boxSizing: "border-box" }}>
                      <table
                        style={{
                          width: "100%",
                          minWidth: mode === "projected" ? 780 : 580,
                          borderCollapse: "collapse",
                          tableLayout: "auto",
                        }}
                      >
                        <thead>
                          <tr>
                            <th style={{ ...thStyle, textAlign: "right", paddingLeft: 14 }}>#</th>
                            <th style={{ ...thStyle, textAlign: "left" }}>Team</th>
                            <th style={thStyle}>W-L</th>
                            <th style={thStyle}>PCT</th>
                            <th style={thStyle}>GB</th>
                            <th style={thStyle}>STRK</th>
                            <th style={thStyle}>L10</th>
                            <th style={thStyle}>DIFF</th>
                            {mode === "projected" && <th style={thStyle}>PROJ</th>}
                            {mode === "projected" && <th style={thStyle}>PLAYOFF</th>}
                          </tr>
                        </thead>
                        <tbody>
                          {div.rows.map((row, idx) => {
                            const rd = Number(row.run_diff || 0);
                            return (
                              <tr key={row.team_id} style={{ borderTop: `1px solid ${COL.border}`, background: idx % 2 ? "#0D1420" : "transparent" }}>
                                <td style={{ ...tdNum, color: COL.textMuted, paddingLeft: 14 }}>{row.rank || "—"}</td>
                                <td style={{ padding: "10px 8px", textAlign: "left", whiteSpace: "nowrap" }}>
                                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                    <Logo team={row.team_name} size={26} />
                                    <span style={{ fontSize: 14, fontWeight: 800, color: COL.text }}>{teamNickname(row.team_name)}</span>
                                  </div>
                                </td>
                                <td style={{ ...tdNum, whiteSpace: "nowrap", fontWeight: 800 }}>{row.wins}-{row.losses}</td>
                                <td style={tdNum}>{Number(row.pct || 0).toFixed(3).replace(/^0/, "")}</td>
                                <td style={tdNum}>{row.games_back || "-"}</td>
                                <td style={{ ...tdNum, color: streakColor(row.streak) }}>{row.streak || "—"}</td>
                                <td style={{ ...tdNum, whiteSpace: "nowrap" }}>{row.last_10 || "—"}</td>
                                <td style={{ ...tdNum, color: rd > 0 ? COL.positive : rd < 0 ? COL.negative : COL.textMuted }}>
                                  {rd > 0 ? `+${rd}` : rd}
                                </td>
                                {mode === "projected" && (
                                  <td style={{ ...tdNum, color: COL.model, whiteSpace: "nowrap" }}>{row.projected_record || "—"}</td>
                                )}
                                {mode === "projected" && (
                                  <td style={{ ...tdNum, color: playoffOddsColor(row.playoff_odds) }}>{formatPlayoffOdds(row.playoff_odds)}</td>
                                )}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function transactionCategoryLabel(category) {
  if (category === "injury") return "Injuries";
  if (category === "callup") return "Callups";
  if (category === "trade") return "Trades";
  if (category === "signing") return "Signings";
  if (category === "dfa") return "DFA";
  return "Other";
}

function transactionPillStyle(category) {
  const colors = {
    injury: { bg: "rgba(239,68,68,0.16)", border: "rgba(239,68,68,0.45)", color: "#FCA5A5" },
    callup: { bg: "rgba(16,185,129,0.16)", border: "rgba(16,185,129,0.45)", color: "#6EE7B7" },
    trade: { bg: "rgba(245,158,11,0.16)", border: "rgba(245,158,11,0.45)", color: COL.model },
    dfa: { bg: "rgba(107,114,128,0.18)", border: "rgba(107,114,128,0.45)", color: "#D1D5DB" },
    signing: { bg: "rgba(59,130,246,0.16)", border: "rgba(59,130,246,0.45)", color: "#93C5FD" },
    other: { bg: "rgba(156,163,175,0.12)", border: "rgba(156,163,175,0.28)", color: "#9CA3AF" },
  };
  const c = colors[category] || colors.other;
  return {
    display: "inline-flex",
    alignItems: "center",
    padding: "4px 8px",
    borderRadius: 999,
    background: c.bg,
    border: `1px solid ${c.border}`,
    color: c.color,
    fontSize: 10,
    fontWeight: 900,
    letterSpacing: "0.06em",
    textTransform: "uppercase",
    whiteSpace: "nowrap",
  };
}

function TransactionTeamBadge({ teamName }) {
  const badgeColor = teamBadgeColor(teamName);
  const onBadge = hexLuminance(badgeColor) < 0.52 ? "#FFFFFF" : "#111827";
  return (
    <div style={{
      minWidth: 38,
      height: 26,
      borderRadius: 8,
      background: badgeColor,
      border: "1px solid rgba(255,255,255,0.14)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.12)",
    }}>
      <span style={{ fontSize: 10, fontWeight: 900, color: onBadge, fontFamily: FONT_MONO }}>
        {teamAbbr(teamName)}
      </span>
    </div>
  );
}

function TransactionsPanel({ enabled, date, refreshKey }) {
  const isMobile = useIsMobile();
  const [days, setDays] = useState(14);
  const [teamFilter, setTeamFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const { data, error, loading } = useTransactionsData(enabled, date, days, refreshKey);
  const rows = data?.transactions || [];
  const teams = data?.teams || [];
  const typeOptions = [["all", "All"], ["injury", "Injuries"], ["callup", "Callups"], ["trade", "Trades"], ["signing", "Signings"]];
  const filtered = rows.filter((row) => (
    (teamFilter === "all" || String(row.team_id || "") === teamFilter)
    && (typeFilter === "all" || row.category === typeFilter)
  ));
  const grouped = filtered.reduce((acc, row) => {
    const key = row.transaction_date || "Unknown";
    if (!acc[key]) acc[key] = [];
    acc[key].push(row);
    return acc;
  }, {});
  const dates = Object.keys(grouped).sort().reverse();
  const selectStyle = {
    background: COL.card,
    color: COL.text,
    border: `1px solid ${COL.border}`,
    borderRadius: 10,
    padding: "8px 10px",
    fontSize: 12,
    fontWeight: 800,
    fontFamily: "inherit",
    outline: "none",
  };

  return (
    <div style={pageShellStyle(980, { horizontal: isMobile ? -12 : 0 })}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "flex-start", marginBottom: 18, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ fontFamily: FONT_DISPLAY, fontSize: 42, fontWeight: 400, color: COL.model, margin: "0 0 8px", letterSpacing: "0.04em", lineHeight: 1 }}>
            Transactions
          </h1>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "#9CA3AF", maxWidth: 640 }}>
            Recent roster moves that can reshape lineup quality, pitching depth, and model inputs.
          </p>
        </div>
        <div style={{ fontFamily: FONT_MONO, fontSize: 12, color: COL.textMuted, paddingTop: 7 }}>
          Last {days} days
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", marginBottom: 18 }}>
        <select value={teamFilter} onChange={(e) => setTeamFilter(e.target.value)} style={selectStyle}>
          <option value="all">All teams</option>
          {teams.map((team) => (
            <option key={team.team_id} value={String(team.team_id)}>{team.team_name}</option>
          ))}
        </select>
        <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} style={selectStyle}>
          {typeOptions.map(([id, label]) => (
            <option key={id} value={id}>{label}</option>
          ))}
        </select>
        {(teamFilter !== "all" || typeFilter !== "all") && (
          <button type="button" onClick={() => { setTeamFilter("all"); setTypeFilter("all"); }} style={{ ...selectStyle, cursor: "pointer" }}>
            Clear filters
          </button>
        )}
      </div>

      {loading && <p style={{ color: COL.textSecondary, fontSize: 14 }}>Loading transactions...</p>}
      {error && <p style={{ color: COL.negative, fontSize: 14 }}>Transactions unavailable: {error}</p>}
      {!loading && !error && filtered.length === 0 && (
        <div style={{ border: `1px solid ${COL.border}`, borderRadius: 14, background: COL.card, padding: 18, color: COL.textSecondary, fontSize: 14 }}>
          No transactions match the current filters.
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <div style={{ border: `1px solid ${COL.border}`, borderRadius: 16, background: COL.card, overflow: "hidden" }}>
          {dates.map((day, dayIdx) => (
            <div key={day}>
              <div style={{ padding: "11px 16px", background: "#111827", borderTop: dayIdx ? `1px solid ${COL.border}` : "none", borderBottom: `1px solid ${COL.border}`, color: COL.model, fontSize: 11, fontWeight: 900, letterSpacing: "0.09em", textTransform: "uppercase", fontFamily: FONT_MONO }}>
                {shortDateLabel(day)}
              </div>
              {grouped[day].map((row) => (
                <div
                  key={row.transaction_id}
                  style={
                    isMobile
                      ? { display: "flex", flexDirection: "column", gap: 8, padding: "14px 16px", borderBottom: `1px solid ${EDGE_CARD_BORDER}` }
                      : { display: "grid", gridTemplateColumns: "44px minmax(90px, 130px) 1fr", gap: 12, padding: "14px 16px", borderBottom: `1px solid ${EDGE_CARD_BORDER}`, alignItems: "start" }
                  }
                >
                  {isMobile ? (
                    <>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                        <TransactionTeamBadge teamName={row.team_name} />
                        <span style={transactionPillStyle(row.category)}>{transactionCategoryLabel(row.category)}</span>
                        <span style={{ fontSize: 11, color: COL.textMuted, fontWeight: 800 }}>{teamNickname(row.team_name || "—")}</span>
                      </div>
                      <div>
                        <div style={{ fontSize: 14, color: COL.text, fontWeight: 800, lineHeight: 1.4 }}>
                          {row.description || `${row.player_name} — ${row.transaction_type || "Transaction"}`}
                        </div>
                        <div style={{ marginTop: 5, fontSize: 12, color: COL.textMuted }}>
                          {row.player_name}{row.transaction_type ? ` · ${row.transaction_type}` : ""}
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <TransactionTeamBadge teamName={row.team_name} />
                      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                        <span style={transactionPillStyle(row.category)}>{transactionCategoryLabel(row.category)}</span>
                        <span style={{ fontSize: 11, color: COL.textMuted, fontWeight: 800 }}>{teamNickname(row.team_name || "—")}</span>
                      </div>
                      <div>
                        <div style={{ fontSize: 14, color: COL.text, fontWeight: 800, lineHeight: 1.35 }}>
                          {row.description || `${row.player_name} — ${row.transaction_type || "Transaction"}`}
                        </div>
                        <div style={{ marginTop: 5, fontSize: 12, color: COL.textMuted }}>
                          {row.player_name}{row.transaction_type ? ` · ${row.transaction_type}` : ""}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {days < 60 && (
        <div style={{ display: "flex", justifyContent: "center", marginTop: 18 }}>
          <button type="button" onClick={() => setDays((d) => Math.min(60, d + 14))} style={{ background: "rgba(245,158,11,0.12)", border: `1px solid ${COL.model}`, color: COL.model, borderRadius: 999, padding: "9px 16px", fontSize: 12, fontWeight: 900, cursor: "pointer", fontFamily: "inherit" }}>
            Load more
          </button>
        </div>
      )}
    </div>
  );
}

function AboutIcon({ children }) {
  return (
    <div
      style={{
        width: 28,
        height: 28,
        borderRadius: 8,
        background: "rgba(245,158,11,0.12)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        color: COL.model,
      }}
    >
      {children}
    </div>
  );
}

function AboutModelCard({ icon, title, body }) {
  return (
    <div
      style={{
        background: COL.card,
        border: `1px solid ${COL.border}`,
        borderRadius: 12,
        padding: "20px 22px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <AboutIcon>{icon}</AboutIcon>
        <div style={{ fontSize: 16, fontWeight: 800, color: CSS_TEXT, letterSpacing: "-0.01em" }}>{title}</div>
      </div>
      <p style={{ margin: 0, fontSize: 14, lineHeight: 1.65, color: "#9CA3AF" }}>{body}</p>
    </div>
  );
}

function AuthorCredit({ style = {} }) {
  const linkStyle = {
    color: "#9CA3AF",
    textDecoration: "none",
    fontWeight: 500,
  };
  return (
    <span style={{ fontSize: 12, color: "#6B7280", lineHeight: 1.5, ...style }}>
      Built by{" "}
      <a
        href="https://www.linkedin.com/in/rodrigo-cuadra/"
        target="_blank"
        rel="noopener noreferrer"
        style={linkStyle}
      >
        Rodrigo Cuadra
      </a>
      <span style={{ color: COL.border }}> · </span>
      <a
        href="https://github.com/rcuadra27"
        target="_blank"
        rel="noopener noreferrer"
        style={linkStyle}
      >
        GitHub
      </a>
    </span>
  );
}

function AboutUsPanel() {
  return (
    <div style={pageShellStyle(880, { horizontal: 4 })}>
      <div style={{ marginBottom: 32 }}>
        <div
          style={{
            fontSize: 11,
            fontWeight: 800,
            letterSpacing: "0.14em",
            color: COL.model,
            marginBottom: 10,
          }}
        >
          ABOUT
        </div>
        <h1
          style={{
            fontFamily: FONT_BODY,
            fontSize: "clamp(32px, 5vw, 42px)",
            fontWeight: 800,
            color: CSS_TEXT,
            margin: "0 0 14px",
            letterSpacing: "-0.03em",
            lineHeight: 1.15,
          }}
        >
          The Hot Corner
        </h1>
        <p
          style={{
            margin: "0 0 18px",
            maxWidth: 620,
            fontSize: 15,
            lineHeight: 1.65,
            color: "#9CA3AF",
          }}
        >
          Forecasts MLB game and player outcomes using machine learning — moneylines, totals, batter props, and four starting-pitcher prop families. Built on Statcast, scheduled lineups, and daily batch inference. Model outputs only; not betting advice.
        </p>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 12px",
            borderRadius: 999,
            border: `1px solid ${COL.border}`,
            background: "rgba(31,41,55,0.6)",
            fontSize: 12,
            fontFamily: FONT_MONO,
            color: "#9CA3AF",
          }}
        >
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: COL.positive, flexShrink: 0 }} />
          v10 — 7 models live · since May 28, 2026
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 16,
          marginBottom: 20,
        }}
      >
        <AboutModelCard
          title="Moneyline model"
          body="Shallow LightGBM trained directly on game outcomes. Six features: SP quality, lineup xwOBA, season run differential, win percentage, park factor, home field. Calibrated against closing market odds."
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <circle cx="8" cy="8" r="4" />
              <circle cx="16" cy="16" r="4" />
            </svg>
          )}
        />
        <AboutModelCard
          title="Totals model"
          body="Ridge regression predicting total runs. Inputs include park factor, offense and defense environment, umpire tendencies, league average, and opposing SP xwOBA. Outputs OVER / UNDER recommendation with edge size."
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <rect x="4" y="3" width="16" height="18" rx="2" />
              <path d="M8 7h8M8 11h3M8 15h8" />
            </svg>
          )}
        />
        <AboutModelCard
          title="Batter props"
          body="Six logistic classifiers per batter: hit, 2+ hits, home run, strikeout, 2+ total bases, walk. Trained on 500k+ Statcast batter-game outcomes (2015–2024) with xwOBA, matchup score, platoon advantage, and 15 input features."
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <circle cx="12" cy="7" r="3" />
              <path d="M6 21v-1a6 6 0 0 1 12 0v1" />
            </svg>
          )}
        />
        <AboutModelCard
          title="Pitcher strikeouts (K)"
          body="Poisson regression for SP strikeout counts — full P(K=0)…P(K=10+) distribution plus over/under probabilities at common lines. On the 2024 holdout, over/under calibration is within ~1.1 percentage points at tested thresholds."
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M12 22c4-3 6-6.5 6-10a6 6 0 0 0-12 0c0 3.5 2 7 6 10z" />
            </svg>
          )}
        />
        <AboutModelCard
          title="Pitcher walks allowed"
          body="Poisson model for walks per SP start. Uses season BB rate, recent BB/9, opposing lineup walk rate, and expected batters faced. Outputs expected BB and over/under probabilities; calibrated independently on 2024 holdout data."
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M4 12h16M4 8h10M4 16h14" />
            </svg>
          )}
        />
        <AboutModelCard
          title="Pitcher hits allowed"
          body="Poisson model for hits allowed per SP start. Inputs include SP xwOBA against, opposing lineup contact quality, park run factor, and expected batters faced. Outputs expected hits and over/under probabilities; calibrated independently on 2024 holdout data."
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <circle cx="12" cy="12" r="9" />
              <path d="M8 12h8" />
            </svg>
          )}
        />
        <AboutModelCard
          title="Pitcher earned runs (ER)"
          body="Poisson model for earned runs per SP start. Uses recent ER form, SP and lineup contact quality, park and umpire run environment, and expected innings. Outputs expected ER and over/under probabilities; calibrated independently on 2024 holdout data."
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M12 3v18M7 8l5-5 5 5M7 16l5 5 5-5" />
            </svg>
          )}
        />
      </div>

      <div
        style={{
          background: COL.card,
          border: `1px solid ${COL.border}`,
          borderLeft: `4px solid ${COL.model}`,
          borderRadius: 12,
          padding: "18px 22px",
          marginBottom: 40,
        }}
      >
        <p style={{ margin: 0, fontSize: 14, lineHeight: 1.65, color: "#9CA3AF" }}>
          All predictions are model outputs, not guaranteed picks. The Hot Corner does not provide betting advice. Probabilities represent statistical estimates based on historical data — they will be wrong. Please wager responsibly.
        </p>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          fontSize: 13,
          color: "#6B7280",
        }}
      >
        <span>© 2026 The Hot Corner · contact@the-hot-corner.com</span>
        <a
          href="mailto:contact@the-hot-corner.com"
          style={{ color: COL.model, fontWeight: 700, textDecoration: "none" }}
        >
          Get in touch
        </a>
      </div>
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
  const isMobile = useIsMobile();
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState(() => [
    {
      role: "assistant",
      content: "Hi! I'm The Hot Corner assistant. Ask me about today's slate, why the model favors a team, or how it's been doing lately. ⚾\n\n*These are model outputs, not guaranteed picks — please wager responsibly.*",
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
        aria-label={open ? "Close Hot Corner assistant" : "Open Hot Corner assistant"}
        onClick={() => setOpen(v => !v)}
        style={{
          position: "fixed",
          bottom: isMobile ? 16 : 20,
          right: isMobile ? 12 : 20,
          zIndex: 2147483000,
          width: isMobile ? 52 : 58,
          height: isMobile ? 52 : 58,
          borderRadius: "50%",
          border: "none",
          background: `linear-gradient(135deg, ${COL.model} 0%, ${COL.accentGold} 100%)`,
          color: "#111827",
          boxShadow: "0 8px 24px rgba(245,158,11,0.4)",
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
            bottom: isMobile ? 78 : 90,
            right: isMobile ? 12 : 20,
            zIndex: 2147482999,
            width: 380,
            maxWidth: "calc(100vw - 32px)",
            height: 560,
            maxHeight: "calc(100vh - 120px)",
            background: COL.card,
            borderRadius: 16,
            boxShadow: "0 24px 60px rgba(0,0,0,0.45)",
            border: `1px solid ${COL.border}`,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            fontFamily: "inherit",
          }}
        >
          <div style={{
            background: `linear-gradient(135deg, ${COL.bg} 0%, ${COL.card} 100%)`,
            padding: "14px 18px",
            color: COL.text,
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
          >
            <img
              src="/the-hot-corner-mark.svg"
              alt=""
              width={34}
              height={34}
              style={{ display: "block", borderRadius: 8, flexShrink: 0 }}
            />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 14, fontWeight: 800, letterSpacing: 0.2, fontFamily: FONT_DISPLAY }}>The Hot Corner</div>
              <div style={{ fontSize: 11, fontWeight: 500, color: COL.textMuted }}>
                {context?.game_id ? "Viewing game" : (context?.date ? `Slate ${context.date}` : "Today")}
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
              background: COL.cardInner,
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
                  background: m.role === "user" ? COL.model : COL.card,
                  color: m.role === "user" ? "#111827" : COL.text,
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
                background: COL.card,
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
                      background: COL.card,
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
              background: COL.card,
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
  const dateInputRef = useRef(null);
  const days = [-3, -2, -1, 0, 1, 2, 3];

  const openDatePicker = () => {
    const el = dateInputRef.current;
    if (!el) return;
    try {
      el.showPicker();
    } catch {
      el.click();
    }
  };

  const chevronBtn = (label, delta) => (
    <button
      type="button"
      aria-label={label}
      onClick={() => onPick(addDaysToYmd(date, delta))}
      style={{
        width: 28,
        height: 28,
        borderRadius: "50%",
        border: "none",
        background: "transparent",
        color: "#6B7280",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "inherit",
        flexShrink: 0,
        padding: 0,
      }}
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
        {delta < 0 ? <path d="M15 18l-6-6 6-6" /> : <path d="M9 18l6-6-6-6" />}
      </svg>
    </button>
  );

  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
        {chevronBtn("Previous day", -1)}
        {days.map((off, idx) => {
          const d = addDaysToYmd(date, off);
          const prevD = idx > 0 ? addDaysToYmd(date, days[idx - 1]) : null;
          const isSel = off === 0;
          return (
            <button
              key={d}
              type="button"
              onClick={() => onPick(d)}
              style={{
                minWidth: 44,
                padding: "4px 6px 6px",
                borderRadius: 8,
                border: "none",
                background: "transparent",
                color: isSel ? COL.model : "#6B7280",
                fontFamily: "inherit",
                cursor: "pointer",
                textAlign: "center",
                flex: "0 0 auto",
              }}
            >
              <div
                style={{
                  fontSize: 9,
                  fontWeight: 700,
                  letterSpacing: "0.08em",
                  color: isSel ? COL.model : "#6B7280",
                }}
              >
                {weekdayShort(d)}
              </div>
              <div
                style={{
                  fontSize: isSel ? 15 : 13,
                  fontWeight: isSel ? 800 : 600,
                  marginTop: 2,
                  ...GAMES_MONO,
                  color: isSel ? COL.model : "#9CA3AF",
                }}
              >
                {dayStripNumberLabel(d, prevD)}
              </div>
            </button>
          );
        })}
        {chevronBtn("Next day", 1)}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
          marginTop: 10,
          paddingTop: 10,
          borderTop: `1px solid ${COL.border}`,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            ref={dateInputRef}
            type="date"
            value={date}
            onChange={(e) => onPick(e.target.value)}
            tabIndex={-1}
            aria-hidden="true"
            style={{
              position: "absolute",
              width: 1,
              height: 1,
              opacity: 0,
              pointerEvents: "none",
              overflow: "hidden",
            }}
          />
          <button
            type="button"
            onClick={openDatePicker}
            style={{
              fontFamily: "inherit",
              fontSize: 12,
              fontWeight: 600,
              padding: "6px 12px",
              borderRadius: 8,
              border: `1px solid ${COL.border}`,
              background: COL.card,
              color: "#9CA3AF",
              cursor: "pointer",
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <rect x="3" y="4" width="18" height="18" rx="2" />
              <path d="M16 2v4M8 2v4M3 10h18" />
            </svg>
            Jump to date
          </button>
          <button
            type="button"
            onClick={onManualRefresh}
            disabled={!!refreshing}
            title="Refresh games"
            aria-label="Refresh games"
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              border: `1px solid ${COL.border}`,
              background: COL.card,
              color: "#9CA3AF",
              cursor: refreshing ? "wait" : "pointer",
              opacity: refreshing ? 0.6 : 1,
              fontFamily: "inherit",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 0,
            }}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M21 12a9 9 0 1 1-2.64-6.36" />
              <path d="M21 3v6h-6" />
            </svg>
          </button>
        </div>
        {statusLine && (
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 7,
              fontSize: 12,
              color: "#6B7280",
              fontFamily: FONT_MONO,
              marginLeft: "auto",
              whiteSpace: "nowrap",
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: COL.positive,
                flexShrink: 0,
              }}
            />
            {statusLine}
          </span>
        )}
      </div>
    </div>
  );
}

/* ── Players tab ─────────────────────────────────────────────────────── */

function normalizePitcherName(name) {
  if (!name) return "";
  return String(name).trim().toLowerCase();
}

function pitcherNameMatches(a, b) {
  const na = normalizePitcherName(a);
  const nb = normalizePitcherName(b);
  if (!na || !nb) return false;
  if (na === nb) return true;
  const lastA = na.split(/\s+/).pop();
  const lastB = nb.split(/\s+/).pop();
  return lastA === lastB && lastA.length > 2;
}

/** Match starter to home/away using game SP names (avoids stale inverted is_home flags). */
function resolveGamePitcher(gamePitchers, game, side) {
  const spName = side === "home" ? game?.home_sp_name : game?.away_sp_name;
  if (spName) {
    const byName = gamePitchers.find((p) => pitcherNameMatches(p.pitcher_name, spName));
    if (byName) return byName;
  }
  const wantHome = side === "home";
  return gamePitchers.find((p) => {
    const isHome = p.is_home === true || p.is_home === "true";
    return wantHome ? isHome : !isHome;
  });
}

function propPct(p) {
  if (p == null || Number.isNaN(Number(p))) return "—";
  return `${Math.round(Number(p) * 100)}%`;
}

function propPctNum(p) {
  if (p == null || Number.isNaN(Number(p))) return null;
  return Number(p) * 100;
}

function hitCellColor(pct) {
  if (pct == null) return COL.text;
  if (pct >= 68) return COL.positive;
  if (pct <= 52) return COL.negative;
  return COL.text;
}

function hits2CellColor(pct) {
  if (pct == null) return COL.text;
  if (pct >= 28) return COL.positive;
  if (pct <= 16) return COL.negative;
  return COL.text;
}

function hrCellColor(pct) {
  if (pct == null) return COL.text;
  if (pct >= 18) return COL.positive;
  if (pct <= 8) return COL.negative;
  return COL.text;
}

function kCellColor(pct) {
  if (pct == null) return COL.text;
  if (pct >= 65) return COL.negative;
  if (pct <= 48) return COL.positive;
  return COL.text;
}

function bbCellColor(pct) {
  if (pct == null) return COL.text;
  if (pct >= 35) return COL.positive;
  if (pct <= 22) return COL.negative;
  return COL.text;
}

function tbCellColor(pct) {
  if (pct == null) return COL.text;
  if (pct >= 42) return COL.positive;
  if (pct <= 26) return COL.negative;
  return COL.text;
}

function ouLineColor(pct) {
  if (pct == null) return COL.text;
  if (pct > 60) return COL.positive;
  if (pct < 40) return COL.negative;
  return COL.text;
}

const BATTER_PROP_LEAGUE_AVG = {
  p_hit: 61,
  p_2plus_hits: 22,
  p_hr: 12,
  p_k: 61,
  p_2plus_bases: 34,
  p_walk: 30,
};

const BATTER_PROP_DEFS = [
  { key: "p_hit", label: "Hit" },
  { key: "p_2plus_hits", label: "2+ Hits" },
  { key: "p_hr", label: "Home Run" },
  { key: "p_k", label: "Strikeout", invertColor: true },
  { key: "p_2plus_bases", label: "2+ Total Bases" },
  { key: "p_walk", label: "Walk" },
];

function propVsLeagueColor(pct, leagueAvg, invertColor = false) {
  if (pct == null || leagueAvg == null) return COL.textMuted;
  const diff = pct - leagueAvg;
  if (invertColor) {
    if (diff >= 5) return COL.negative;
    if (diff <= -5) return COL.positive;
    return COL.model;
  }
  if (diff >= 5) return COL.positive;
  if (diff <= -5) return COL.negative;
  return COL.model;
}

function propVsLeagueContext(pct, leagueAvg) {
  if (pct == null) return `League avg ${leagueAvg}%`;
  const diff = Math.round(pct - leagueAvg);
  const absDiff = Math.abs(diff);
  if (absDiff === 0) return `League avg ${leagueAvg}% · at league avg`;
  if (diff > 0) return `League avg ${leagueAvg}% ↑ ${absDiff} pts above avg`;
  return `League avg ${leagueAvg}% ↓ ${absDiff} pts below avg`;
}

function BatterPropBox({ label, prob, leagueAvg, invertColor = false }) {
  const pct = propPctNum(prob);
  const color = propVsLeagueColor(pct, leagueAvg, invertColor);
  const barW = pct != null ? Math.max(2, Math.min(100, pct)) : 0;

  return (
    <div
      style={{
        padding: "12px 14px",
        borderRadius: 10,
        background: COL.card,
        border: `1px solid ${COL.border}`,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: COL.textSecondary }}>{label}</span>
        <span style={{ fontSize: 15, fontWeight: 900, color, fontVariantNumeric: "tabular-nums" }}>
          {propPct(prob)}
        </span>
      </div>
      <div style={{ height: 5, borderRadius: 3, background: COL.cardInner, overflow: "hidden", marginBottom: 6 }}>
        <div style={{ height: "100%", width: `${barW}%`, background: color, borderRadius: 3, transition: "width 0.2s ease" }} />
      </div>
      <div style={{ fontSize: 10, color: COL.textMuted, lineHeight: 1.4, fontWeight: 600 }}>
        {propVsLeagueContext(pct, leagueAvg)}
      </div>
    </div>
  );
}

function ExpandedBatterCard({ batter, teamTitle }) {
  const platoon = batter.platoon_advantage;
  const hasPlatoonAdv = platoon === 1 || platoon === "1";
  const hasPlatoonDis = platoon === -1 || platoon === "-1";

  return (
    <div>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <div style={{ fontSize: 14, fontWeight: 900, color: COL.text }}>
          {batter.batter_name} · {teamTitle} · #{displayBattingOrder(batter)}
        </div>
        {hasPlatoonAdv && (
          <span style={{
            fontSize: 10,
            fontWeight: 800,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            padding: "3px 10px",
            borderRadius: 999,
            background: "rgba(34,197,94,0.15)",
            color: COL.positive,
            border: "1px solid rgba(34,197,94,0.35)",
          }}>
            Platoon advantage
          </span>
        )}
        {hasPlatoonDis && (
          <span style={{
            fontSize: 10,
            fontWeight: 800,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            padding: "3px 10px",
            borderRadius: 999,
            background: "rgba(239,68,68,0.12)",
            color: COL.negative,
            border: "1px solid rgba(239,68,68,0.35)",
          }}>
            Platoon disadvantage
          </span>
        )}
      </div>
      <div style={{ fontSize: 12, color: COL.textSecondary, marginBottom: 14 }}>
        vs {batter.sp_name || "TBD"}
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
        gap: 10,
        marginBottom: 14,
      }}
      >
        {BATTER_PROP_DEFS.map(({ key, label, invertColor }) => (
          <BatterPropBox
            key={key}
            label={label}
            prob={batter[key]}
            leagueAvg={BATTER_PROP_LEAGUE_AVG[key]}
            invertColor={invertColor}
          />
        ))}
      </div>

      <div style={{ height: 1, background: COL.border, marginBottom: 12 }} />

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
        gap: 12,
      }}
      >
        {[
          {
            label: "Season xwOBA",
            value: batter.batter_xwoba_season != null ? Number(batter.batter_xwoba_season).toFixed(3) : "—",
          },
          {
            label: "30-day hit rate",
            value: batter.batter_hit_rate_30d != null ? Number(batter.batter_hit_rate_30d).toFixed(3) : "—",
          },
          {
            label: "Matchup score",
            value: batter.matchup_score != null ? Number(batter.matchup_score).toFixed(3) : "—",
          },
        ].map(({ label, value }) => (
          <div
            key={label}
            style={{
              padding: "10px 12px",
              borderRadius: 8,
              background: COL.card,
              border: `1px solid ${COL.border}`,
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: 10, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 4 }}>
              {label}
            </div>
            <div style={{ fontSize: 16, fontWeight: 900, color: COL.text, fontVariantNumeric: "tabular-nums", fontFamily: FONT_MONO }}>
              {value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PitcherPropOuRows({ pitcher, rows, barHeight = 4 }) {
  return rows.map(({ label, key }) => {
    const pct = propPctNum(pitcher[key]);
    const w = pct != null ? Math.max(2, Math.min(100, pct)) : 0;
    return (
      <div key={key} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2, fontSize: 10 }}>
        <span style={{ width: 58, color: COL.textSecondary, fontWeight: 700 }}>{label}</span>
        <div style={{ flex: 1, height: barHeight, borderRadius: 2, background: COL.cardInner, overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${w}%`, background: ouLineColor(pct) === COL.positive ? COL.positive : ouLineColor(pct) === COL.negative ? COL.negative : COL.model, borderRadius: 2 }} />
        </div>
        <span style={{ width: 32, textAlign: "right", fontWeight: 800, color: ouLineColor(pct), fontVariantNumeric: "tabular-nums" }}>
          {propPct(pitcher[key])}
        </span>
      </div>
    );
  });
}

const PITCHER_DEFAULT_EXPECTED_IP = 5.5;

function pitcherExpectedIp(pitcher) {
  const ip = Number(pitcher?.expected_ip);
  return Number.isFinite(ip) && ip > 0 ? ip : PITCHER_DEFAULT_EXPECTED_IP;
}

function pitcherPerNine(count, expectedIp) {
  const total = Number(count);
  const ip = Number(expectedIp);
  if (!Number.isFinite(total) || !Number.isFinite(ip) || ip <= 0) return null;
  return (total / ip) * 9;
}

function pitcherLineThreshold(row) {
  const m = String(row?.label || "").match(/([\d.]+)/);
  return m ? parseFloat(m[1]) : null;
}

function pickPickemLineRow(rows, pitcher) {
  let best = null;
  for (const row of rows) {
    const pct = propPctNum(pitcher[row.key]);
    if (pct == null) continue;
    const dist = Math.abs(pct - 50);
    if (!best || dist < best.dist) best = { row, dist };
  }
  return best?.row || rows[Math.floor(rows.length / 2)] || rows[0] || null;
}

function pickThreeLineRows(rows, lambdaValue, pitcher) {
  const lam = Number(lambdaValue);
  const enriched = rows
    .map((row) => ({ row, thresh: pitcherLineThreshold(row), pct: propPctNum(pitcher[row.key]) }))
    .filter((x) => x.thresh != null && x.pct != null);
  if (!enriched.length) return rows.slice(0, 3);

  const below = enriched.filter((x) => x.thresh < lam).sort((a, b) => b.thresh - a.thresh)[0]?.row;
  const above = enriched.filter((x) => x.thresh >= lam).sort((a, b) => a.thresh - b.thresh)[0]?.row;
  const pickem = pickPickemLineRow(rows, pitcher);

  const picked = [];
  const add = (row) => {
    if (row && !picked.some((p) => p.key === row.key)) picked.push(row);
  };
  add(below);
  add(pickem);
  add(above);
  if (picked.length < 3) {
    const byDist = [...enriched].sort((a, b) => {
      const da = Number.isFinite(lam) ? Math.abs(a.thresh - lam) : Math.abs(a.pct - 50);
      const db = Number.isFinite(lam) ? Math.abs(b.thresh - lam) : Math.abs(b.pct - 50);
      return da - db;
    });
    for (const { row } of byDist) {
      add(row);
      if (picked.length >= 3) break;
    }
  }
  return picked.sort((a, b) => pitcherLineThreshold(a) - pitcherLineThreshold(b));
}

function parsePitcherIp(ip) {
  if (ip == null || ip === "") return null;
  if (typeof ip === "number" && Number.isFinite(ip)) return ip;
  const s = String(ip).trim();
  if (!s) return null;
  if (s.includes(".")) {
    const [whole, frac] = s.split(".");
    const outs = Number(frac) || 0;
    return Number(whole) + outs / 3;
  }
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function formatPitcherStatActual(value, label) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  if (label === "IP") {
    const whole = Math.floor(n);
    const outs = Math.round((n - whole) * 3);
    return outs > 0 ? `${whole}.${outs}` : String(whole);
  }
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function pitcherActualDirectionColor(expected, actual, tolerance = 0.5) {
  const exp = Number(expected);
  const act = Number(actual);
  if (!Number.isFinite(exp) || !Number.isFinite(act)) return COL.text;
  const diff = act - exp;
  if (Math.abs(diff) < tolerance) return "#9CA3AF";
  return diff > 0 ? "#F59E0B" : "#60A5FA";
}

function PitcherSummaryStatBox({ label, expected, actual, perNine, emphasize = false }) {
  const expNum = Number(expected);
  const expStr = Number.isFinite(expNum) ? expNum.toFixed(1) : "—";
  const actNum = actual != null ? Number(actual) : null;
  const hasActual = actNum != null && Number.isFinite(actNum);
  const tolerance = label === "IP" ? 0.34 : 0.5;
  const perNineTip = perNine != null && Number.isFinite(Number(perNine))
    ? `${Number(perNine).toFixed(1)} per 9 IP`
    : undefined;

  return (
    <div
      title={perNineTip}
      style={{
        flex: emphasize ? "1.15 1 0" : "1 1 0",
        minWidth: emphasize ? 80 : 72,
        minHeight: hasActual ? 108 : 76,
        padding: "10px 12px",
        borderRadius: 10,
        background: emphasize ? "rgba(255,255,255,0.06)" : COL.cardInner,
        border: `1px solid ${COL.border}`,
        boxShadow: emphasize ? "inset 0 0 0 1px rgba(255,255,255,0.04)" : "none",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{
        fontSize: 10,
        fontWeight: 800,
        color: COL.textMuted,
        letterSpacing: "0.1em",
        textTransform: "uppercase",
        marginBottom: 4,
      }}
      >
        {label}
      </div>
      <div style={{
        fontSize: 9,
        fontWeight: 700,
        color: "#6B7280",
        letterSpacing: "0.06em",
        marginBottom: 2,
      }}
      >
        projected
      </div>
      <div style={{
        fontSize: emphasize ? 24 : 22,
        fontWeight: 900,
        color: COL.text,
        fontVariantNumeric: "tabular-nums",
        lineHeight: 1.05,
      }}
      >
        {expStr}
      </div>
      {hasActual ? (
        <>
          <div style={{ height: 1, background: COL.border, margin: "8px 0 6px", opacity: 0.7 }} />
          <div style={{
            fontSize: 9,
            fontWeight: 700,
            color: "#6B7280",
            letterSpacing: "0.06em",
            marginBottom: 2,
          }}
          >
            actual
          </div>
          <div style={{
            fontSize: emphasize ? 20 : 18,
            fontWeight: 900,
            color: pitcherActualDirectionColor(expNum, actNum, tolerance),
            fontVariantNumeric: "tabular-nums",
            lineHeight: 1.05,
          }}
          >
            {formatPitcherStatActual(actNum, label)}
          </div>
        </>
      ) : null}
    </div>
  );
}

function PitcherPropLineBlock({
  propId,
  headline,
  rows,
  lambdaValue,
  pitcher,
  accentText,
  expanded,
  onToggle,
}) {
  if (lambdaValue == null && !rows.some(({ key }) => pitcher[key] != null)) return null;
  const pickemRow = pickPickemLineRow(rows, pitcher);
  const displayRows = expanded ? pickThreeLineRows(rows, lambdaValue, pitcher) : (pickemRow ? [pickemRow] : []);
  if (!displayRows.length) return null;

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 12, fontWeight: 800, color: COL.text, marginBottom: 4, fontVariantNumeric: "tabular-nums" }}>
        {headline}
      </div>
      <PitcherPropOuRows pitcher={pitcher} rows={displayRows} />
      <button
        type="button"
        onClick={() => onToggle(propId)}
        style={{
          marginTop: 2,
          padding: 0,
          border: "none",
          background: "none",
          cursor: "pointer",
          fontSize: 10,
          fontWeight: 700,
          color: accentText,
          letterSpacing: "0.02em",
        }}
      >
        {expanded ? "Show fewer lines" : "Show all lines"}
      </button>
    </div>
  );
}

function PitcherPropCard({ pitcher, teamName, theme, oppSpName, liveActuals = null }) {
  const [expandedProps, setExpandedProps] = useState({});
  if (!pitcher) {
    return (
      <div style={{ flex: 1, minWidth: 0, borderRadius: 12, border: `1px solid ${COL.border}`, background: COL.card, padding: 20, color: COL.textMuted, fontSize: 13 }}>
        No pitcher data
      </div>
    );
  }
  const th = theme || { primary: COL.model, soft: COL.cardInner, stroke: COL.border };
  const accentText = teamAccentText(th);
  const expectedIp = pitcherExpectedIp(pitcher);
  const kPer9 = pitcherPerNine(pitcher.lambda_k, expectedIp);
  const bbPer9 = pitcherPerNine(pitcher.lambda_walks, expectedIp);
  const hitsPer9 = pitcherPerNine(pitcher.lambda_hits, expectedIp);
  const erPer9 = pitcherPerNine(pitcher.lambda_er, expectedIp);
  const toggleProp = (id) => setExpandedProps((cur) => ({ ...cur, [id]: !cur[id] }));

  const kRows = [
    { label: "Over 3.5", key: "p_over_3_5" },
    { label: "Over 4.5", key: "p_over_4_5" },
    { label: "Over 5.5", key: "p_over_5_5" },
    { label: "Over 6.5", key: "p_over_6_5" },
    { label: "Over 7.5", key: "p_over_7_5" },
  ];
  const walkRows = [
    { label: "Over 0.5", key: "p_walks_over_0_5" },
    { label: "Over 1.5", key: "p_walks_over_1_5" },
    { label: "Over 2.5", key: "p_walks_over_2_5" },
    { label: "Over 3.5", key: "p_walks_over_3_5" },
    { label: "Over 4.5", key: "p_walks_over_4_5" },
  ];
  const hitRows = [
    { label: "Over 3.5", key: "p_hits_over_3_5" },
    { label: "Over 4.5", key: "p_hits_over_4_5" },
    { label: "Over 5.5", key: "p_hits_over_5_5" },
    { label: "Over 6.5", key: "p_hits_over_6_5" },
    { label: "Over 7.5", key: "p_hits_over_7_5" },
  ];
  const erRows = [
    { label: "Over 1.5", key: "p_er_over_1_5" },
    { label: "Over 2.5", key: "p_er_over_2_5" },
    { label: "Over 3.5", key: "p_er_over_3_5" },
    { label: "Over 4.5", key: "p_er_over_4_5" },
    { label: "Over 5.5", key: "p_er_over_5_5" },
  ];

  const fmtLambda = (v) => (v != null && Number.isFinite(Number(v)) ? Number(v).toFixed(1) : "—");

  return (
    <div style={{
      flex: 1,
      minWidth: 0,
      borderRadius: 12,
      border: `1px solid ${COL.border}`,
      background: COL.card,
      boxShadow: "0 4px 16px rgba(15,23,42,0.07)",
      overflow: "hidden",
    }}
    >
      <div style={{ height: 2, background: th.primary }} />
      <div style={{ padding: "12px 14px", background: "#1F2937", borderBottom: `1px solid ${COL.border}` }}>
        <div style={{ fontSize: 17, fontWeight: 900, color: COL.text, letterSpacing: "-0.02em" }}>
          {pitcher.pitcher_name || "SP TBD"}
        </div>
        <div style={{ fontSize: 12, color: COL.text, marginTop: 3 }}>{teamName}</div>
        {oppSpName && (
          <div style={{ fontSize: 10, color: COL.textMuted, marginTop: 4 }}>vs {oppSpName}</div>
        )}
      </div>
      <div style={{ padding: "12px 14px" }}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
          <PitcherSummaryStatBox
            label="IP"
            expected={expectedIp}
            actual={liveActuals?.ip}
            emphasize
          />
          <PitcherSummaryStatBox
            label="K"
            expected={pitcher.lambda_k}
            actual={liveActuals?.k}
            perNine={kPer9}
          />
          <PitcherSummaryStatBox
            label="BB"
            expected={pitcher.lambda_walks}
            actual={liveActuals?.bb}
            perNine={bbPer9}
          />
          <PitcherSummaryStatBox
            label="H"
            expected={pitcher.lambda_hits}
            actual={liveActuals?.hits}
            perNine={hitsPer9}
          />
          <PitcherSummaryStatBox
            label="ER"
            expected={pitcher.lambda_er}
            actual={liveActuals?.er}
            perNine={erPer9}
          />
        </div>
        <PitcherPropLineBlock
          propId="k"
          headline={`Expected K ${fmtLambda(pitcher.lambda_k)}`}
          rows={kRows}
          lambdaValue={pitcher.lambda_k}
          pitcher={pitcher}
          accentText={accentText}
          expanded={!!expandedProps.k}
          onToggle={toggleProp}
        />
        <PitcherPropLineBlock
          propId="bb"
          headline={`Expected BB ${fmtLambda(pitcher.lambda_walks)}`}
          rows={walkRows}
          lambdaValue={pitcher.lambda_walks}
          pitcher={pitcher}
          accentText={accentText}
          expanded={!!expandedProps.bb}
          onToggle={toggleProp}
        />
        <PitcherPropLineBlock
          propId="hits"
          headline={`Expected Hits ${fmtLambda(pitcher.lambda_hits)}`}
          rows={hitRows}
          lambdaValue={pitcher.lambda_hits}
          pitcher={pitcher}
          accentText={accentText}
          expanded={!!expandedProps.hits}
          onToggle={toggleProp}
        />
        <PitcherPropLineBlock
          propId="er"
          headline={`Expected ER ${fmtLambda(pitcher.lambda_er)}`}
          rows={erRows}
          lambdaValue={pitcher.lambda_er}
          pitcher={pitcher}
          accentText={accentText}
          expanded={!!expandedProps.er}
          onToggle={toggleProp}
        />
      </div>
    </div>
  );
}

function formatScrollerDateLabel(dateStr) {
  if (!dateStr) return null;
  const d = new Date(`${dateStr}T12:00:00`);
  return {
    month: d.toLocaleDateString("en-US", { month: "short", timeZone: "America/Los_Angeles" }).toUpperCase(),
    day: d.toLocaleDateString("en-US", { day: "numeric", timeZone: "America/Los_Angeles" }),
  };
}

function playersScrollerInningLabel(liveRow) {
  if (!liveRow?.inningState || liveRow?.currentInning == null) return null;
  const s = liveRow.inningState.toUpperCase();
  let prefix = "MID";
  if (s.includes("TOP")) prefix = "TOP";
  else if (s.includes("BOT")) prefix = "BOT";
  else if (s.includes("MID")) prefix = "MID";
  return `${prefix} ${liveRow.currentInning}`;
}

function playersScrollerStatus(g, liveRow) {
  const detailed = liveRow?.status ?? g.status ?? "";
  const abstract = liveRow?.abstractGameState ?? null;
  const coded = liveRow?.codedGameState ?? null;
  const gameFinished = isMlbGameFinished(detailed, abstract, coded);
  const gamePostponed = isPostponedOrCancelled(detailed, abstract);
  const gameLive = !gameFinished && !gamePostponed && (isLiveStatus(detailed) || isLiveStatus(g.status));

  if (gamePostponed) return { label: "PPD", color: COL.textMuted, live: false };
  if (gameFinished) return { label: "FINAL", color: "#9CA3AF", live: false };
  if (gameLive) {
    const inn = playersScrollerInningLabel(liveRow);
    return { label: inn || "LIVE", color: COL.negative, live: true };
  }
  if (detailed.toLowerCase().includes("warmup")) {
    return { label: "WARMUP", color: "#9CA3AF", live: false };
  }
  if (detailed.toLowerCase().includes("delayed")) {
    return { label: "DELAY", color: COL.model, live: false };
  }
  const parts = formatFirstPitchParts(g.first_pitch_utc);
  if (parts?.et) {
    return { label: parts.et.toUpperCase(), color: "#9CA3AF", live: false };
  }
  return { label: "TBD", color: COL.textMuted, live: false };
}

function scrollerFinalScores(g, liveRow) {
  const detailed = liveRow?.status ?? g.status ?? "";
  const abstract = liveRow?.abstractGameState ?? null;
  const coded = liveRow?.codedGameState ?? null;
  if (!isMlbGameFinished(detailed, abstract, coded)) return null;
  const away = pickFinishedGameRuns(liveRow?.awayRuns, g.away_runs);
  const home = pickFinishedGameRuns(liveRow?.homeRuns, g.home_runs);
  if (away == null || home == null) return null;
  return { away: Number(away), home: Number(home) };
}

function gradeModelEdge(edge, games, live) {
  if (!edge?.game_id || !games?.length) return null;
  const game = games.find((g) => String(g.game_id) === String(edge.game_id));
  if (!game) return null;
  const liveRow = live?.[game.game_id];
  const detailed = liveRow?.status ?? game.status ?? "";
  const abstract = liveRow?.abstractGameState ?? null;
  const coded = liveRow?.codedGameState ?? null;
  if (!isMlbGameFinished(detailed, abstract, coded)) return null;

  const awayRuns = pickFinishedGameRuns(liveRow?.awayRuns, game.away_runs);
  const homeRuns = pickFinishedGameRuns(liveRow?.homeRuns, game.home_runs);
  if (awayRuns == null || homeRuns == null) return null;
  if (Number(awayRuns) === Number(homeRuns)) return "push";

  const awayWon = Number(awayRuns) > Number(homeRuns);
  const edgeType = (edge.type || "").toLowerCase();

  if (edgeType === "total") {
    const totalRuns = Number(awayRuns) + Number(homeRuns);
    const line = edge.comparison_value_num != null ? Number(edge.comparison_value_num) : null;
    if (line == null || !Number.isFinite(line)) return null;
    const side = (edge.direction || "over").toLowerCase();
    if (Math.abs(totalRuns - line) < 0.01) return "push";
    if (side === "over") return totalRuns > line ? "hit" : "miss";
    return totalRuns < line ? "hit" : "miss";
  }

  const title = edge.title || edge.pick_description || "";
  const pickTeam = title.includes(" — ") ? title.split(" — ").pop().trim() : null;
  if (!pickTeam) return null;

  if (lineupBatterMatches(game.home_team, pickTeam) || pickTeam === game.home_team) {
    return awayWon ? "miss" : "hit";
  }
  if (lineupBatterMatches(game.away_team, pickTeam) || pickTeam === game.away_team) {
    return awayWon ? "hit" : "miss";
  }
  return null;
}

function EdgeGradeMark({ result, size = 20 }) {
  if (!result) return null;
  if (result === "push") {
    return (
      <span
        aria-label="Push"
        title="Push"
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: size,
          height: size,
          borderRadius: "50%",
          background: "rgba(107,114,128,0.18)",
          color: COL.textMuted,
          border: "1.5px solid rgba(107,114,128,0.45)",
          fontSize: size * 0.55,
          fontWeight: 900,
          lineHeight: 1,
          flexShrink: 0,
        }}
      >
        —
      </span>
    );
  }
  const isHit = result === "hit";
  return (
    <span
      aria-label={isHit ? "Hit" : "Miss"}
      title={isHit ? "Prediction hit" : "Prediction missed"}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: size,
        height: size,
        borderRadius: "50%",
        background: isHit ? "rgba(34,197,94,0.18)" : "rgba(239,68,68,0.18)",
        color: isHit ? COL.positive : COL.negative,
        border: `1.5px solid ${isHit ? "rgba(22,163,74,0.45)" : "rgba(220,38,38,0.45)"}`,
        fontSize: size * 0.55,
        fontWeight: 900,
        lineHeight: 1,
        flexShrink: 0,
      }}
    >
      {isHit ? "✓" : "✕"}
    </span>
  );
}

function PlayersGameScroller({ games, live, selectedGameId, onSelectGame, scheduleDate }) {
  const dateLabel = formatScrollerDateLabel(scheduleDate);

  useEffect(() => {
    const id = "mlb-live-pulse-kf";
    if (typeof document === "undefined" || document.getElementById(id)) return;
    const s = document.createElement("style");
    s.id = id;
    s.textContent = `@keyframes mlbLivePulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }`;
    document.head.appendChild(s);
  }, []);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "stretch",
        marginBottom: 20,
        border: `1px solid ${COL.border}`,
        borderRadius: 10,
        overflow: "hidden",
        background: COL.card,
        boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
      }}
    >
      {dateLabel && (
        <div
          style={{
            flexShrink: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "10px 12px",
            borderRight: `1px solid ${COL.border}`,
            background: "#0D1420",
            minWidth: 52,
          }}
        >
          <span style={{ fontSize: 10, fontWeight: 800, color: COL.model, letterSpacing: "0.08em", lineHeight: 1.2 }}>
            {dateLabel.month}
          </span>
          <span style={{ fontSize: 20, fontWeight: 900, color: COL.text, lineHeight: 1.1, marginTop: 2 }}>
            {dateLabel.day}
          </span>
        </div>
      )}
      <div
        style={{
          display: "flex",
          overflowX: "auto",
          flex: 1,
          scrollbarWidth: "thin",
        }}
      >
        {games.map((g, i) => {
          const active = String(g.game_id) === String(selectedGameId);
          const liveRow = live?.[g.game_id];
          const status = playersScrollerStatus(g, liveRow);
          const abA = teamAbbr(g.away_team);
          const abH = teamAbbr(g.home_team);
          const finalScores = scrollerFinalScores(g, liveRow);

          const teamRow = (team, abbr, score, isWinner) => (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, width: "100%" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
                <Logo team={team} size={20} />
                <span style={{ fontSize: 13, fontWeight: 800, color: COL.text, letterSpacing: "0.04em" }}>{abbr}</span>
              </div>
              {score != null && (
                <span style={{
                  fontFamily: FONT_MONO,
                  fontSize: 13,
                  fontWeight: 800,
                  fontVariantNumeric: "tabular-nums",
                  color: isWinner ? COL.positive : COL.textMuted,
                  flexShrink: 0,
                }}>
                  {score}
                </span>
              )}
            </div>
          );

          return (
            <button
              key={g.game_id}
              type="button"
              onClick={() => onSelectGame(g.game_id)}
              title={`${g.away_team} @ ${g.home_team}`}
              style={{
                flexShrink: 0,
                fontFamily: "inherit",
                cursor: "pointer",
                padding: "10px 14px",
                minWidth: finalScores ? 118 : 108,
                border: "none",
                borderRight: i < games.length - 1 ? `1px solid ${COL.border}` : "none",
                borderBottom: active ? `3px solid ${COL.model}` : "3px solid transparent",
                background: active ? "rgba(245,158,11,0.08)" : COL.card,
                transition: "background 0.12s ease, border-color 0.12s ease",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  fontWeight: 800,
                  color: status.color,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  textAlign: "center",
                  marginBottom: 8,
                  minHeight: 14,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 4,
                }}
              >
                {status.live && (
                  <span
                    style={{
                      width: 5,
                      height: 5,
                      borderRadius: "50%",
                      background: COL.negative,
                      flexShrink: 0,
                      animation: "mlbLivePulse 1.6s ease-in-out infinite",
                    }}
                  />
                )}
                {status.label}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                {teamRow(
                  g.away_team,
                  abA,
                  finalScores?.away,
                  finalScores && finalScores.away > finalScores.home,
                )}
                {teamRow(
                  g.home_team,
                  abH,
                  finalScores?.home,
                  finalScores && finalScores.home > finalScores.away,
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function PlayersTab({ games, batters, pitchers, loading, selectedGameId, onSelectGame, live, scheduleDate }) {
  const isMobile = useIsMobile();
  const [expandedBatterId, setExpandedBatterId] = useState(null);

  const activeGameId = selectedGameId ?? (games[0]?.game_id ?? null);
  const activeGame = games.find((g) => String(g.game_id) === String(activeGameId));

  const { home: homeBatters, away: awayBatters, all: gameBatters } = useMemo(
    () => prepareGameBatterLineups(batters, activeGameId),
    [batters, activeGameId],
  );
  const gamePitchers = useMemo(
    () => pitchers.filter((p) => String(p.game_id) === String(activeGameId)),
    [pitchers, activeGameId],
  );

  const homePitcher = resolveGamePitcher(gamePitchers, activeGame, "home");
  const awayPitcher = resolveGamePitcher(gamePitchers, activeGame, "away");

  const homeTeam = activeGame?.home_team || homePitcher?.home_team || homeBatters[0]?.home_team || "Home";
  const awayTeam = activeGame?.away_team || awayPitcher?.away_team || awayBatters[0]?.away_team || "Away";
  const themeHome = getTeamTheme(homeTeam);
  const themeAway = getTeamTheme(awayTeam);

  const liveRow = live?.[activeGameId];
  const detailed = liveRow?.status ?? activeGame?.status ?? "";
  const abstract = liveRow?.abstractGameState ?? null;
  const coded = liveRow?.codedGameState ?? null;
  const gameFinished = isMlbGameFinished(detailed, abstract, coded);
  const gamePostponed = isPostponedOrCancelled(detailed, abstract);
  const gameLive = !gameFinished && !gamePostponed && (isLiveStatus(detailed) || isLiveStatus(activeGame?.status));
  const showBoxScore = gameLive || gameFinished;
  const gameFeed = useMlbGameFeed(activeGameId, showBoxScore, gameLive);

  const homePitcherLive = showBoxScore
    ? findFeedPitcherActuals(gameFeed?.homePitchers, homePitcher?.pitcher_name, homePitcher?.pitcher_id)
    : null;
  const awayPitcherLive = showBoxScore
    ? findFeedPitcherActuals(gameFeed?.awayPitchers, awayPitcher?.pitcher_name, awayPitcher?.pitcher_id)
    : null;

  if (loading) {
    return <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "3rem" }}>Loading player props…</p>;
  }

  if (!games.length) {
    return (
      <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "3rem" }}>
        No games scheduled for this date.
      </p>
    );
  }

  if (!gameBatters.length && !gamePitchers.length) {
    return (
      <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "3rem" }}>
        Player prop predictions are not available yet for this date. Run morning inference or early roster inference.
      </p>
    );
  }

  return (
    <div style={pageShellStyle(1200, { top: 12, bottom: 32, horizontal: isMobile ? -12 : 0 })}>
      <PlayersGameScroller
        games={games}
        live={live}
        selectedGameId={activeGameId}
        onSelectGame={(id) => { onSelectGame(id); setExpandedBatterId(null); }}
        scheduleDate={scheduleDate}
      />

      <div style={{ fontSize: 11, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginBottom: 10 }}>STARTING PITCHERS</div>
      <div style={{ display: "flex", flexDirection: isMobile ? "column" : "row", gap: 16, flexWrap: isMobile ? "nowrap" : "wrap", marginBottom: 24 }}>
        <div style={{ flex: isMobile ? "none" : 1, minWidth: 0, width: isMobile ? "100%" : undefined }}>
          <PitcherPropCard pitcher={homePitcher} teamName={homeTeam} theme={themeHome} oppSpName={awayPitcher?.pitcher_name} liveActuals={homePitcherLive} />
        </div>
        <div style={{ flex: isMobile ? "none" : 1, minWidth: 0, width: isMobile ? "100%" : undefined }}>
          <PitcherPropCard pitcher={awayPitcher} teamName={awayTeam} theme={themeAway} oppSpName={homePitcher?.pitcher_name} liveActuals={awayPitcherLive} />
        </div>
      </div>

      <div style={{ fontSize: 11, fontWeight: 800, color: COL.textMuted, letterSpacing: "0.08em", marginBottom: 10 }}>BATTING LINEUPS</div>
      <BatterPropLineupPair
        homeTeam={homeTeam}
        awayTeam={awayTeam}
        homeBatters={homeBatters}
        awayBatters={awayBatters}
        homePitcherName={awayPitcher?.pitcher_name}
        awayPitcherName={homePitcher?.pitcher_name}
        themeHome={themeHome}
        themeAway={themeAway}
        expandedBatterId={expandedBatterId}
        onToggleBatter={(id) => setExpandedBatterId((cur) => (cur === id ? null : id))}
        findGameLine={findFeedBatterGameLine}
        feedHomeBatters={gameFeed?.homeBatters}
        feedAwayBatters={gameFeed?.awayBatters}
        showBoxScore={showBoxScore}
        gameFinished={gameFinished}
        renderExpanded={(batter, teamTitle) => (
          <ExpandedBatterCard batter={batter} teamTitle={teamTitle} />
        )}
      />
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
  const [menuOpen, setMenuOpen] = useState(false);
  const isMobile = useIsMobile();
  const [accuracyRefreshKey, setAccuracyRefreshKey] = useState(0);
  const [playersBatters, setPlayersBatters] = useState([]);
  const [playersPitchers, setPlayersPitchers] = useState([]);
  const [playersLoading, setPlayersLoading] = useState(false);
  const [selectedPlayerGameId, setSelectedPlayerGameId] = useState(null);
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

  const enrich = useGameEnrichment(gameIds, seasonYear, boxRefresh, date);
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

  const fetchPlayers = useCallback((force = false) => {
    const cached = playersCache.get(date);
    if (cached && !force) {
      if (cached.games?.length) setGames(cached.games);
      setPlayersBatters(filterBattersForDisplay(cached.batters || []));
      setPlayersPitchers(cached.pitchers || []);
      setPlayersLoading(false);
      return;
    }
    setPlayersLoading(true);
    fetchPlayersData(date, { force })
      .then((d) => {
        if (d.games?.length) setGames(d.games);
        setPlayersBatters(filterBattersForDisplay(d.batters || []));
        setPlayersPitchers(d.pitchers || []);
        setPlayersLoading(false);
        setLastUpdated(new Date());
      })
      .catch(() => setPlayersLoading(false));
  }, [date]);

  // eslint-disable-next-line react-hooks/set-state-in-effect -- reset loading then async fetch
  useEffect(() => { setLoading(true); fetchGames(); }, [date, fetchGames]);

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { setSelectedPlayerGameId(null); }, [date]);

  useEffect(() => {
    if (mainTab === "players" || detailGameId) fetchPlayers();
  }, [mainTab, detailGameId, date, fetchPlayers]);

  useEffect(() => {
    if (mainTab !== "players" || !sortedGames.length || selectedPlayerGameId != null) return;
    setSelectedPlayerGameId(sortedGames[0].game_id);
  }, [mainTab, sortedGames, selectedPlayerGameId]);

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

  const navigateToTab = useCallback((tabId) => {
    if (detailGameId) closeGameDetail();
    setMainTab(tabId);
    setMenuOpen(false);
  }, [detailGameId, closeGameDetail]);

  const goHome = useCallback(() => {
    navigateToTab("games");
  }, [navigateToTab]);

  useEffect(() => {
    const onHash = () => setDetailGameId(readHashGameId());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    let title = "The Hot Corner — MLB Predictions";
    if (detailGameId) {
      const label = detailGame
        ? `${detailGame.away_team} @ ${detailGame.home_team}`
        : `Game ${detailGameId}`;
      title = `The Hot Corner — ${label}`;
    } else if (mainTab === "edges") {
      title = "The Hot Corner — Top Edges";
    } else if (mainTab === "trends") {
      title = "The Hot Corner — Trends";
    } else if (mainTab === "standings") {
      title = "The Hot Corner — Standings";
    } else if (mainTab === "transactions") {
      title = "The Hot Corner — Transactions";
    } else if (mainTab === "accuracy") {
      title = "The Hot Corner — Model Performance";
    } else if (mainTab === "players") {
      title = "The Hot Corner — Players";
    } else if (mainTab === "about") {
      title = "The Hot Corner — About";
    }
    document.title = title;
  }, [mainTab, detailGameId, detailGame]);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.gtag !== "function") return;
    let title = "The Hot Corner — MLB Predictions";
    let path = "/";
    if (detailGameId) {
      const label = detailGame
        ? `${detailGame.away_team} @ ${detailGame.home_team}`
        : `Game ${detailGameId}`;
      title = `The Hot Corner — ${label}`;
      path = `/game/${detailGameId}`;
    } else if (mainTab === "edges") {
      title = "The Hot Corner — Top Edges";
      path = "/edges";
    } else if (mainTab === "trends") {
      title = "The Hot Corner — Trends";
      path = "/trends";
    } else if (mainTab === "standings") {
      title = "The Hot Corner — Standings";
      path = "/standings";
    } else if (mainTab === "transactions") {
      title = "The Hot Corner — Transactions";
      path = "/transactions";
    } else if (mainTab === "accuracy") {
      title = "The Hot Corner — Model Performance";
      path = "/accuracy";
    } else if (mainTab === "players") {
      title = "The Hot Corner — Players";
      path = "/players";
    } else if (mainTab === "about") {
      title = "The Hot Corner — About";
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

  useEffect(() => {
    if (!isMobile) setMenuOpen(false);
  }, [isMobile]);

  useEffect(() => {
    setMenuOpen(false);
  }, [detailGameId]);

  const headerDateLabel = new Date().toLocaleDateString("en-US", {
    timeZone: "America/Los_Angeles",
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  return (
    <div style={{ background: COL.bg, minHeight: "100vh", width: "100%", fontFamily: FONT_BODY, color: COL.text }}>
      <style>{`
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${COL.border}; border-radius: 2px; }
        .hc-nav-tab:hover { color: #F9FAFB !important; }
        .hc-logo-btn:hover { opacity: 0.88; }
      `}</style>

      <div
        style={{
          position: "sticky",
          top: 0,
          zIndex: 200,
          background: "#111827",
          borderBottom: `1px solid ${COL.border}`,
        }}
      >
        <div
          style={{
            height: 56,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
            padding: "0 clamp(16px, 3vw, 28px)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 16, minWidth: 0, flex: 1 }}>
            <button
              type="button"
              className="hc-logo-btn"
              onClick={goHome}
              aria-label="The Hot Corner — home"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 10,
                border: "none",
                background: "transparent",
                padding: 0,
                cursor: "pointer",
                flexShrink: 0,
              }}
            >
              <img
                src="/the-hot-corner-mark.svg"
                alt=""
                width={32}
                height={32}
                style={{ display: "block", borderRadius: 6 }}
              />
              <span
                style={{
                  fontFamily: FONT_DISPLAY,
                  fontSize: 22,
                  letterSpacing: "0.06em",
                  color: COL.model,
                  whiteSpace: "nowrap",
                }}
              >
                The Hot Corner
              </span>
            </button>
            {isMobile ? (
              <button
                type="button"
                aria-label={menuOpen ? "Close menu" : "Menu"}
                onClick={() => setMenuOpen((o) => !o)}
                style={{
                  marginLeft: "auto",
                  display: "inline-flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  alignItems: "center",
                  gap: menuOpen ? 0 : 5,
                  width: 40,
                  height: 40,
                  border: "none",
                  background: "transparent",
                  cursor: "pointer",
                  padding: 8,
                  flexShrink: 0,
                }}
              >
                {menuOpen ? (
                  <span style={{ fontSize: 22, lineHeight: 1, color: CSS_TEXT, fontWeight: 300 }}>×</span>
                ) : (
                  <>
                    <span style={{ width: 20, height: 2, background: CSS_TEXT, borderRadius: 2 }} />
                    <span style={{ width: 20, height: 2, background: CSS_TEXT, borderRadius: 2 }} />
                    <span style={{ width: 20, height: 2, background: CSS_TEXT, borderRadius: 2 }} />
                  </>
                )}
              </button>
            ) : (
              <>
                <div style={{ width: 1, height: 24, background: "#374151", flexShrink: 0 }} />
                <nav style={{ display: "flex", alignItems: "center", gap: 20, minWidth: 0, overflowX: "auto" }}>
                  {NAV_TABS.map((t) => {
                    const active = detailGameId ? t.id === "games" : mainTab === t.id;
                    return (
                      <button
                        key={t.id}
                        type="button"
                        className={active ? undefined : "hc-nav-tab"}
                        onMouseEnter={() => {
                          if (t.id === "trends") prefetchTrends();
                          if (t.id === "edges") prefetchEdges(date);
                          if (t.id === "players") prefetchPlayers(date);
                          if (t.id === "standings") prefetchStandings(date);
                          if (t.id === "transactions") prefetchTransactions(date);
                        }}
                        onClick={() => navigateToTab(t.id)}
                        style={{
                          fontFamily: "inherit",
                          fontSize: 14,
                          fontWeight: 600,
                          padding: 0,
                          border: "none",
                          background: "transparent",
                          cursor: "pointer",
                          color: active ? CSS_TEXT : "#9CA3AF",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 6,
                          whiteSpace: "nowrap",
                          flexShrink: 0,
                          transition: "color 0.15s",
                        }}
                      >
                        {t.label}
                        {active && (
                          <span
                            style={{
                              width: 4,
                              height: 4,
                              borderRadius: "50%",
                              background: COL.model,
                              flexShrink: 0,
                            }}
                          />
                        )}
                      </button>
                    );
                  })}
                </nav>
              </>
            )}
          </div>
          <div
            style={{
              fontFamily: FONT_MONO,
              fontSize: isMobile ? 11 : 13,
              color: "#6B7280",
              whiteSpace: "nowrap",
              flexShrink: 0,
              marginRight: isMobile ? 4 : 0,
            }}
          >
            {headerDateLabel}
          </div>
        </div>
        {isMobile && menuOpen && (
          <div style={{ borderTop: `1px solid ${COL.border}`, background: "#111827" }}>
            {NAV_TABS.map((t) => {
              const active = detailGameId ? t.id === "games" : mainTab === t.id;
              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => navigateToTab(t.id)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    width: "100%",
                    padding: "16px 20px",
                    border: "none",
                    borderBottom: `1px solid ${COL.border}`,
                    background: active ? "rgba(234,88,0,0.08)" : "transparent",
                    color: active ? CSS_TEXT : "#9CA3AF",
                    fontFamily: "inherit",
                    fontSize: 16,
                    fontWeight: active ? 800 : 600,
                    cursor: "pointer",
                  }}
                >
                  {t.label}
                  {active && <span style={{ width: 6, height: 6, borderRadius: "50%", background: COL.model }} />}
                </button>
              );
            })}
          </div>
        )}
        {(detailGameId || mainTab === "games" || mainTab === "players" || mainTab === "edges") && (
          <div style={{ maxWidth: detailGameId ? 1160 : 1200, margin: "0 auto", padding: `${PAGE_HEADER_TOP_PADDING}px clamp(16px, 3vw, 28px) 10px` }}>
            <ScheduleDateStrip
              date={date}
              todayStr={today}
              onPick={applyDate}
              refreshing={refreshing || (mainTab === "players" && playersLoading)}
              onManualRefresh={() => (mainTab === "players" ? fetchPlayers(true) : fetchGames(true))}
              statusLine={
                lastUpdated
                  ? `Updated ${formatTime(lastUpdated)}${isGameHours() ? " · auto-refreshing" : ""}`
                  : null
              }
            />
          </div>
        )}
      </div>

      <div style={{
        maxWidth: detailGameId ? 1160 : 1200,
        margin: "0 auto",
        padding: isMobile ? "0 12px calc(96px + env(safe-area-inset-bottom))" : "0 12px 24px",
      }}>

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
            propsBatters={playersBatters}
            propsPitchers={playersPitchers}
            propsLoading={playersLoading}
          />
        )}
        {!detailGameId && mainTab === "games" && loading && (
          <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "3rem" }}>Loading games...</p>
        )}
        {!detailGameId && mainTab === "games" && !loading && games.length === 0 && (
          <p style={{ color: COL.textSecondary, fontSize: 14, textAlign: "center", marginTop: "3rem" }}>
            No games with predictions for this date yet. Early inference runs overnight; confirmed-lineup updates appear through the morning.
          </p>
        )}
        {!detailGameId && mainTab === "games" && !loading && games.length > 0 && (
          <GamesTable
            sortedGames={sortedGames}
            live={live}
            enrich={enrich}
            onOpenDetail={openGameDetail}
            standingsMap={allStandings}
            scheduleDate={date}
          />
        )}
        {!detailGameId && mainTab === "players" && (
          <PlayersTab
            games={sortedGames}
            batters={playersBatters}
            pitchers={playersPitchers}
            loading={playersLoading || loading}
            selectedGameId={selectedPlayerGameId}
            onSelectGame={setSelectedPlayerGameId}
            live={live}
            scheduleDate={date}
          />
        )}
        {!detailGameId && mainTab === "edges" && (
          <TopEdgesPanel enabled date={date} refreshKey={accuracyRefreshKey} />
        )}
        {!detailGameId && mainTab === "trends" && (
          <TrendsPanel
            enabled
            refreshKey={accuracyRefreshKey}
            onGoToEdges={() => setMainTab("edges")}
            games={sortedGames}
            live={live}
            date={date}
          />
        )}
        {!detailGameId && mainTab === "standings" && (
          <StandingsPanel enabled date={date} refreshKey={accuracyRefreshKey} />
        )}
        {!detailGameId && mainTab === "transactions" && (
          <TransactionsPanel enabled date={date} refreshKey={accuracyRefreshKey} />
        )}
        {!detailGameId && mainTab === "accuracy" && (
          <ModelPerformancePanel enabled refreshKey={accuracyRefreshKey} />
        )}
        {!detailGameId && mainTab === "about" && (
          <AboutUsPanel />
        )}

        <footer style={{
          marginTop: 32,
          padding: "20px 0 8px",
          borderTop: `1px solid ${COL.border}`,
          textAlign: "center",
          color: COL.textMuted,
          fontSize: 13,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 8,
        }}>
          <span>© 2026 The Hot Corner</span>
          <AuthorCredit />
        </footer>

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
