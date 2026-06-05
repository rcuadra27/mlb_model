import { Fragment, useEffect, useState } from "react";

function useIsMobile(breakpoint = 768) {
  const [isMobile, setIsMobile] = useState(
    typeof window !== "undefined"
      ? window.matchMedia(`(max-width: ${breakpoint}px)`).matches
      : false,
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

/** Shared palette — matches App.jsx design tokens */
const COL = {
  card: "#1F2937",
  cardInner: "#172032",
  border: "#374151",
  model: "#F59E0B",
  positive: "#16A34A",
  negative: "#DC2626",
  text: "#F9FAFB",
  textMuted: "#6B7280",
};

const FONT_MONO = "'JetBrains Mono', ui-monospace, monospace";

const PROP_GRADE_HIT = "#4ADE80";
const PROP_GRADE_MISS_TEXT = "#9CA3AF";
const PROP_GRADE_MISS_DOT = "#5A6472";

/** Players-tab source of truth: column order and labels */
export const BATTER_PROP_TABLE_COLUMNS = [
  { key: "p_hit", label: "HIT" },
  { key: "p_2plus_hits", label: "2+H" },
  { key: "p_hr", label: "HR" },
  { key: "p_k", label: "K" },
  { key: "p_walk", label: "BB" },
  { key: "p_2plus_bases", label: "2+TB" },
];

function propPct(p) {
  if (p == null || p === "") return "—";
  return `${Math.round(Number(p) * 100)}%`;
}

function propPctNum(p) {
  if (p == null || p === "") return null;
  const n = Number(p);
  return Number.isFinite(n) ? n * 100 : null;
}

function hitCellColor(pct) {
  if (pct == null) return COL.text;
  if (pct >= 62) return COL.positive;
  if (pct <= 48) return COL.negative;
  return COL.text;
}

function hits2CellColor(pct) {
  if (pct == null) return COL.text;
  if (pct >= 28) return COL.positive;
  if (pct <= 14) return COL.negative;
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
  if (pct >= 68) return COL.negative;
  if (pct <= 52) return COL.positive;
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

const COLUMN_COLOR = {
  p_hit: hitCellColor,
  p_2plus_hits: hits2CellColor,
  p_hr: hrCellColor,
  p_k: kCellColor,
  p_walk: bbCellColor,
  p_2plus_bases: tbCellColor,
};

function batterPropOutcome(propKey, actuals) {
  if (!actuals) return false;
  if (propKey === "p_hit") return actuals.hits >= 1;
  if (propKey === "p_2plus_hits") return actuals.hits >= 2;
  if (propKey === "p_hr") return actuals.homeRuns >= 1;
  if (propKey === "p_k") return actuals.strikeouts >= 1;
  if (propKey === "p_walk") return actuals.walks >= 1;
  if (propKey === "p_2plus_bases") return actuals.totalBases >= 2;
  return false;
}

function isBatterPropDecided(propKey, actuals, gameFinished) {
  if (!actuals) return false;
  if (gameFinished) return true;
  return batterPropOutcome(propKey, actuals);
}

function batterPropGradePctColor(propKey, actuals, gameFinished, fallbackColor) {
  if (!isBatterPropDecided(propKey, actuals, gameFinished)) return fallbackColor;
  return batterPropOutcome(propKey, actuals) ? PROP_GRADE_HIT : PROP_GRADE_MISS_TEXT;
}

function BatterPropGradeIndicator({ propKey, actuals, gameFinished }) {
  if (!actuals || !isBatterPropDecided(propKey, actuals, gameFinished)) return null;
  const hit = batterPropOutcome(propKey, actuals);
  return (
    <span
      aria-label={hit ? "Prop hit" : "Prop did not hit"}
      title={hit ? "Prop hit" : "Prop did not hit"}
      style={{
        display: "inline-block",
        marginLeft: 3,
        fontSize: hit ? 11 : 13,
        lineHeight: 1,
        fontWeight: hit ? 900 : 600,
        color: hit ? PROP_GRADE_HIT : PROP_GRADE_MISS_DOT,
        verticalAlign: "baseline",
      }}
    >
      {hit ? "✓" : "·"}
    </span>
  );
}

export function BatterPropGradeLegend() {
  return (
    <div style={{ padding: "6px 16px", fontSize: 10, color: COL.textMuted, background: "#1F2937", borderBottom: `1px solid ${COL.border}` }}>
      <span style={{ color: PROP_GRADE_HIT, fontWeight: 800 }}>✓</span>
      {" prop hit · "}
      <span style={{ color: PROP_GRADE_MISS_DOT, fontWeight: 700 }}>·</span>
      {" did not hit"}
    </div>
  );
}

export function isLineupConfirmed(b) {
  return b?.lineup_confirmed === true || b?.lineup_confirmed === "true" || b?.lineup_confirmed === 1;
}

export function displayBattingOrder(b) {
  if (!isLineupConfirmed(b)) return "—";
  const o = b?.batting_order;
  if (o == null || o === "") return "—";
  const n = Number(o);
  return Number.isFinite(n) ? n : "—";
}

export function sortBattersForPropTable(batters) {
  if (!batters?.length) return [];
  if (batters.every(isLineupConfirmed)) {
    return [...batters].sort((a, b) => (Number(a.batting_order) || 99) - (Number(b.batting_order) || 99));
  }
  return [...batters].sort((a, b) => (a.batter_name || "").localeCompare(b.batter_name || ""));
}

export function filterBattersForDisplay(batters) {
  if (!batters?.length) return [];
  const confirmedGameIds = new Set();
  for (const b of batters) {
    if (isLineupConfirmed(b)) confirmedGameIds.add(String(b.game_id));
  }
  if (!confirmedGameIds.size) return batters;
  return batters.filter((b) => {
    if (!confirmedGameIds.has(String(b.game_id))) return true;
    return isLineupConfirmed(b) && b.batting_order != null && b.batting_order !== "";
  });
}

/** Split + sort batters for one game (same pipeline on Players tab and game details). */
export function prepareGameBatterLineups(batters, gameId) {
  const forGame = filterBattersForDisplay(
    (batters || []).filter((b) => String(b.game_id) === String(gameId)),
  );
  const home = sortBattersForPropTable(
    forGame.filter((b) => b.is_home === true || b.is_home === "true"),
  );
  const away = sortBattersForPropTable(
    forGame.filter((b) => b.is_home === false || b.is_home === "false"),
  );
  return { home, away, all: [...home, ...away] };
}

const HEADER_CELL = {
  padding: "8px 6px",
  textAlign: "center",
  fontSize: 10,
  fontWeight: 800,
  letterSpacing: 0.6,
  textTransform: "uppercase",
  color: "#4B5563",
  background: "#1F2937",
  whiteSpace: "nowrap",
};

/**
 * Single-team batter prop table — shared by Players tab and game details.
 */
export function BatterPropTable({
  title,
  subtitle,
  batters,
  theme,
  expandedId,
  onToggle,
  resolveGameLine = null,
  showBoxScore = false,
  gameFinished = false,
  renderExpanded = null,
}) {
  const th = theme || { primary: COL.model, soft: COL.cardInner };
  const [hoverId, setHoverId] = useState(null);
  const colSpan = 2 + BATTER_PROP_TABLE_COLUMNS.length;

  return (
    <div
      style={{
        flex: 1,
        minWidth: 0,
        width: "100%",
        borderRadius: 12,
        border: `1px solid ${COL.border}`,
        background: COL.card,
        overflow: "hidden",
        boxShadow: "0 4px 16px rgba(15,23,42,0.07)",
      }}
    >
      <div style={{ height: 2, background: th.primary }} />
      <div style={{ padding: "14px 16px", background: "#1F2937", borderBottom: `1px solid ${COL.border}` }}>
        <div style={{ fontSize: 14, fontWeight: 900, color: COL.text }}>{title}</div>
        {subtitle && <div style={{ fontSize: 11, color: COL.textMuted, marginTop: 4 }}>{subtitle}</div>}
      </div>
      {showBoxScore && <BatterPropGradeLegend />}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, minWidth: 520 }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${COL.border}` }}>
              <th style={{ ...HEADER_CELL, textAlign: "center" }}>#</th>
              <th style={{ ...HEADER_CELL, textAlign: "left" }}>Player</th>
              {BATTER_PROP_TABLE_COLUMNS.map((c) => (
                <th key={c.key} style={HEADER_CELL}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {batters.length === 0 ? (
              <tr>
                <td
                  colSpan={colSpan}
                  style={{
                    padding: "24px 16px",
                    textAlign: "center",
                    color: COL.textMuted,
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                >
                  Lineup not yet available
                </td>
              </tr>
            ) : batters.map((b, idx) => {
              const rowKey = `${b.game_id}-${b.batter_id}`;
              const expanded = expandedId === rowKey;
              const hovered = hoverId === rowKey;
              const confirmed = isLineupConfirmed(b);
              const hand = b.batter_hand || b.bats || null;
              const stripeBg = idx % 2 === 1 ? "#0D1420" : "transparent";
              const gameLine = showBoxScore && resolveGameLine ? resolveGameLine(b.batter_name) : null;
              const gameLineActuals = gameLine ? parseBatterGameLine(gameLine) : null;
              const rowBg = expanded ? th.soft : (hovered ? "#141E2E" : stripeBg);
              return (
                <Fragment key={rowKey}>
                  <tr
                    onClick={() => onToggle?.(rowKey)}
                    onMouseEnter={() => setHoverId(rowKey)}
                    onMouseLeave={() => setHoverId(null)}
                    style={{
                      cursor: onToggle ? "pointer" : "default",
                      background: rowBg,
                      borderBottom: `1px solid ${COL.border}`,
                      transition: "background 0.12s ease",
                    }}
                  >
                    <td style={{ padding: "8px 6px", textAlign: "center", fontWeight: 800, color: COL.textMuted }}>
                      {displayBattingOrder(b)}
                    </td>
                    <td style={{ padding: "8px 6px", fontWeight: 800, color: COL.text, whiteSpace: "nowrap" }}>
                      <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                        <div>
                          {b.batter_name || "—"}
                          {hand && (
                            <span style={{ fontSize: 10, color: COL.textMuted, marginLeft: 4, fontWeight: 600 }}>
                              {hand}
                            </span>
                          )}
                          {!confirmed && (
                            <span style={{ fontSize: 10, color: COL.textMuted, marginLeft: 6, fontWeight: 600 }}>
                              Projected
                            </span>
                          )}
                          {confirmed && (
                            <span style={{ fontSize: 10, color: COL.positive, marginLeft: 6, fontWeight: 600, opacity: 0.85 }}>
                              Confirmed
                            </span>
                          )}
                        </div>
                        {gameLine && (
                          <div style={{ fontSize: 10, fontWeight: 700, color: COL.model, fontFamily: FONT_MONO, letterSpacing: "0.01em" }}>
                            {gameLine}
                          </div>
                        )}
                      </div>
                    </td>
                    {BATTER_PROP_TABLE_COLUMNS.map((c) => {
                      const pct = propPctNum(b[c.key]);
                      const colorFn = COLUMN_COLOR[c.key] || (() => COL.text);
                      const cellColor = showBoxScore
                        ? batterPropGradePctColor(c.key, gameLineActuals, gameFinished, colorFn(pct))
                        : colorFn(pct);
                      return (
                        <td
                          key={c.key}
                          style={{
                            padding: "8px 4px",
                            textAlign: "center",
                            fontWeight: 800,
                            color: cellColor,
                            fontVariantNumeric: "tabular-nums",
                          }}
                        >
                          {propPct(b[c.key])}
                          <BatterPropGradeIndicator
                            propKey={c.key}
                            actuals={gameLineActuals}
                            gameFinished={gameFinished}
                          />
                        </td>
                      );
                    })}
                  </tr>
                  {expanded && renderExpanded && (
                    <tr>
                      <td colSpan={colSpan} style={{ padding: "16px 18px", background: COL.cardInner, borderBottom: `1px solid ${COL.border}` }}>
                        {renderExpanded(b, title)}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Home + away batter prop tables — identical layout on Players tab and game details. */
export function BatterPropLineupPair({
  homeTeam,
  awayTeam,
  homeBatters,
  awayBatters,
  homePitcherName,
  awayPitcherName,
  themeHome,
  themeAway,
  expandedBatterId,
  onToggleBatter,
  findGameLine = null,
  showBoxScore,
  gameFinished,
  renderExpanded,
  feedHomeBatters = null,
  feedAwayBatters = null,
}) {
  const isMobile = useIsMobile();
  const resolveHome = findGameLine && feedHomeBatters
    ? (name) => findGameLine(feedHomeBatters, name)
    : null;
  const resolveAway = findGameLine && feedAwayBatters
    ? (name) => findGameLine(feedAwayBatters, name)
    : null;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: isMobile ? "column" : "row",
        gap: 16,
        flexWrap: "wrap",
      }}
    >
      <BatterPropTable
        title={(homeTeam || "Home").toUpperCase()}
        subtitle={awayPitcherName ? `Batting vs ${awayPitcherName}` : null}
        batters={homeBatters}
        theme={themeHome}
        expandedId={expandedBatterId}
        onToggle={onToggleBatter}
        resolveGameLine={resolveHome}
        showBoxScore={showBoxScore}
        gameFinished={gameFinished}
        renderExpanded={renderExpanded}
      />
      <BatterPropTable
        title={(awayTeam || "Away").toUpperCase()}
        subtitle={homePitcherName ? `Batting vs ${homePitcherName}` : null}
        batters={awayBatters}
        theme={themeAway}
        expandedId={expandedBatterId}
        onToggle={onToggleBatter}
        resolveGameLine={resolveAway}
        showBoxScore={showBoxScore}
        gameFinished={gameFinished}
        renderExpanded={renderExpanded}
      />
    </div>
  );
}
