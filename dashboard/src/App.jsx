import { useEffect, useState, useCallback } from "react";

const API = "https://us-central1-mlb-model-491223.cloudfunctions.net/get-daily-predictions";
const today = new Date().toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });
const REFRESH_INTERVAL = 60000;

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
  const n = parseInt(price);
  return isNaN(n) ? "—" : n > 0 ? `+${n}` : `${n}`;
}

function computeOU(total, line) {
  if (!line) return null;
  return total - line >= 0.3 ? "over" : total - line <= -0.3 ? "under" : "push";
}

function formatTime(date) {
  if (!date) return null;
  return date.toLocaleTimeString("en-US", {
    timeZone: "America/Los_Angeles",
    hour: "numeric", minute: "2-digit", hour12: true,
  }) + " PT";
}

function Logo({ team, size = 44 }) {
  const url = logoUrl(team);
  const ini = team ? team.split(" ").map(w => w[0]).slice(-2).join("") : "?";
  if (!url) return (
    <div style={{ width: size, height: size, borderRadius: "50%", background: "#1a2235", display: "flex", alignItems: "center", justifyContent: "center", fontSize: size * 0.28, fontWeight: 700, color: "#4a5568", flexShrink: 0 }}>{ini}</div>
  );
  return <img src={url} alt={team} style={{ width: size, height: size, objectFit: "contain", flexShrink: 0 }} onError={e => { e.target.style.display = "none"; }} />;
}

function MoveChip({ move, isProb }) {
  if (move === null || move === undefined) return <span style={{ fontSize: 13, color: "#4a5568" }}>—</span>;
  const v = parseFloat(move);
  const display = isProb ? `${Math.abs(v * 100).toFixed(1)}%` : Math.abs(v).toFixed(2);
  if (v > (isProb ? 0.001 : 0.01)) return <span style={{ fontSize: 13, color: "#22c55e", fontWeight: 700 }}>▲ {display}</span>;
  if (v < (isProb ? -0.001 : -0.01)) return <span style={{ fontSize: 13, color: "#ef4444", fontWeight: 700 }}>▼ {display}</span>;
  return <span style={{ fontSize: 13, color: "#4a5568" }}>—</span>;
}

function Pill({ children, color }) {
  const c = {
    green: { bg: "rgba(34,197,94,0.15)", text: "#22c55e", bd: "rgba(34,197,94,0.3)" },
    red: { bg: "rgba(239,68,68,0.15)", text: "#ef4444", bd: "rgba(239,68,68,0.3)" },
    blue: { bg: "rgba(59,130,246,0.15)", text: "#60a5fa", bd: "rgba(59,130,246,0.3)" },
    gray: { bg: "rgba(100,116,139,0.1)", text: "#64748b", bd: "rgba(100,116,139,0.2)" },
  }[color] || { bg: "rgba(100,116,139,0.1)", text: "#64748b", bd: "rgba(100,116,139,0.2)" };
  return (
    <span style={{ background: c.bg, color: c.text, border: `1px solid ${c.bd}`, padding: "3px 10px", borderRadius: 100, fontSize: 11, fontWeight: 700, whiteSpace: "nowrap" }}>
      {children}
    </span>
  );
}

function GameCard({ g, oddsMode }) {
  const isVal = g.is_value_ml_away || g.is_value_ml_home || g.is_value_ou_over || g.is_value_ou_under;
  const ouLine = oddsMode === "morning" ? g.morning_ou_line : (g.closing_ou_line ?? g.morning_ou_line);
  const mph = oddsMode === "morning" ? g.morning_p_home : g.closing_p_home;
  const homeWins = g.p_win_home > g.p_win_away;
  const rawH = oddsMode === "morning" ? (g.morning_home_price ?? null) : (g.closing_home_price ?? g.morning_home_price ?? null);
  const rawA = oddsMode === "morning" ? (g.morning_away_price ?? null) : (g.closing_away_price ?? g.morning_away_price ?? null);
  const homeML = rawH !== null ? fmt(rawH) : (toAmerican(mph) ?? "—");
  const awayML = rawA !== null ? fmt(rawA) : (toAmerican(mph ? 1 - mph : null) ?? "—");
  const ouRec = g.ou_recommendation ? g.ou_recommendation.toLowerCase() : computeOU(g.total_runs_pred, ouLine);

  const gameFinished = g.status === 'Final' || g.status === 'Game Over';
  const teams = [
    { team: g.away_team, sp: g.away_sp_name, pct: g.p_win_away, ml: awayML, wins: !homeWins, score: g.away_runs },
    { team: g.home_team, sp: g.home_sp_name, pct: g.p_win_home, ml: homeML, wins: homeWins, score: g.home_runs },
  ];

  return (
    <div style={{
      background: "#0f1623",
      border: isVal ? "1.5px solid rgba(34,197,94,0.5)" : "1px solid #1a2235",
      borderRadius: 16,
      marginBottom: 12,
      overflow: "hidden",
    }}>
      <div style={{ padding: "16px 16px 12px" }}>
        {teams.map((r, i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            marginBottom: i === 0 ? 14 : 0,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 0 }}>
              <Logo team={r.team} size={38} />
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 14, color: "#e2e8f4", fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.team}</div>
                <div style={{ fontSize: 11, color: "#4a5568", marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.sp ?? "SP TBD"}</div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 16, flexShrink: 0 }}>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 10, color: "#4a5568", fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 3 }}>Win%</div>
                <span style={{
                  fontSize: 15, fontWeight: 700,
                  color: r.wins ? "#60a5fa" : "#64748b",
                  background: r.wins ? "rgba(59,130,246,0.12)" : "transparent",
                  padding: r.wins ? "2px 8px" : "0",
                  borderRadius: 100,
                }}>{r.pct}%</span>
              </div>
              <div style={{ textAlign: "right", minWidth: 44 }}>
                <div style={{ fontSize: 10, color: "#4a5568", fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 3 }}>Odds</div>
                <span style={{ fontSize: 15, color: "#94a3b8", fontWeight: 600 }}>{r.ml}</span>
              </div>
              {gameFinished && (
                <div style={{ textAlign: "right", minWidth: 28 }}>
                  <div style={{ fontSize: 10, color: "#4a5568", fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 3 }}>Final</div>
                  <span style={{ fontSize: 15, fontWeight: 700, color: r.score > (i === 0 ? g.home_runs : g.away_runs) ? "#22c55e" : "#e2e8f4" }}>{r.score !== null && r.score !== undefined ? Math.round(r.score) : ""}</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div style={{ background: "#0a0f1a", borderTop: "1px solid #1a2235", borderBottom: "1px solid #1a2235", padding: "12px 16px", display: "flex", gap: 8, justifyContent: "space-between" }}>
        {[
          { label: "Away Pred", value: g.away_runs_pred },
          { label: "Home Pred", value: g.home_runs_pred },
          { label: "Total", value: g.total_runs_pred, sub: ouLine ? `line ${ouLine}` : "no line", hi: true },
        ].map((s, i) => (
          <div key={i} style={{
            flex: 1, textAlign: "center",
            background: s.hi ? "rgba(59,130,246,0.1)" : "rgba(255,255,255,0.02)",
            border: `1px solid ${s.hi ? "rgba(59,130,246,0.3)" : "#1a2235"}`,
            borderRadius: 10, padding: "8px 4px",
          }}>
            <div style={{ fontSize: 9, color: "#4a5568", fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 4 }}>{s.label}</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: s.hi ? "#60a5fa" : "#e2e8f4" }}>{s.value ?? "—"}</div>
            {s.sub && <div style={{ fontSize: 10, color: "#4a5568", marginTop: 3 }}>{s.sub}</div>}
          </div>
        ))}
      </div>

      <div style={{ padding: "12px 16px", overflowX: "auto" }}>
        <div style={{ display: "flex", gap: 20, minWidth: "max-content" }}>
          {[
            {
              label: "O/U Rec",
              content: ouRec === "over" ? <Pill color="green">Over</Pill>
                : ouRec === "under" ? <Pill color="red">Under</Pill>
                : ouRec === "push" ? <Pill color="gray">Push</Pill>
                : <Pill color="gray">—</Pill>
            },
            {
              label: "ML Edge",
              content: (g.status === 'In Progress' || g.status === 'Warmup') 
                ? <span style={{ fontSize: 11, color: "#4a5568" }}>in progress</span>
                : g.edge_home >= 5 ? <Pill color="green">+{g.edge_home?.toFixed(1)}%</Pill>
                : g.edge_home <= -5 ? <Pill color="red">{g.edge_home?.toFixed(1)}%</Pill>
                : g.edge_home !== null && g.edge_home !== undefined ? <Pill color="gray">{g.edge_home?.toFixed(1)}%</Pill>
                : <span style={{ fontSize: 11, color: "#4a5568" }}>needs closing</span>
            },
            {
              label: "Sharp $",
              content: g.sharp_action_home === 1 ? <Pill color="blue">Home</Pill>
                : g.sharp_action_home === -1 ? <Pill color="red">Away</Pill>
                : g.sharp_action_home === 0 ? <span style={{ fontSize: 11, color: "#4a5568" }}>Neutral</span>
                : <span style={{ fontSize: 11, color: "#4a5568" }}>needs closing</span>
            },
            {
              label: "O/U Move",
              content: <MoveChip move={g.total_line_move} isProb={false} />
            },
            {
              label: "ML Move",
              content: <MoveChip move={g.home_line_move} isProb={true} />
            },
          ].map((col, i) => (
            <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, minWidth: 72 }}>
              <span style={{ fontSize: 9, fontWeight: 700, color: "#4a5568", letterSpacing: "0.07em", textTransform: "uppercase" }}>{col.label}</span>
              {col.content}
            </div>
          ))}
          {isVal && <div style={{ display: "flex", alignItems: "center" }}><Pill color="green">⚡ Value</Pill></div>}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [oddsMode, setOddsMode] = useState("morning");
  const [date, setDate] = useState(today);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

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

  const valueBets = games.filter(g => g.is_value_ml_away || g.is_value_ml_home || g.is_value_ou_over || g.is_value_ou_under);

  return (
    <div style={{ background: "#080d14", minHeight: "100vh", width: "100%", fontFamily: "'SF Pro Display',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" }}>
      <style>{`
        * { box-sizing: border-box; }
        input[type="date"]::-webkit-calendar-picker-indicator { filter: invert(0.5); }
        ::-webkit-scrollbar { height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1a2235; border-radius: 2px; }
      `}</style>

      <div style={{ maxWidth: 680, margin: "0 auto", padding: "0 12px 32px" }}>

        <div style={{ textAlign: "center", padding: "28px 0 20px" }}>
          <h1 style={{ fontSize: "clamp(24px, 6vw, 34px)", fontWeight: 800, color: "#e2e8f4", margin: "0 0 20px", letterSpacing: "-0.03em" }}>
            MLB Game Predictor
          </h1>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, flexWrap: "wrap" }}>
            <input type="date" value={date} onChange={e => setDate(e.target.value)}
              style={{ background: "#0f1623", border: "1px solid #1a2235", borderRadius: 10, color: "#e2e8f4", padding: "8px 12px", fontSize: 13, outline: "none", fontFamily: "inherit" }} />
            <div style={{ display: "flex", background: "#0f1623", border: "1px solid #1a2235", borderRadius: 10, padding: 3, gap: 3 }}>
              {["morning", "closing"].map(m => (
                <button key={m} onClick={() => setOddsMode(m)} style={{
                  fontSize: 12, fontWeight: 600, padding: "6px 14px", borderRadius: 8, cursor: "pointer", border: "none",
                  background: oddsMode === m ? "#1e3a5f" : "transparent",
                  color: oddsMode === m ? "#60a5fa" : "#64748b",
                  transition: "all 0.15s",
                }}>
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>
            <button onClick={() => fetchGames(true)} style={{
              fontSize: 12, fontWeight: 600, padding: "8px 14px", borderRadius: 10, cursor: "pointer",
              border: "1px solid #1a2235", background: "#0f1623", color: "#64748b",
              opacity: refreshing ? 0.5 : 1, transition: "opacity 0.15s",
            }}>
              {refreshing ? "..." : "↺"}
            </button>
          </div>
          {lastUpdated && (
            <p style={{ fontSize: 11, color: "#334155", margin: "10px 0 0" }}>
              Updated {formatTime(lastUpdated)}{isGameHours() ? " · auto-refreshing" : ""}
            </p>
          )}
        </div>

        {valueBets.length > 0 && (
          <div style={{ background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.25)", borderRadius: 12, padding: "12px 16px", marginBottom: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#22c55e", marginBottom: 6 }}>⚡ Value bets flagged</div>
            {valueBets.map(g => (
              <div key={g.game_id} style={{ fontSize: 12, color: "#22c55e", opacity: 0.8, marginBottom: 2 }}>
                {g.away_team} @ {g.home_team}
                {g.is_value_ml_away && " · ML Away"}
                {g.is_value_ml_home && " · ML Home"}
                {g.is_value_ou_over && " · Over"}
                {g.is_value_ou_under && " · Under"}
              </div>
            ))}
          </div>
        )}

        {loading
          ? <p style={{ color: "#334155", fontSize: 14, textAlign: "center", marginTop: "3rem" }}>Loading games...</p>
          : games.length === 0
          ? <p style={{ color: "#334155", fontSize: 14, textAlign: "center", marginTop: "3rem" }}>No games found for {date}</p>
          : [...games.filter(g => g.status !== 'Final' && g.status !== 'Game Over'),
               ...games.filter(g => g.status === 'Final' || g.status === 'Game Over')]
            .map(g => <GameCard key={g.game_id} g={g} oddsMode={oddsMode} />)
        }

        {/* Disclaimer */}
        <div style={{ textAlign: "center", padding: "24px 16px 8px", marginTop: 8 }}>
          <p style={{ fontSize: 11, color: "#1e2a3d", margin: 0, lineHeight: 1.6 }}>
            ⚠️ For informational and entertainment purposes only. This tool does not constitute
            financial or betting advice. Past model performance does not guarantee future results.
            Please gamble responsibly.
          </p>
        </div>

      </div>
    </div>
  );
}