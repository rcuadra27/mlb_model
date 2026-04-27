# MLB Game Prediction System

A production ML system that predicts MLB game outcomes using gradient-boosted run models and market calibration, deployed on Google Cloud Platform. The public site is **[mlbpredictor.com](https://mlbpredictor.com)** — a React dashboard with live scores, model vs market edges, model accuracy tracking, and an optional **AI assistant** that answers questions using the same data the model uses.

## Architecture

```
MLB Stats API / Statcast / Odds API / Open-Meteo (weather)
        ↓
   Cloud SQL (PostgreSQL) ← source of truth
        ↓
   Cloud Run jobs (daily pipeline, PT schedule)
        ↓
   BigQuery (mlb_model_logs.daily_games) — serving mirror
        ↓
   Cloud Functions (HTTP) ──→ React dashboard (Cloud Run / static)
        • get-daily-predictions — games JSON, model accuracy, odds board
        • mlb-agent-chat — Claude-powered Q&A over predictions + features
```

## Web dashboard

Built with **Vite + React** (`dashboard/`). Served as a static build (nginx in `dashboard/Dockerfile`).

| Area | What it does |
|------|----------------|
| **Games** | Pacific **date strip**; schedule of games with model win %, market %, edges, run predictions, O/U line and recommendation. Games are shown **only after starting lineups are confirmed** (when the pipeline has a full snapshot). |
| **Model accuracy** | Rolling **moneyline** and **over/under** stats (flat $10 stake), confidence buckets, and daily P&amp;L charts. Backed by `get-daily-predictions?view=accuracy` (grades finished games from Postgres or BigQuery). |
| **About us** | Long-form explanation of the model, features, training, deployment, and limitations (static copy + architecture diagrams). |
| **Assistant** | Floating chat widget calling **`mlb-agent-chat`**. Context includes the selected slate date and (on a game page) the matchup. |

Branding uses the project logos in `dashboard/public/`. All **calendar days** in the UI are **US Pacific** (`America/Los_Angeles`), aligned with MLB slate dates.

## MLB Agent Chat

`agent_chat/` is a **2nd gen Cloud Function** that runs a small **Claude** (Haiku) agent with **tools**:

- Pull game predictions and market lines from BigQuery  
- Load per-game **feature rows** from Cloud SQL (why the model leans a certain way)  
- Team lookup, recent accuracy summaries, feature-importance CSVs  

**Time zone:** tools and the system prompt use **`America/Los_Angeles`** for “today” and default dates, matching the dashboard (not UTC).

**Config:** `ANTHROPIC_API_KEY`, `PG_DSN`, and the same BigQuery table as the rest of the stack. See `agent_chat/requirements.txt` and `agent_chat/main.py`.

## Cloud Function: `get-daily-predictions`

`cloud_function/main.py` — single HTTP entrypoint with **views**:

| Query | Purpose |
|-------|--------|
| `?date=YYYY-MM-DD` | Daily game list for the dashboard (from BigQuery). |
| `?view=accuracy` | Aggregated ML and O/U grading, buckets, daily P&amp;L (Postgres preferred via `PG_DSN`, else BigQuery). |
| `?view=odds_board` | Odds board payload (when enabled). |

Accuracy logic applies the same grading rules in one place so the **Model accuracy** tab matches how picks are evaluated.

## Model (summary)

- **Core:** LightGBM **team run** models (home/away expected runs), plus derived **win probability** and **totals** vs the market.  
- **Features:** Pitching (season + rolling windows), bullpen workload, lineup/matchup signals, **weather** (Open-Meteo), **umpire** tendencies, **market** lines and movement.  
- **Calibration:** Inference can apply a stored **calibrator** (e.g. isotonic) to raw win probabilities so tail behavior lines up with historical closing odds.  
- **Production** metadata in exports refer to the current pipeline generation (e.g. **v9** in app copy); artifact filenames in-repo may still say `optionA` / older tags — treat the **inference + export** path as source of truth for what’s live.

For a full narrative, see the **About us** page on the site.

## Project structure (selected)

```
mlb_model/
├── agent_chat/                 # mlb-agent-chat Cloud Function (Claude + tools)
├── cloud_function/             # get-daily-predictions (games, accuracy, odds)
├── dashboard/                  # React app + public assets + Dockerfile
├── features/                   # Feature pipelines (build_features1, lineups, odds scheduler, …)
├── ingest/                     # Schedule, pitchers, lineups, weather, odds pulls
├── inference/                  # Daily inference + calibration hooks
├── models/                     # Training scripts
├── calibration/                # Calibrators used at inference
├── artifacts/                  # Trained models & calibrators (large; see .gitignore)
├── export_to_bigquery.py
├── run_daily.sh                # Pipeline steps (PT) for Cloud Run
├── Dockerfile.pipeline
├── cloudbuild.yaml             # Example build for pipeline image
└── README.md
```

## Daily pipeline (PT)

Jobs are orchestrated from **`run_daily.sh`** and Cloud Run; exact clock times live with your GCP scheduler. Typical flow:

1. Backfill schedule and results  
2. Pitchers and appearances  
3. Features (optional statcast pass after lineups / closing window)  
4. Odds and market movement  
5. **Lineups** → full feature build + **inference** → **export to BigQuery** → games appear on the site  

## Inference chain (lineups or closing odds)

```
lineup confirmed OR closing odds stored (per your scheduler rules)
    → build_features1.py --date TODAY
    → inference.py (with calibrator when configured)
    → export_to_bigquery.py --date TODAY
    → rows available to get-daily-predictions + agent tools
```

## Key design decisions

- **Closing odds frozen pre-game** where the schema enforces a single write  
- **Live odds** do not overwrite stored morning/closing snapshots inappropriately  
- **BigQuery** mirror uses an explicit schema for `daily_games`  
- **Pacific time** for schedule dates and dashboard behavior; **agent** aligned to the same calendar  
- **Model accuracy** O/U stats can exclude specific slate dates when configured in `cloud_function/main.py` (e.g. bad data window) — moneyline unchanged  

## GCP resources

| Resource | Notes |
|----------|--------|
| **Project** | `mlb-model-491223` |
| **Region** | `us-central1` (typical for functions + run) |
| **Cloud SQL** | `mlb-postgres` (PostgreSQL) |
| **BigQuery** | `mlb_model_logs.daily_games` |
| **Images** | e.g. `gcr.io/mlb-model-491223/mlb-pipeline:v1`, `mlb-dashboard:v1` |
| **Cloud Functions** | `get-daily-predictions`, `mlb-agent-chat` |

## Environment variables

```bash
# Database (pipeline, inference, export, cloud function accuracy, agent tools)
PG_DSN=postgresql+psycopg2://user:password@host/dbname

# Odds
ODDS_API_KEY=your_odds_api_key

# Agent chat only
ANTHROPIC_API_KEY=sk-ant-...
```

## Deploying (quick reference)

**Dashboard** (static + nginx):

```bash
cd dashboard && npm run build
gcloud builds submit --tag gcr.io/mlb-model-491223/mlb-dashboard:v1 .
gcloud run deploy mlb-dashboard --image gcr.io/mlb-model-491223/mlb-dashboard:v1 --region us-central1
```

**`get-daily-predictions`** (from repo root):

```bash
gcloud functions deploy get-daily-predictions \
  --gen2 --runtime=python311 --region=us-central1 \
  --source=cloud_function --entry-point=get_daily_predictions \
  --trigger-http --allow-unauthenticated --project=mlb-model-491223
```

(Preserve existing `--set-env-vars` / secrets for `PG_DSN`, etc.)

**`mlb-agent-chat`** (from repo root):

```bash
gcloud functions deploy mlb-agent-chat \
  --gen2 --runtime=python311 --region=us-central1 \
  --source=agent_chat --entry-point=agent_chat \
  --trigger-http --allow-unauthenticated --project=mlb-model-491223
```

Set `ANTHROPIC_API_KEY` (and `PG_DSN` if tools need SQL) via `--set-env-vars` or Secret Manager.

## Local development

```bash
# Cloud SQL Proxy
./cloud-sql-proxy mlb-model-491223:us-central1:mlb-postgres --port=5434

# Manual feature + inference + export
PG_DSN="..." python features/build_features1.py --date 2026-03-31
PG_DSN="..." ODDS_API_KEY="..." python inference/inference.py --date 2026-03-31
PG_DSN="..." python export_to_bigquery.py --date 2026-03-31
```

## Contact

See the **About us** page on [mlbpredictor.com](https://mlbpredictor.com) for project context and contact email.
