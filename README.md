# MLB Game Prediction System

A production ML system that predicts MLB game outcomes using gradient-boosted models and market calibration, deployed on Google Cloud Platform. The public site is **[the-hot-corner.com](https://the-hot-corner.com)** — a React dashboard with live scores, model vs market edges, player props, model accuracy tracking, and an **AI assistant** that answers questions using the same data the model uses.

## Architecture

```
MLB Stats API / Statcast / Odds API (server only) / Open-Meteo (weather)
        ↓
   Cloud SQL (PostgreSQL) ← source of truth
        ↓
   Cloud Run jobs (daily pipeline, PT schedule)
        ↓
   BigQuery (mlb_model_logs.daily_games) — serving mirror
        ↓
   Cloud Functions (HTTP) ──→ React dashboard (Cloud Run)
        • get-daily-predictions — games JSON, model accuracy
        • mlb-agent-chat — Claude-powered Q&A over predictions + features
```

**Odds API is server-only.** The dashboard never calls `the-odds-api.com`. Morning and closing lines are pulled by Cloud Run jobs (`market_movement`, `closing_odds`) and stored in Postgres/BigQuery. The API key lives in **Secret Manager** (`odds-api-key`) on those jobs only — not in the dashboard build, not on `get-daily-predictions`, and not in any client bundle.

## Web dashboard

Built with **Vite + React** (`dashboard/`). Served as a static build (nginx in `dashboard/Dockerfile`).

| Area | What it does |
|------|----------------|
| **Games** | Pacific **date strip**; schedule with model win %, **market %**, **edges**, run predictions, O/U line and recommendation. Data comes from `get-daily-predictions` (BigQuery mirror), not live browser odds polling. |
| **Player props** | Batter and pitcher prop tables backed by inference exports. |
| **Model accuracy** | Rolling **moneyline** and **over/under** stats (flat $10 stake), confidence buckets, and daily P&amp;L charts. |
| **About us** | Model narrative, features, training, deployment, and limitations. |
| **Assistant** | Floating chat widget calling **`mlb-agent-chat`**. |

Branding: **The Hot Corner** (`dashboard/public/the-hot-corner-*.svg`). Contact: **contact@the-hot-corner.com**.

All **calendar days** in the UI are **US Pacific** (`America/Los_Angeles`).

### What the dashboard does *not* do (by design)

Per-book live moneyline / runline / totals grids in the game detail modal are **disabled**. Those previously polled the Odds API from every visitor’s browser and caused runaway quota usage. Consensus morning market %, O/U line, edge, and rec still display from the server pipeline.

## MLB Agent Chat

`agent_chat/` is a **2nd gen Cloud Function** with **Claude** (Haiku) and tools:

- Game predictions and market lines from BigQuery  
- Per-game **feature rows** from Cloud SQL  
- Team lookup, accuracy summaries, feature-importance CSVs  

Tools use **`America/Los_Angeles`** for slate dates, matching the dashboard. Config: `ANTHROPIC_API_KEY`, `PG_DSN`, BigQuery table names.

## Cloud Function: `get-daily-predictions`

`cloud_function/main.py` — HTTP entrypoint:

| Query | Purpose |
|-------|--------|
| `?date=YYYY-MM-DD` | Daily game list (BigQuery `daily_games`). |
| `?view=accuracy` | ML and O/U grading, buckets, daily P&amp;L. |
| `?view=odds_board` | **Disabled** (returns 410). Live odds board removed. |
| `?view=edges` / `?view=trends` / etc. | Supporting views for dashboard tabs. |

**No `ODDS_API_KEY`** on this service. Responses are cached ~5 minutes in-process.

**Scaling:** `--max-instances=6` (see `scripts/configure_api_scaling.sh`).

## Model (production: v10)

| Component | Artifact / script |
|-----------|-------------------|
| **Moneyline** | `inference/inference_v10.py` → `artifacts/baseline_v10_production.joblib` |
| **Totals / O-U** | `inference/inference_v10_total.py` → `artifacts/totals_v10_umpire_runs_boost_sp_xwoba_total.joblib` |
| **Player props** | `inference/inference_props_v1.py` → `artifacts/props_v1_expanded.joblib`, pitcher prop models |
| **Edges** | `features/build_edges.py` (ML + prop edges → `daily_edges`) |

Features include pitching (rolling windows, pitch mix), bullpen workload, lineup matchups, weather (Open-Meteo), umpire tendencies, and **morning market lines** for edge computation.

Legacy `inference/inference.py` (v9) remains in-repo for reference; **`run_daily.sh` morning path uses v10**.

## Project structure

```
mlb_model/
├── agent_chat/                 # mlb-agent-chat Cloud Function
├── cloud_function/             # get-daily-predictions
├── dashboard/                  # React app (no VITE_ODDS_API_KEY)
├── features/                   # Features, edges, trends, market_movement, odds_quota_alert, …
├── ingest/                     # Schedule, pitchers, lineups, rosters, standings, transactions
├── inference/                  # inference_v10, inference_v10_total, inference_props_v1
├── models/                     # Training scripts (v10, props, pitcher extras)
├── scripts/                    # Ops: key rotation, quota alerts, image pinning, scaling
├── artifacts/                  # .joblib models (gitignored; COPY into Docker build)
├── export_to_bigquery.py
├── run_daily.sh                # Pipeline entrypoint for Cloud Run jobs
├── Dockerfile.pipeline
├── cloudbuild.yaml
└── README.md
```

## Daily pipeline (PT)

Orchestrated by **`run_daily.sh`** on Cloud Run. Key jobs:

| Cloud Run job | `run_daily.sh` mode | Role |
|---------------|---------------------|------|
| `mlb-morning-inference` | `morning_inference` | Full morning chain: lineups → features → v10 inference → edges → BQ export |
| `mlb-market-movement` | `market_movement` | Morning h2h + totals pull (2 credits/day typical) |
| `mlb-closing-odds` | `closing_odds` | Pre-first-pitch closing snapshot |
| `mlb-ingest-lineups` | `ingest_lineups` / chain | Lineup-triggered re-inference |
| `mlb-odds-quota-check` | `odds_quota_check` | Hourly usage → Pushover if over threshold |
| `mlb-early-inference` | `early_inference` | Early slate without full lineups |

### Morning inference chain

```
ingest standings / transactions / rosters
  → starting pitchers, pitch mix, lineups, umpire, weather
  → market_movement (morning odds → Postgres)
  → build_features1, build_trends, statcast pass
  → inference_v10 + inference_v10_total + inference_props_v1
  → build_edges, project_standings, build_model_performance
  → export_to_bigquery.py
  → pipeline_smoke_test.py
```

### Recovery modes (no full morning re-run)

| Mode | When to use |
|------|-------------|
| `export_bq_only` | Postgres already has preds; refresh BigQuery |
| `inference_export_refresh` | Morning odds landed after inference — recompute v10 + totals + export **without** re-pulling odds |
| `edges_refresh` | Rebuild edges table only |
| `market_movement` | Re-pull morning lines after quota/key fix |

```bash
gcloud run jobs execute mlb-morning-inference --region=us-central1 \
  --args=run_daily.sh,inference_export_refresh --wait
```

## Odds API operations

### Key rotation (server only)

```bash
NEW_ODDS_API_KEY='your-new-key' ./scripts/rotate_odds_server_key.sh
```

Writes Secret Manager `odds-api-key` and mounts it on `mlb-morning-inference`, `mlb-market-movement`, `mlb-closing-odds`. Strips `ODDS_API_KEY` from `get-daily-predictions` and unused jobs.

### Quota alert (Pushover)

```bash
ODDS_MONTHLY_QUOTA=20000 ODDS_ALERT_PCT=25 ./scripts/setup_odds_quota_pushover.sh
```

Creates `mlb-odds-quota-check` + hourly scheduler. Default **25% of monthly quota** (5,000 on a 20K plan).

### Pin pipeline image digest

After `gcloud builds submit`, pin all inference jobs to one digest:

```bash
./scripts/pin_pipeline_jobs.sh
```

Writes `config/production_image.env` (gitignored).

## Key design decisions

- **Odds API server-only** — no browser or Cloud Function calls; prevents per-visitor quota burn  
- **Morning odds frozen** for edge computation; closing odds written once pre-game  
- **BigQuery** explicit schema for `daily_games` (market %, edges, movement fields)  
- **Pacific time** for slate dates across dashboard, pipeline, and agent  
- **Lineup-gated games** — full snapshots when lineups confirmed  
- **Per-book live grids removed** until a server-backed design replaces client polling  

## GCP resources

| Resource | Notes |
|----------|--------|
| **Project** | `mlb-model-491223` |
| **Region** | `us-central1` |
| **Cloud SQL** | `mlb-postgres` (PostgreSQL) |
| **BigQuery** | `mlb_model_logs.daily_games`, `daily_edges`, props tables, … |
| **Secret Manager** | `odds-api-key` (server jobs only) |
| **Images** | `gcr.io/mlb-model-491223/mlb-model`, `mlb-dashboard` |
| **Cloud Functions** | `get-daily-predictions`, `mlb-agent-chat` |

## Environment variables

```bash
# Database (pipeline, inference, export, cloud function, agent)
PG_DSN=postgresql+psycopg2://user:password@host/dbname

# Odds — SERVER JOBS ONLY (Secret Manager in production, not dashboard)
ODDS_API_KEY=your_odds_api_key

# Pipeline alerts (Pushover)
PIPELINE_ALERT_WEBHOOK_URL=https://api.pushover.net/1/messages.json
PUSHOVER_USER_KEY=...
PUSHOVER_APP_TOKEN=...

# Agent chat
ANTHROPIC_API_KEY=sk-ant-...
```

**Never** set `VITE_ODDS_API_KEY` in `dashboard/.env`.

## Deploying

### Pipeline image

Models must exist under `artifacts/*.joblib` locally before build (gitignored; not in GitHub).

```bash
gcloud builds submit --config=cloudbuild.yaml --project=mlb-model-491223 .
# Or build pipeline only:
docker build -f Dockerfile.pipeline -t gcr.io/mlb-model-491223/mlb-model:latest .
gcloud push gcr.io/mlb-model-491223/mlb-model:latest

./scripts/pin_pipeline_jobs.sh
```

### Dashboard

```bash
cd dashboard && npm run build
gcloud builds submit --tag gcr.io/mlb-model-491223/mlb-dashboard:latest ./dashboard
gcloud run deploy mlb-dashboard \
  --image gcr.io/mlb-model-491223/mlb-dashboard:latest \
  --region us-central1 --project=mlb-model-491223
```

### `get-daily-predictions`

```bash
gcloud functions deploy get-daily-predictions \
  --gen2 --runtime=python311 --region=us-central1 \
  --source=cloud_function --entry-point=get_daily_predictions \
  --trigger-http --allow-unauthenticated --project=mlb-model-491223 \
  --max-instances=6
```

Or: `./scripts/configure_api_scaling.sh`

### `mlb-agent-chat`

```bash
gcloud functions deploy mlb-agent-chat \
  --gen2 --runtime=python311 --region=us-central1 \
  --source=agent_chat --entry-point=agent_chat \
  --trigger-http --allow-unauthenticated --project=mlb-model-491223
```

### Alerts

```bash
PIPELINE_ALERT_EMAIL=you@example.com ./scripts/setup_pipeline_alerts.sh
PIPELINE_ALERT_EMAIL=you@example.com ./scripts/setup_agent_chat_alerts.sh
```

## Local development

```bash
# Cloud SQL Proxy
./cloud-sql-proxy mlb-model-491223:us-central1:mlb-postgres --port=5434

export PG_DSN="postgresql+psycopg2://..."
export ODDS_API_KEY="..."   # local only — never commit

# Feature + inference + export
python features/build_features1.py --date 2026-06-04
python inference/inference_v10.py --date 2026-06-04 \
  --model artifacts/baseline_v10_production.joblib --fill_missing
python export_to_bigquery.py --date 2026-06-04
```

## Contact

**contact@the-hot-corner.com** — also linked in the dashboard footer.
