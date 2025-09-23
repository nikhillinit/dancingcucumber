# AI Hedge Fund Platform

Production-ready trading platform with hardening, observability, and scalability.

## Features

- **TimescaleDB**: Time-series optimized database with compression & retention policies
- **Risk Management**: Dual kill-switch (API + DB triggers), position limits
- **Data Pipeline**: Watermark-based ingestion with Bull queues, parallel processing
- **Observability**: Prometheus metrics, ingestion lag monitoring, queue depth tracking
- **Backtesting**: Vectorbt-powered MA crossover with costs, slippage, equity curves
- **Safety First**: Default kill-switch ON, defense-in-depth architecture

## Quick Start

1. **Setup Infrastructure**:
```bash
# Start TimescaleDB + Redis
npm run db:up

# Copy environment variables
cp .env.example .env
# Edit .env with your Alpaca API credentials
```

2. **Run Migrations**:
```bash
# Install dependencies
npm install

# Apply all migrations
cd AIHedgeFund
for f in migrations/*.sql; do PGPASSWORD=aihf psql -h localhost -U aihf -d aihf -f "$f"; done
```

3. **Install Python Dependencies**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

4. **Start Services** (3 terminal windows):

Terminal 1 - Quant Service:
```bash
source .venv/bin/activate
uvicorn AIHedgeFund.apps.quant.main:app --reload --port 8001
```

Terminal 2 - API Service:
```bash
npm run api
```

Terminal 3 - Worker Service:
```bash
npm run worker
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   API Service   │────▶│   Quant Service │     │  Worker Service │
│   (Fastify)     │     │    (FastAPI)    │     │   (Bull/Redis)  │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
         │                                                 │
         ▼                                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TimescaleDB                              │
│  - Hypertables (bars_1m, bars_1d)                              │
│  - Continuous Aggregates (bars_5m)                             │
│  - Compression & Retention Policies                            │
└─────────────────────────────────────────────────────────────────┘
```

## Endpoints

### API Service (Port 8081)
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /data/bars` - Fetch market bars
- `POST /orders` - Submit orders (guarded by kill switch)
- `POST /backtest/ma` - Run MA crossover backtest

### Quant Service (Port 8001)
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /v1/bt/ma-crossover` - MA crossover backtest engine

### Worker Metrics (Port 9101)
- `GET /metrics` - Worker-specific metrics (ingestion lag, queue depth)

## Safety Features

1. **Kill Switch**: Database trigger prevents all order execution when active
2. **Rate Limiting**: Vendor API throttling with exponential backoff
3. **Watermark Tracking**: Prevents duplicate data ingestion
4. **Position Limits**: Configurable max notional and position percentage
5. **Default Safe**: Kill switch ON by default until explicitly disabled

## Database Management

```sql
-- Disable kill switch (enable trading)
UPDATE execution_controls SET kill_switch = false WHERE id = 1;

-- Check ingestion lag
SELECT symbol, source, last_ts,
       EXTRACT(EPOCH FROM (now() - last_ts)) as lag_seconds
FROM ingestion_state;

-- View 5-minute bars
SELECT * FROM bars_5m
WHERE instrument_id = 1
ORDER BY ts DESC LIMIT 100;
```

## Monitoring

Access Adminer at http://localhost:8080 for database management.

Prometheus metrics available at:
- API: http://localhost:8081/metrics
- Quant: http://localhost:8001/metrics
- Worker: http://localhost:9101/metrics

## Production Deployment

1. Set `NODE_ENV=production` and update database credentials
2. Use proper secrets management for API keys
3. Configure monitoring/alerting (Prometheus + Grafana)
4. Set up log aggregation (ELK stack or similar)
5. Implement backup/recovery procedures
6. Add authentication/authorization layer
7. Deploy behind load balancer with SSL termination