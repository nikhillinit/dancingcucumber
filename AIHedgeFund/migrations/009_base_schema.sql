-- Base schema: instruments, vendor map, corporate actions, bars, backtests, OMS
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- instruments master
CREATE TABLE IF NOT EXISTS instruments (
  instrument_id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  asset_class TEXT NOT NULL CHECK (asset_class IN ('EQUITY','ETF','CRYPTO','FX','FUTURE')),
  exchange TEXT,
  currency TEXT DEFAULT 'USD',
  exchange_tz TEXT DEFAULT 'America/New_York',
  is_active BOOLEAN DEFAULT TRUE,
  UNIQUE(symbol, asset_class)
);

-- vendor symbol map
CREATE TABLE IF NOT EXISTS instrument_vendor_map (
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  vendor TEXT NOT NULL,
  vendor_symbol TEXT NOT NULL,
  UNIQUE(instrument_id, vendor)
);

-- corporate actions (daily)
CREATE TABLE IF NOT EXISTS corporate_actions (
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  ex_date DATE NOT NULL,
  split_ratio DOUBLE PRECISION DEFAULT 1.0,
  dividend DOUBLE PRECISION DEFAULT 0.0,
  PRIMARY KEY (instrument_id, ex_date)
);

-- 1m bars (raw)
CREATE TABLE IF NOT EXISTS bars_1m (
  ts TIMESTAMPTZ NOT NULL,
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low  DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION,
  vendor TEXT,
  ingested_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (instrument_id, ts)
);
SELECT create_hypertable('bars_1m', 'ts', if_not_exists => TRUE);

-- EOD bars (raw + optional adj fields)
CREATE TABLE IF NOT EXISTS bars_1d (
  ts DATE NOT NULL,
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low  DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  adj_close DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  adj_factor DOUBLE PRECISION DEFAULT 1.0,
  vendor TEXT,
  ingested_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (instrument_id, ts)
);
SELECT create_hypertable('bars_1d', 'ts', if_not_exists => TRUE);

-- backtests registry
CREATE TABLE IF NOT EXISTS backtests (
  id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT now(),
  strategy TEXT,
  params JSONB,
  universe JSONB,
  start_date DATE,
  end_date DATE,
  costs JSONB,
  metrics JSONB,
  equity_curve JSONB
);

-- OMS
CREATE TABLE IF NOT EXISTS oms_orders (
  id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT now(),
  client_order_id TEXT UNIQUE NOT NULL,
  side TEXT CHECK (side IN ('BUY','SELL')),
  type TEXT CHECK (type IN ('MARKET','LIMIT','STOP','STOP_LIMIT')),
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  qty DOUBLE PRECISION NOT NULL,
  limit_price DOUBLE PRECISION,
  time_in_force TEXT DEFAULT 'DAY',
  mode TEXT NOT NULL CHECK (mode IN ('DRYRUN','PAPER','LIVE')),
  status TEXT NOT NULL DEFAULT 'NEW',
  error TEXT
);

CREATE TABLE IF NOT EXISTS oms_executions (
  id UUID PRIMARY KEY,
  order_id UUID NOT NULL REFERENCES oms_orders(id),
  ts TIMESTAMPTZ NOT NULL,
  fill_qty DOUBLE PRECISION NOT NULL,
  fill_price DOUBLE PRECISION NOT NULL,
  fees DOUBLE PRECISION DEFAULT 0.0,
  liquidity TEXT CHECK (liquidity IN ('MAKER','TAKER')),
  UNIQUE(order_id, ts)
);

-- Execution controls: DB-level kill switch
CREATE TABLE IF NOT EXISTS execution_controls (
  id INT PRIMARY KEY DEFAULT 1,
  kill_switch BOOLEAN NOT NULL DEFAULT TRUE,
  max_notional DOUBLE PRECISION DEFAULT 0.0
);
INSERT INTO execution_controls (id, kill_switch)
VALUES (1, TRUE) ON CONFLICT (id) DO NOTHING;

-- ingestion watermarks
CREATE TABLE IF NOT EXISTS ingestion_state (
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  source TEXT NOT NULL,            -- e.g., 'alpaca_1m'
  last_ts TIMESTAMPTZ,
  PRIMARY KEY (instrument_id, source)
);

-- universe definition (optional but recommended for reproducibility)
CREATE TABLE IF NOT EXISTS universes (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  description TEXT,
  mode TEXT CHECK (mode IN ('INTRADAY','EOD')) DEFAULT 'EOD'
);
CREATE TABLE IF NOT EXISTS universe_membership (
  universe_id INT NOT NULL REFERENCES universes(id),
  ts DATE NOT NULL,
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  included BOOLEAN NOT NULL DEFAULT TRUE,
  PRIMARY KEY (universe_id, ts, instrument_id)
);