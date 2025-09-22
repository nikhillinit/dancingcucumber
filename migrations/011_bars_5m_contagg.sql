-- Create 5m continuous aggregate with toolkit if available, else fallback MV
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb_toolkit') THEN
    RAISE NOTICE 'timescaledb_toolkit not found; creating fallback materialized view bars_5m';
    -- Fallback: standard MV with manual refresh
    CREATE MATERIALIZED VIEW IF NOT EXISTS bars_5m AS
    WITH b AS (
      SELECT time_bucket('5 minutes', ts) AS bucket_ts,
             instrument_id, ts, open, high, low, close, volume
      FROM bars_1m
    )
    SELECT
      bucket_ts AS ts,
      instrument_id,
      (ARRAY_AGG(open ORDER BY ts ASC))[1] AS open,
      MAX(high) AS high,
      MIN(low)  AS low,
      (ARRAY_AGG(close ORDER BY ts ASC))[array_length(ARRAY_AGG(close),1)] AS close,
      SUM(volume) AS volume
    FROM b
    GROUP BY bucket_ts, instrument_id
    WITH NO DATA;

    CREATE UNIQUE INDEX IF NOT EXISTS idx_bars_5m_pk ON bars_5m(instrument_id, ts);
  ELSE
    RAISE NOTICE 'timescaledb_toolkit present; creating continuous aggregate bars_5m';
    CREATE MATERIALIZED VIEW IF NOT EXISTS bars_5m
    WITH (timescaledb.continuous) AS
    SELECT
      time_bucket('5 minutes', ts) AS ts,
      instrument_id,
      first(open, ts) AS open,
      max(high) AS high,
      min(low) AS low,
      last(close, ts) AS close,
      sum(volume) AS volume
    FROM bars_1m
    GROUP BY 1, 2;

    -- Add refresh policy
    SELECT add_continuous_aggregate_policy(
      'bars_5m',
      start_offset => INTERVAL '90 days',
      end_offset   => INTERVAL '1 minute',
      schedule_interval => INTERVAL '1 minute',
      if_not_exists => TRUE
    );
  END IF;
END$$;