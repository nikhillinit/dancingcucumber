-- Compression & retention policies
-- Enable compression on minute bars and set policies
ALTER TABLE bars_1m SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id');
SELECT add_compression_policy('bars_1m', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention: prune minute bars older than 90 days; keep EOD 10 years
SELECT add_retention_policy('bars_1m', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('bars_1d', INTERVAL '10 years', if_not_exists => TRUE);