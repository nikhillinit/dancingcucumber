import pg from "pg";
import { Env } from "../../packages/shared/env";

const pool = new pg.Pool({ connectionString: Env.DATABASE_URL });

export async function getWatermark(instrumentId: number, source: string) {
  const { rows } = await pool.query(
    'SELECT last_ts FROM ingestion_state WHERE instrument_id=$1 AND source=$2',
    [instrumentId, source]
  );
  return rows[0]?.last_ts ?? null;
}

export async function setWatermark(instrumentId: number, source: string, ts: string) {
  await pool.query(
    `INSERT INTO ingestion_state (instrument_id, source, last_ts)
     VALUES ($1,$2,$3)
     ON CONFLICT (instrument_id, source) DO UPDATE SET last_ts=EXCLUDED.last_ts`,
    [instrumentId, source, ts]
  );
}