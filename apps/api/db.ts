import pg from "pg";
import { Env } from "../../packages/shared/env";

export const pool = new pg.Pool({
  connectionString: Env.DATABASE_URL
});