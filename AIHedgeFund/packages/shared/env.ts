import { z } from "zod";
import 'dotenv/config';

export const Env = z.object({
  NODE_ENV: z.enum(['development','test','production']).default('development'),
  PORT: z.string().default('8081'),
  WORKER_METRICS_PORT: z.string().default('9101'),
  DATABASE_URL: z.string(),
  REDIS_URL: z.string().default('redis://localhost:6379'),
  ALPACA_KEY_ID: z.string(),
  ALPACA_SECRET_KEY: z.string(),
  ALPACA_DATA_BASE_URL: z.string().default('https://data.alpaca.markets'),
  KILL_SWITCH: z.string().default('true'),        // default ON until explicitly disabled
  MAX_NOTIONAL: z.coerce.number().default(0),
  MAX_POS_PCT: z.coerce.number().default(0.1),
  MAX_DRAWDOWN: z.coerce.number().default(0.2),
}).parse(process.env);