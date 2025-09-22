import { Pool, neonConfig } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-serverless';
import ws from "ws";
import * as schema from "@shared/schema";

// Configure Neon for serverless environment
neonConfig.webSocketConstructor = ws;
neonConfig.poolQueryViaFetch = true;
neonConfig.useSecureWebSocket = true;

if (!process.env.DATABASE_URL) {
  throw new Error(
    "DATABASE_URL must be set. Did you forget to provision a database?",
  );
}

// Configure connection pool with appropriate settings for serverless
export const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 1, // Limit connections for serverless environment
  idleTimeoutMillis: 1000, // Close idle connections quickly
  connectionTimeoutMillis: 5000, // 5 second connection timeout
});

export const db = drizzle({ client: pool, schema });

// Graceful shutdown handling
process.on('SIGINT', async () => {
  console.log('Received SIGINT, closing database pool...');
  await pool.end();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('Received SIGTERM, closing database pool...');
  await pool.end();
  process.exit(0);
});
