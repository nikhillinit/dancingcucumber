import { FastifyReply, FastifyRequest } from "fastify";
import { Env } from "../../../packages/shared/env";
import pg from "pg";

const pool = new pg.Pool({ connectionString: Env.DATABASE_URL });

export async function enforceRisk(req: FastifyRequest, rep: FastifyReply) {
  // DB kill switch (server-side trigger exists too)
  const { rows: [ctrl] } = await pool.query(
    'SELECT kill_switch, max_notional FROM execution_controls WHERE id=1'
  );
  if (ctrl?.kill_switch) {
    return rep.code(403).send({ ok: false, error: "Kill switch active (DB)" });
  }
  // TODO: add notional exposure & per-position checks here based on portfolio state
  return;
}