import fetch from "cross-fetch";
import { v4 as uuidv4 } from "uuid";
import { pool } from "./db";
import { BarsReq, BacktestReq } from "../../packages/shared/types";
import { getBarsFromAlpaca } from "./alpacaAdapter";

type BtResult = {
  total_return: number;
  sharpe: number;
  max_drawdown: number;
  turnover: number;
  trades: number;
  equity_curve: Array<[string, number]>;
};

export async function runAndPersistBacktest(req: BacktestReq): Promise<{ id: string; result: BtResult }> {
  // 1) Get bars (for demo: from vendor; you can swap to DB)
  const bars = await getBarsFromAlpaca({
    symbol: req.symbol,
    timeframe: req.timeframe,
    start: req.start,
    end: req.end
  } as BarsReq);

  // 2) Call Quant service
  const resp = await fetch("http://localhost:8001/v1/bt/ma-crossover", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      symbol: req.symbol,
      bars,
      fast: req.fast,
      slow: req.slow,
      init_cash: req.init_cash,
      fees_bps: req.fees_bps,
      slip_bps: req.slip_bps
    })
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Quant error ${resp.status}: ${text}`);
  }
  const result: BtResult = await resp.json();

  // 3) Persist into backtests
  const id = uuidv4();
  const start_date = req.start ? req.start.substring(0,10) : null;
  const end_date = req.end ? req.end.substring(0,10) : null;
  await pool.query(
    `INSERT INTO backtests
      (id, strategy, params, universe, start_date, end_date, costs, metrics, equity_curve)
     VALUES
      ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
    [
      id,
      "ma_crossover",
      JSON.stringify({ fast: req.fast, slow: req.slow }),
      JSON.stringify({ symbol: req.symbol, timeframe: req.timeframe }),
      start_date, end_date,
      JSON.stringify({ fees_bps: req.fees_bps, slip_bps: req.slip_bps }),
      JSON.stringify({
        total_return: result.total_return, sharpe: result.sharpe,
        max_drawdown: result.max_drawdown, turnover: result.turnover, trades: result.trades
      }),
      JSON.stringify(result.equity_curve)
    ]
  );
  return { id, result };
}