from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from .backtest_costs import run_ma_with_costs
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

class OHLC(BaseModel):
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

class BacktestReq(BaseModel):
    symbol: str
    bars: list[OHLC]
    fast: int = 10
    slow: int = 30
    init_cash: float = 100_000
    fees_bps: float = 2
    slip_bps: float = 5

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/bt/ma-crossover")
def bt_ma(req: BacktestReq):
    df = pd.DataFrame([b.dict() for b in req.bars]).set_index(pd.to_datetime([b.ts for b in req.bars]))
    price = df["close"]
    result = run_ma_with_costs(price, req.fast, req.slow, req.init_cash, req.fees_bps, req.slip_bps)
    return result