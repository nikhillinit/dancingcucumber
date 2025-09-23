from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from .backtest_costs import run_ma_with_costs
from prometheus_fastapi_instrumentator import Instrumentator

# Import AI Personas system
try:
    from .ai_personas_api import app as personas_app
    AI_PERSONAS_AVAILABLE = True
except ImportError:
    AI_PERSONAS_AVAILABLE = False
    print("Warning: AI Personas system not available. Install required dependencies.")

app = FastAPI(
    title="AI Hedge Fund Quant Engine",
    description="Quantitative analysis and AI-powered investment recommendations",
    version="1.0.0"
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Mount AI Personas sub-application if available
if AI_PERSONAS_AVAILABLE:
    app.mount("/ai-personas", personas_app)

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