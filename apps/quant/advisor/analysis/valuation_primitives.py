"""Deterministic valuation primitives.

Vendored and adapted from virattt/ai-hedge-fund (MIT License):
https://github.com/virattt/ai-hedge-fund (src/agents/warren_buffett.py, src/agents/valuation.py)
Adapted to our data types; math unchanged. MIT notice retained per LICENSE.
"""
from __future__ import annotations


def owner_earnings(net_income: float, depreciation: float, capex: float,
                   working_capital_change: float = 0.0) -> float:
    """Buffett owner earnings: NI + D&A - maintenance capex - delta working capital."""
    return net_income + depreciation - abs(capex) - working_capital_change


def intrinsic_value_dcf(base_cash_flow: float, growth_rate: float = 0.06,
                        terminal_growth: float = 0.025, discount_rate: float = 0.10,
                        horizon: int = 10, margin_of_safety: float = 0.15) -> float:
    """Multi-stage DCF with Gordon terminal value and a margin-of-safety haircut."""
    if base_cash_flow <= 0:
        return 0.0
    pv = 0.0
    cf = base_cash_flow
    for year in range(1, horizon + 1):
        cf = cf * (1 + growth_rate)
        pv += cf / ((1 + discount_rate) ** year)
    terminal_cf = cf * (1 + terminal_growth)
    terminal_value = terminal_cf / (discount_rate - terminal_growth)
    pv += terminal_value / ((1 + discount_rate) ** horizon)
    return pv * (1 - margin_of_safety)
