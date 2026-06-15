from __future__ import annotations


def vol_adjusted_fraction(annualized_vol: float) -> float:
    """Fraction of net liquidation value allowed in one name, scaled by volatility."""
    if annualized_vol < 0.15:
        return 0.25
    if annualized_vol < 0.30:
        return 0.20
    if annualized_vol < 0.50:
        return 0.15
    return 0.10


def correlation_multiplier(correlation: float) -> float:
    """Penalize crowded (highly correlated) positions; reward diversifiers."""
    if correlation >= 0.80:
        return 0.70
    if correlation >= 0.50:
        return 1.0
    return 1.10


def position_limit(net_liq: float, vol: float, correlation: float) -> float:
    """Dollar cap for a single position."""
    if net_liq <= 0:
        return 0.0
    return max(0.0, net_liq * vol_adjusted_fraction(vol) * correlation_multiplier(correlation))
