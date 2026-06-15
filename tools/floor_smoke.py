import sys

sys.path.insert(0, "apps/quant")
from advisor.backtest.floor import beats_floor, purged_walk_forward_sharpe  # noqa: F401

print("floor: importable")
