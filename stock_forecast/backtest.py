# stock_forecast/backtest.py
"""Long/short backtest with transaction costs."""
from typing import Dict

import numpy as np

TRADING_DAYS = 252.0


def simple_long_short(returns, pred, tc_bps: float = 3.0) -> Dict[str, float]:
    """
    Go long when the prediction is positive, short otherwise.

    Two corrections against the original version:

      * The signal is lagged by one bar. Previously the position for a bar
        was set using the prediction for that same bar and multiplied by
        that bar's return, so the strategy traded on information it could
        not have had at the open. That alone can turn a worthless model
        into an apparently excellent one.
      * Costs are charged on the bar the position actually changes, not one
        bar late.
    """
    r = np.asarray(returns, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    n = min(len(r), len(p))
    r, p = r[:n], p[:n]
    if n < 2:
        return {"cumret": 0.0, "sharpe": 0.0, "hit": 0.0,
                "turnover": 0.0, "max_drawdown": 0.0}

    raw = np.where(p > 0, 1.0, -1.0)
    signal = np.concatenate([[0.0], raw[:-1]])          # traded next bar

    gross = signal * r
    flips = np.abs(np.diff(np.concatenate([[0.0], signal])))
    cost = flips * (float(tc_bps) / 1e4)
    net = gross - cost

    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    dd = float(np.min(equity / peak - 1.0))

    sd = np.std(net)
    sharpe = float(np.mean(net) / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0

    traded = signal != 0
    hit = float(np.mean(np.sign(signal[traded]) == np.sign(r[traded]))) if traded.any() else 0.0

    return {
        "cumret": float(equity[-1] - 1.0),
        "sharpe": sharpe,
        "hit": hit,
        "turnover": float(np.mean(flips)),
        "max_drawdown": dd,
    }
