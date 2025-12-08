# stock_forecast/backtest.py
from typing import Dict
import numpy as np

def simple_long_short(returns: np.ndarray, pred: np.ndarray, tc_bps: float = 3.0) -> Dict[str, float]:
    signal = np.where(pred > 0, 1.0, -1.0)
    gross = signal * returns
    flips = (np.abs(np.diff(signal)) > 0).astype(float)
    cost = np.concatenate([[0.0], flips * (tc_bps / 1e4)])
    net = gross - cost
    cumret = np.cumprod(1 + net) - 1
    sharpe = np.mean(net) / (np.std(net) + 1e-9) * np.sqrt(252)
    return {
        "cumret": float(cumret[-1]),
        "sharpe": float(sharpe),
        "hit": float(np.mean(np.sign(pred) == np.sign(returns))),
        "turnover": float(np.mean(flips)),
    }