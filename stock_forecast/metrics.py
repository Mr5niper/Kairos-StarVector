# stock_forecast/metrics.py
"""Error and direction metrics, plus the Diebold-Mariano test."""
import numpy as np


def _pair(y, yhat):
    a = np.asarray(y, dtype=float).ravel()
    b = np.asarray(yhat, dtype=float).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def rmse(y, yhat) -> float:
    a, b = _pair(y, yhat)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def mae(y, yhat) -> float:
    a, b = _pair(y, yhat)
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def mape(y, yhat) -> float:
    """
    Mean absolute percentage error.

    Near useless on log returns, which cross zero constantly, so the
    denominator explodes and the metric reports millions of percent. Kept
    for continuity with the original reports but excluded from the default
    metric set; read RMSE and DPA instead.
    """
    a, b = _pair(y, yhat)
    if not len(a):
        return float("nan")
    denom = np.where(np.abs(a) < 1e-8, np.nan, a)
    return float(np.nanmean(np.abs((a - b) / denom)))


def directional_accuracy(y, yhat) -> float:
    """Share of bars where the predicted sign matches the realised sign."""
    a, b = _pair(y, yhat)
    mask = a != 0
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(np.sign(b[mask]) == np.sign(a[mask])))


def diebold_mariano(e1, e2, power: int = 2):
    """
    Test whether two forecasts differ in accuracy.

    Uses a Newey-West variance so serially correlated loss differentials
    do not inflate the statistic. The original version divided by the
    plain sample variance, which understates the standard error whenever
    errors are autocorrelated, and forecast errors nearly always are.
    Returns (statistic, p-value).
    """
    a = np.asarray(e1, dtype=float).ravel()
    b = np.asarray(e2, dtype=float).ravel()
    n = min(len(a), len(b))
    d = np.abs(a[:n]) ** power - np.abs(b[:n]) ** power
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return float("nan"), float("nan")

    dbar = d.mean()
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    gamma0 = np.sum((d - dbar) ** 2) / n
    var = gamma0
    for k in range(1, max(lag, 1) + 1):
        if k >= n:
            break
        cov = np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    var = max(var, 1e-18)

    stat = dbar / np.sqrt(var / n)
    try:
        from scipy import stats
        p = float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))
    except Exception:
        p = float("nan")
    return float(stat), p
