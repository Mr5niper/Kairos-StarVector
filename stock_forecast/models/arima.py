# stock_forecast/models/arima.py
"""One-step ARIMA forecast with a safe fallback."""
import warnings

import numpy as np


def arima_forecast(train_series, order=(2, 0, 1)) -> float:
    """
    Fit ARIMA and forecast one step.

    The default order is (2, 0, 1) rather than the original (3, 1, 2). The
    input here is already log returns, which are close to stationary, so
    differencing again with d=1 over-differences the series, amplifies
    noise and biases the forecast toward zero.

    Falls back to the mean of recent values if the fit fails, which is a
    more sensible guess for a return series than the last value: returns
    are not persistent, prices are.
    """
    arr = np.asarray(train_series, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    fallback = float(np.mean(arr[-20:]))
    if len(arr) < sum(order) + 10:
        return fallback
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from statsmodels.tsa.arima.model import ARIMA
            fit = ARIMA(arr, order=order).fit()
            out = float(np.asarray(fit.forecast(steps=1))[0])
        return out if np.isfinite(out) else fallback
    except Exception:
        return fallback
