# stock_forecast/models/arima.py
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
def arima_forecast(train_series, order=(3,1,2)):
    import warnings
    warnings.filterwarnings("ignore")
    try:
        arr = np.asarray(train_series, dtype=float).ravel()
        if len(arr) < sum(order) + 3:
            return float(arr[-1])
        model = ARIMA(arr, order=order)
        fitted = model.fit()
        yhat = fitted.forecast(steps=1)
        return float(np.asarray(yhat)[0])
    except Exception:
        return float(np.asarray(train_series).ravel()[-1])
