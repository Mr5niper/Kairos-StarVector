# stock_forecast/models/arima.py
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(train_series, order=(3,1,2)):
    import warnings
    warnings.filterwarnings("ignore")
    try:
        model = ARIMA(train_series, order=order)
        fitted = model.fit()
        yhat = fitted.forecast(steps=1)
        return float(np.asarray(yhat)[0])
    except Exception:
        return float(train_series[-1])