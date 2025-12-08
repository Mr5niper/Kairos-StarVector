# stock_forecast/metrics.py
import numpy as np
from scipy import stats

def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def mape(y, yhat): return float(np.mean(np.abs((y - yhat) / (y + 1e-8))))
def directional_accuracy(y, yhat):
    y = np.asarray(y).ravel()
    yhat = np.asarray(yhat).ravel()
    mask = (y != 0)
    if mask.sum() == 0: return 0.5
    return float(np.mean(np.sign(yhat[mask]) == np.sign(y[mask])))

def diebold_mariano(e1, e2, h=1, power=2):
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_mean = d.mean()
    var = d.var(ddof=1) / n
    DM = d_mean / np.sqrt(var + 1e-12)
    p = 2*(1 - stats.t.cdf(np.abs(DM), df=n-1))
    return float(DM), float(p)