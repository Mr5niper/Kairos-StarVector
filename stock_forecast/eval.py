# stock_forecast/eval.py
from typing import Dict
from stock_forecast.metrics import rmse, mae, mape, directional_accuracy

def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "DPA": directional_accuracy(y_true, y_pred),
    }