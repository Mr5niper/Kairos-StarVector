# stock_forecast/eval.py
from typing import Dict

from .metrics import directional_accuracy, mae, rmse


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    """
    Core metrics. MAPE is deliberately absent: on log returns the
    denominator is near zero constantly and the number is meaningless.
    """
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "DPA": directional_accuracy(y_true, y_pred),
    }
