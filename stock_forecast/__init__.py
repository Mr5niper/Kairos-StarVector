"""
stock_forecast
==============
Optional forecasting benchmark: ARIMA, conditional LSTM, conditional
WGAN-GP, LightGBM residual fusion and meta-labelling, evaluated on
walk-forward windows.

Requires the machine-learning extras:

    pip install -r requirements-ml.txt

Nothing here is imported by the main GUI at startup, so the app runs
without torch installed. Import failures surface where they are used, in
the Forecast models tab, rather than preventing the program from opening.
"""
__version__ = "6.0.2"
__all__ = [
    "pipeline", "metrics", "eval", "backtest", "splits",
    "meta_labeling", "train_lstm", "train_gan", "models", "utils",
]
