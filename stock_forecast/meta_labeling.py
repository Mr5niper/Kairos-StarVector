# stock_forecast/meta_labeling.py
"""
Meta-labelling: a second model that decides whether to act on the first
model's signal, in the style of Lopez de Prado.
"""
from typing import Optional

import numpy as np


def build_meta_features(cond_last, preds, returns_vol=None):
    """Conditioning row, the prediction, and its magnitude."""
    c = np.asarray(cond_last, dtype=np.float32)
    p = np.asarray(preds, dtype=np.float32).reshape(-1, 1)
    parts = [c, p, np.abs(p)]
    if returns_vol is not None:
        parts.append(np.asarray(returns_vol, dtype=np.float32).reshape(-1, 1))
    return np.hstack(parts).astype(np.float32)


def build_meta_labels(y_true, preds, abs_threshold: float = 0.0):
    """1 when the primary model got the direction right, else 0."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(preds, dtype=float).ravel()
    n = min(len(y), len(p))
    y, p = y[:n], p[:n]
    tradeable = np.abs(p) >= float(abs_threshold)
    return ((np.sign(p) == np.sign(y)) & tradeable).astype(int)


def safe_train_meta_clf(x_train, y_train, params: dict):
    """
    Train the filter, returning None when the data cannot support it.

    Guards against a single-class target, too few rows, and a missing
    LightGBM install. Returning None is meaningful: `apply_meta_clf` then
    passes signals through unchanged instead of silently zeroing them.
    """
    if x_train is None or len(x_train) < 20:
        return None
    y = np.asarray(y_train).ravel()
    if len(np.unique(y)) < 2:
        return None
    if min(np.bincount(y.astype(int))) < 5:
        return None
    try:
        import lightgbm as lgb
    except Exception:
        return None
    p = dict(params or {})
    p.setdefault("verbosity", -1)
    p.setdefault("min_child_samples", 5)
    clf = lgb.LGBMClassifier(**p, random_state=42)
    clf.fit(x_train, y)
    return clf


# Kept for backwards compatibility with older scripts.
train_meta_clf = safe_train_meta_clf


def apply_meta_clf(clf, x_test, preds, threshold: float = 0.5):
    """Zero out predictions the filter rejects. Returns (filtered, probs)."""
    p = np.asarray(preds, dtype=float).ravel()
    if clf is None:
        return p, np.full_like(p, 0.5, dtype=float)
    try:
        prob = clf.predict_proba(x_test)[:, 1]
    except Exception:
        return p, np.full_like(p, 0.5, dtype=float)
    accept = (prob >= float(threshold)).astype(float)
    return p * accept, prob
