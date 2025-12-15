# stock_forecast/meta_labeling.py
import numpy as np
import lightgbm as lgb
def build_meta_features(C_last: np.ndarray, preds: np.ndarray, returns_vol: np.ndarray = None):
    feats = [C_last, preds.reshape(-1,1), np.abs(preds).reshape(-1,1)]
    if returns_vol is not None:
        feats.append(returns_vol.reshape(-1,1))
    return np.hstack(feats).astype(np.float32)
def build_meta_labels(y_true: np.ndarray, preds: np.ndarray, abs_threshold: float = 0.0):
    mask = (np.abs(preds) >= abs_threshold)
    labels = ((np.sign(preds) == np.sign(y_true)) & mask).astype(int)
    return labels
def safe_train_meta_clf(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    # skip training if only one class in y
    if len(np.unique(y_train)) < 2 or len(y_train) < 5:
        return None
    clf = lgb.LGBMClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)
    return clf
def apply_meta_clf(clf, X_test: np.ndarray, preds: np.ndarray):
    if clf is None:
        return preds, np.ones_like(preds, dtype=float)*0.5
    accept_prob = clf.predict_proba(X_test)[:,1]
    accept = (accept_prob >= 0.5).astype(int)
    return preds * accept, accept_prob
