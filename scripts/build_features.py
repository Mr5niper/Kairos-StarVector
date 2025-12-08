# scripts/build_features.py
import os
import yaml
import pandas as pd
from stock_forecast.utils import ensure_dir
from stock_forecast.dataset import fetch_ohlc_yf, assemble_conditional

def main(cfg_path="configs/default.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    ensure_dir("artifacts")
    mode = cfg['features']['mode']
    if mode != "real":
        print("features.mode != 'real'; nothing to build.")
        return

    df = fetch_ohlc_yf(cfg['data']['ticker'], start=cfg['data']['start'], end=cfg['data']['end'])
    cond_df, cond_dim = assemble_conditional("real", df, cfg['features'])
    cache_path = cfg['features']['cache_path']
    if cache_path:
        out = cond_df.copy()
        out.insert(0, 'Date', out.index)
        out.to_csv(cache_path, index=False)
        print(f"[build_features] wrote cache: {cache_path} (cond_dim={cond_dim})")

if __name__ == "__main__":
    main()