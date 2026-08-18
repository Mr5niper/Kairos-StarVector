#!/usr/bin/env python3
"""
Build and cache the conditioning feature matrix from the command line.

    python scripts/build_features.py
    python scripts/build_features.py --ticker SPY --start 2010-01-01

Writes artifacts/cond_features.csv. Needs no ML extras: the astronomical
features come from ephem and the price context from pandas.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml

from kairos import astro as A
from kairos import market as M
from kairos import paths as P
from kairos import waves as W


def main() -> int:
    ap = argparse.ArgumentParser(description="Build conditioning features.")
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path = args.config or P.config_path("default.yaml")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    data = cfg.get("data", {})
    astro_cfg = cfg.get("astro", {})
    wave_cfg = cfg.get("wave", {})

    ticker = args.ticker or data.get("ticker", "^GSPC")
    start = args.start or data.get("start", "2015-01-01")
    end = args.end or data.get("end")

    print(f"[features] fetching {ticker} from {start} to {end or 'today'}")
    df = M.fetch_ohlc(ticker, start=start, end=end,
                      interval=data.get("interval", "1d"))
    print(f"[features] {len(df)} bars, {df.index.min().date()} to {df.index.max().date()}")

    bodies = astro_cfg.get("bodies", ["SUN", "VENUS", "MARS", "JUPITER", "SATURN"])
    if astro_cfg.get("include_moon") and "MOON" not in bodies:
        bodies = list(bodies) + ["MOON"]

    cal = A.calendar_index(df.index.min(), df.index.max())
    print(f"[features] computing longitudes for {len(bodies)} bodies over {len(cal)} days")
    lons = A.longitudes(cal, bodies=bodies,
                        frame=astro_cfg.get("frame", "geocentric"))

    spec = A.EventSpec(
        aspects=astro_cfg.get("aspects_deg", [0, 90, 120, 180]),
        orb_deg=float(astro_cfg.get("orb_deg", 2.0)),
        min_separation_days=int(astro_cfg.get("min_separation_days", 3)),
    )
    events = A.aspect_events(lons, spec)
    print(f"[features] {len(events)} alignment events")

    wave = W.composite_pressure(
        lons.index, events,
        tau_days=float(wave_cfg.get("tau_days", 30)),
        horizon_days=int(wave_cfg.get("horizon_days", 180)),
        period_days=float(wave_cfg.get("oscillation_period", 90))
        if wave_cfg.get("oscillate") else None,
        phase_deg=float(wave_cfg.get("oscillation_phase_deg", 0)),
        lead_days=int(wave_cfg.get("lead_days", 0)),
    )

    from stock_forecast.pipeline import build_conditional_features
    feat, cols = build_conditional_features(df, wave=wave, lons=lons)

    out = args.out or os.path.join(P.artifacts_dir(), "cond_features.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frame = feat.copy()
    frame.insert(0, "Date", frame.index)
    frame.to_csv(out, index=False)

    print(f"[features] wrote {out}")
    print(f"[features] {feat.shape[0]} rows x {feat.shape[1]} columns: {cols}")

    ev_out = os.path.join(P.artifacts_dir(), "alignment_events.csv")
    events.to_csv(ev_out, index=False)
    print(f"[features] wrote {ev_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
