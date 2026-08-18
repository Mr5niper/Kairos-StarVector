#!/usr/bin/env python3
"""
Command-line walk-forward benchmark.

    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --ticker SPY --epochs 40 --windows 4

Requires the machine-learning extras:

    pip install -r requirements-ml.txt

Writes artifacts/benchmark_summary.csv, artifacts/benchmark_detail.csv and
artifacts/benchmark_predictions.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from kairos import astro as A
from kairos import market as M
from kairos import paths as P
from kairos import waves as W


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward model benchmark.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--windows", type=int, default=2,
                    help="0 runs every window that fits")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except Exception:
        print("[benchmark] PyTorch is not installed.")
        print("[benchmark] Install the extras first:")
        print("[benchmark]     pip install -r requirements-ml.txt")
        return 1

    cfg_path = args.config or P.config_path("default.yaml")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    data = cfg.get("data", {})
    astro_cfg = cfg.get("astro", {})
    wave_cfg = cfg.get("wave", {})
    model_cfg = cfg.get("model", {})

    ticker = args.ticker or data.get("ticker", "^GSPC")
    start = args.start or data.get("start", "2015-01-01")
    end = args.end or data.get("end")
    seq_len = args.seq_len or int(model_cfg.get("seq_len", 60))
    epochs = args.epochs or int(model_cfg.get("lstm", {}).get("epochs", 25))

    from stock_forecast.utils import device_info, set_seed
    set_seed(int(cfg.get("seed", 42)))
    print(f"[benchmark] device: {device_info()}")

    df = M.fetch_ohlc(ticker, start=start, end=end,
                      interval=data.get("interval", "1d"))
    print(f"[benchmark] {ticker}: {len(df)} bars")

    bodies = astro_cfg.get("bodies", ["SUN", "VENUS", "MARS", "JUPITER", "SATURN"])
    cal = A.calendar_index(df.index.min(), df.index.max())
    lons = A.longitudes(cal, bodies=bodies,
                        frame=astro_cfg.get("frame", "geocentric"))
    events = A.aspect_events(lons, A.EventSpec(
        aspects=astro_cfg.get("aspects_deg", [0, 90, 120, 180]),
        orb_deg=float(astro_cfg.get("orb_deg", 2.0)),
        min_separation_days=int(astro_cfg.get("min_separation_days", 3)),
    ))
    wave = W.composite_pressure(
        lons.index, events,
        tau_days=float(wave_cfg.get("tau_days", 30)),
        horizon_days=int(wave_cfg.get("horizon_days", 180)),
    )
    print(f"[benchmark] {len(events)} alignments feeding the wave")

    from stock_forecast.pipeline import run_benchmark_gui
    res = run_benchmark_gui(
        df=df, wave=wave, lons=lons, seq_len=seq_len, epochs=epochs,
        max_windows=int(args.windows), log=lambda m: print("[benchmark]", m),
    )

    out_dir = P.artifacts_dir()
    res["summary"].to_csv(os.path.join(out_dir, "benchmark_summary.csv"), index=False)
    res["detail"].to_csv(os.path.join(out_dir, "benchmark_detail.csv"), index=False)
    if res.get("predictions") is not None:
        res["predictions"].to_csv(os.path.join(out_dir, "benchmark_predictions.csv"))

    print()
    print("=== averaged over windows ===")
    print(res["summary"].to_string(index=False))
    print()
    print("Read DPA against 0.5, not against zero: 0.5 is what a coin flip")
    print("scores. The Persistence row is the real floor. A model that cannot")
    print("beat it has learned nothing worth keeping, however good its RMSE")
    print("looks, because RMSE on log returns is dominated by predicting")
    print("something close to the mean.")
    print(f"\nWrote results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
