#!/usr/bin/env python3
"""
Fetch market headlines from RSS into a CSV of Date,Title.

    python scripts/fetch_news_rss.py --out features/news_headlines.csv

Only needed for the optional event_sentiment feature. Requires feedparser
from requirements-ml.txt.

A warning about coverage, because the original version of this script could
not work. Its feed list pointed at feeds.reuters.com, which Reuters shut
down in 2020, so every fetch returned nothing and the pipeline fell back to
zero-filled sentiment while reporting success. Beyond that, RSS by design
serves only the current page of items: expect the last few days, not the
last ten years. For a real sentiment feature you need a historical news
archive, supplied as your own CSV with Date and Title columns.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

DEFAULT_FEEDS = [
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]


def entry_date(entry, tz: str = "US/Eastern") -> Optional[str]:
    tm = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if tm is not None:
        try:
            dt = datetime(*tm[:6], tzinfo=timezone.utc)
            return pd.Timestamp(dt).tz_convert(tz).date().isoformat()
        except Exception:
            return None
    text = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_convert(tz).date().isoformat()


def fetch(feeds: List[str], out_csv: str, start=None, end=None,
          tz: str = "US/Eastern") -> int:
    try:
        import feedparser
    except Exception:
        print("[rss] feedparser is not installed.")
        print("[rss]     pip install -r requirements-ml.txt")
        return 1

    rows, dead = [], []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:
            print(f"[rss] error fetching {url}: {exc}")
            dead.append(url)
            continue
        if not parsed.entries:
            print(f"[rss] no entries from {url}")
            dead.append(url)
            continue
        kept = 0
        for entry in parsed.entries:
            title = (getattr(entry, "title", "") or "").strip()
            if not title:
                continue
            date_str = entry_date(entry, tz)
            if not date_str:
                continue
            if start and date_str < pd.Timestamp(start).date().isoformat():
                continue
            if end and date_str > pd.Timestamp(end).date().isoformat():
                continue
            rows.append({"Date": date_str, "Title": title})
            kept += 1
        print(f"[rss] {kept:4d} headlines from {url}")
        time.sleep(0.3)

    if dead:
        print(f"\n[rss] {len(dead)} feed(s) returned nothing. Feed URLs change "
              f"often; pass working ones with --feeds.")

    if not rows:
        print("[rss] nothing collected. No file written.")
        print("[rss] Supply your own CSV with Date,Title columns instead.")
        return 1

    df = pd.DataFrame(rows).drop_duplicates().sort_values("Date")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[rss] wrote {len(df)} rows to {out_csv}")
    print(f"[rss] date span: {df['Date'].min()} to {df['Date'].max()}")
    span = (pd.Timestamp(df["Date"].max()) - pd.Timestamp(df["Date"].min())).days
    if span < 30:
        print(f"[rss] Note: only {span} days of coverage. RSS serves the "
              f"current page of items, so this is expected and is not enough "
              f"for the sentiment feature to matter in a multi-year backtest.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch headlines from RSS feeds.")
    ap.add_argument("--out", default="features/news_headlines.csv")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--tz", default="US/Eastern")
    ap.add_argument("--feeds", nargs="*", default=DEFAULT_FEEDS)
    args = ap.parse_args()
    return fetch(args.feeds, args.out, args.start, args.end, args.tz)


if __name__ == "__main__":
    raise SystemExit(main())
