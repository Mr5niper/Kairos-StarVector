# scripts/fetch_news_rss.py
import os
import argparse
import time
from datetime import datetime, timezone
from typing import List

import feedparser
import pandas as pd


DEFAULT_FEEDS = [
    # Reuters Markets/Business
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/marketsNews",
    # CNBC Markets
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    # MarketWatch Top Stories
    "https://www.marketwatch.com/feeds/topstories",
    # Yahoo Finance (general news RSS aggregator)
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"
]


def parse_entry_date(entry) -> datetime | None:
    """
    Extract published date from an RSS entry and return as naive date (UTC->local date).
    Returns a datetime.date (converted to local date) or None if not parseable.
    """
    # The `published_parsed` or `updated_parsed` fields are time.struct_time
    tm = None
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        tm = entry.published_parsed
    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
        tm = entry.updated_parsed

    if tm is None:
        # As fallback, try 'published' or 'updated' text
        text = getattr(entry, "published", None) or getattr(entry, "updated", None)
        if not text:
            return None
        try:
            # Let pandas parse a wide variety of datetime formats
            ts = pd.to_datetime(text, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.tz_convert("US/Eastern").date()
        except Exception:
            return None

    # Convert struct_time to UTC timestamp
    try:
        dt_utc = datetime(*tm[:6], tzinfo=timezone.utc)
        return dt_utc.astimezone(tz=None).date()  # convert to local, then take date
    except Exception:
        return None


def fetch_rss_headlines(feeds: List[str], out_csv: str, start_date: str | None = None, end_date: str | None = None):
    """
    Fetch headlines from a list of RSS feeds and save to CSV with columns: Date, Title
    Optional: filter by date range (YYYY-MM-DD).
    """
    rows = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
        except Exception as e:
            print(f"[RSS] Failed to fetch {url}: {e}")
            continue

        if not d.entries:
            print(f"[RSS] No entries found in {url}")
            continue

        for entry in d.entries:
            title = getattr(entry, "title", "").strip()
            if not title:
                continue

            dte = parse_entry_date(entry)
            if dte is None:
                continue

            # Optional date filtering
            if start_date:
                if dte < pd.to_datetime(start_date).date():
                    continue
            if end_date:
                if dte > pd.to_datetime(end_date).date():
                    continue

            rows.append({"Date": dte.isoformat(), "Title": title})

        # be kind to RSS sources
        time.sleep(0.2)

    if not rows:
        print("[RSS] No valid items parsed from the provided feeds.")
        return

    df = pd.DataFrame(rows).dropna()
    df = df.sort_values("Date")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[RSS] Wrote {len(df)} rows to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Fetch headlines from multiple RSS feeds and save to CSV (Date, Title).")
    ap.add_argument("--out", default="features/news_headlines.csv", help="output CSV path")
    ap.add_argument("--start", default=None, help="optional start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="optional end date (YYYY-MM-DD)")
    ap.add_argument("--feeds", nargs="*", default=DEFAULT_FEEDS, help="override RSS feed URLs")
    args = ap.parse_args()

    fetch_rss_headlines(args.feeds, args.out, start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()