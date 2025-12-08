# scripts/fetch_news_yf.py
import os
import argparse
import pandas as pd
import yfinance as yf

def safe_to_local_date(news_item: dict, tz: str = "US/Eastern"):
    """
    Robustly extract a local date from a yfinance news item.
    Tries multiple timestamp fields and formats safely.
    Returns ISO date string (YYYY-MM-DD) or None if it cannot parse.
    """
    # Possible time fields found in the wild
    candidates = [
        "providerPublishTime",      # epoch seconds (int)
        "providerPublishTimeEpoch", # some feeds use this
        "publishTime",              # ISO string or epoch
        "published_at",             # ISO string
        "time_published",           # ISO string from some sources
    ]
    ts_val = None
    for key in candidates:
        if key in news_item and news_item[key]:
            ts_val = news_item[key]
            break

    if ts_val is None:
        return None

    try:
        # Case 1: numeric epoch seconds
        if isinstance(ts_val, (int, float)):
            ts_utc = pd.Timestamp.utcfromtimestamp(ts_val).tz_localize("UTC")
            return ts_utc.tz_convert(tz).date().isoformat()
        # Case 2: parseable datetime string
        if isinstance(ts_val, str):
            ts_utc = pd.to_datetime(ts_val, utc=True, errors="coerce")
            if pd.isna(ts_utc):
                return None
            return ts_utc.tz_convert(tz).date().isoformat()
    except Exception:
        return None

    return None

def fetch_headlines_yf(ticker: str, out_csv: str, tz: str = "US/Eastern"):
    """
    Fetch recent headlines from yfinance (limited history) and write a CSV with columns:
    Date (YYYY-MM-DD), Title
    """
    tk = yf.Ticker(ticker)
    news = tk.news or []
    rows = []
    skipped = 0

    for i, n in enumerate(news):
        # Title fallback keys
        title = (n.get("title") or n.get("headline") or "").strip()
        date_str = safe_to_local_date(n, tz=tz)

        if not title or date_str is None:
            skipped += 1
            continue

        rows.append({"Date": date_str, "Title": title})

    df = pd.DataFrame(rows)
    if not df.empty:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        # We may have multiple headlines per date; that's fine for our pipeline
        df.sort_values("Date").to_csv(out_csv, index=False)
        print(f"[fetch_news_yf] Wrote {len(df)} rows to {out_csv} (skipped {skipped})")
    else:
        print("[fetch_news_yf] No valid news parsed from yfinance.")
        print("Provide your own headlines CSV with columns: Date, Title")

def main():
    ap = argparse.ArgumentParser(description="Fetch recent headlines via yfinance and save to CSV (Date, Title).")
    ap.add_argument("--ticker", required=True, help="e.g., ^GSPC, AAPL")
    ap.add_argument("--out", default="features/news_headlines.csv", help="output CSV path")
    ap.add_argument("--tz", default="US/Eastern", help="local timezone for dates (pytz name)")
    args = ap.parse_args()

    fetch_headlines_yf(args.ticker, args.out, tz=args.tz)

if __name__ == "__main__":
    main()