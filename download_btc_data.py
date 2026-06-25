"""
Download BTC (or any Binance spot symbol) OHLCV history into a clean CSV that
load_ohlc()/run_on_csv() can read directly.

Source: Binance public data dumps at https://data.binance.vision  -- free, no
API key, no rate limits. Monthly zips for completed months, daily zips to fill
the current (incomplete) month. Stdlib only (urllib, zipfile, csv).

Usage
-----
    py download_btc_data.py                      # BTCUSDT 1h from 2018 -> now
    py download_btc_data.py --symbol BTCUSDT --interval 4h --start 2019
    py download_btc_data.py --symbol ETHUSDT --interval 1h --start 2020

Output: <SYMBOL>_<INTERVAL>.csv  with columns Date,Open,High,Low,Close,Volume
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sys
import urllib.error
import urllib.request
import zipfile

BASE = "https://data.binance.vision/data/spot"
KLINE_COLS = 12  # open_time,open,high,low,close,volume,close_time,...,ignore


def _month_range(start_year: int):
    today = dt.date.today()
    y, m = start_year, 1
    while (y, m) <= (today.year, today.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1; y += 1


def _fetch_zip(url: str):
    """Return the bytes of the single CSV inside a Binance .zip, or None on 404."""
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        return z.read(name)


def _to_epoch_seconds(ot: int) -> int:
    """Normalize a Binance open_time to epoch SECONDS.

    Binance dumps use seconds (~1e9), milliseconds (~1e12), or — in newer
    2025+ files — microseconds (~1e15). Pick the unit by magnitude.
    """
    if ot > 1e14:        # microseconds
        return ot // 1_000_000
    if ot > 1e11:        # milliseconds
        return ot // 1_000
    return ot            # seconds


def _parse_rows(csv_bytes: bytes, rows: dict):
    """Parse a Binance kline CSV into rows[open_time_seconds] = [o,h,l,c,v]."""
    text = io.TextIOWrapper(io.BytesIO(csv_bytes), encoding="utf-8")
    for r in csv.reader(text):
        if len(r) < 6:
            continue
        try:
            ot = int(r[0])               # open_time; header row raises -> skipped
        except ValueError:
            continue
        rows[_to_epoch_seconds(ot)] = [r[1], r[2], r[3], r[4], r[5]]


def download(symbol: str, interval: str, start_year: int, out_path: str):
    rows: dict = {}
    months = list(_month_range(start_year))
    print(f"downloading {symbol} {interval} from {start_year} "
          f"({len(months)} months) ...")
    for i, (y, m) in enumerate(months):
        ym = f"{y}-{m:02d}"
        url = f"{BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        blob = _fetch_zip(url)
        if blob is not None:
            _parse_rows(blob, rows)
            tag = "monthly"
        else:
            # monthly not published yet (current month): fall back to daily dumps
            tag = "daily"
            days = 31
            for d in range(1, days + 1):
                try:
                    day = dt.date(y, m, d)
                except ValueError:
                    break
                if day > dt.date.today():
                    break
                durl = (f"{BASE}/daily/klines/{symbol}/{interval}/"
                        f"{symbol}-{interval}-{day.isoformat()}.zip")
                dblob = _fetch_zip(durl)
                if dblob is not None:
                    _parse_rows(dblob, rows)
        print(f"  [{i+1:>3}/{len(months)}] {ym} ({tag})  rows so far: {len(rows):,}",
              end="\r", flush=True)
    print()

    if not rows:
        sys.exit("no data downloaded -- check symbol/interval (and network/geo).")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        for ot in sorted(rows):
            ts = dt.datetime.utcfromtimestamp(ot)
            o, h, l, c, v = rows[ot]
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), o, h, l, c, v])

    first = dt.datetime.utcfromtimestamp(min(rows))
    last = dt.datetime.utcfromtimestamp(max(rows))
    print(f"wrote {out_path}: {len(rows):,} bars  ({first} -> {last} UTC)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"{a.symbol}_{a.interval}.csv"
    download(a.symbol, a.interval, a.start, out)
