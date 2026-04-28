"""Download historical BTC OHLCV klines from Binance public REST API.

Usage:
    ./venv/Scripts/python -m btc.download_klines
    ./venv/Scripts/python -m btc.download_klines --symbol BTCUSDT --interval 1d --years 5
    ./venv/Scripts/python -m btc.download_klines --interval 1h --years 5
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    rows = []
    cursor = start_ms
    step_ms = INTERVAL_MS[interval] * MAX_LIMIT
    session = requests.Session()

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": min(cursor + step_ms, end_ms),
            "limit": MAX_LIMIT,
        }
        for attempt in range(5):
            try:
                resp = session.get(BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                batch = resp.json()
                break
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"  retry {attempt + 1}/5 after {wait}s: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to fetch {symbol} {interval} at cursor={cursor}")

        if not batch:
            cursor += step_ms
            continue

        rows.extend(batch)
        last_open = batch[-1][0]
        cursor = last_open + INTERVAL_MS[interval]
        ts = datetime.fromtimestamp(last_open / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"  fetched {len(batch):>4} rows up to {ts} UTC (total={len(rows)})")
        time.sleep(0.15)

    return rows


def klines_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df = df.drop(columns=["ignore"])
    num_cols = ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[num_cols] = df[num_cols].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1d", choices=list(INTERVAL_MS.keys()))
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--out_dir", default="btc_data")
    args = parser.parse_args()

    end = datetime.now(tz=timezone.utc)
    start = end - pd.Timedelta(days=int(args.years * 365.25))
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print(f"Symbol:   {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Range:    {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')} UTC")

    rows = fetch_klines(args.symbol, args.interval, start_ms, end_ms)
    df = klines_to_df(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.symbol}_{args.interval}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"First: {df['open_time'].iloc[0]}")
    print(f"Last:  {df['open_time'].iloc[-1]}")


if __name__ == "__main__":
    main()
