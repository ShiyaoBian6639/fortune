"""Download auxiliary Binance data for BTC price prediction.

Fetches:
  - Spot klines for ETH/BNB/SOL/XRP (correlated majors)
  - Futures klines: BTCUSDT perp, BTCDOMUSDT (dominance perp)
  - Funding rate history for BTCUSDT perp (full history)
  - Open interest history (last 30 days; Binance limit)
  - Long/short ratios: top accounts, top positions, global accounts (last 30d)
  - Taker buy/sell volume ratio (last 30d)

Usage:
    ./venv/Scripts/python -m btc.download_aux
    ./venv/Scripts/python -m btc.download_aux --years 5 --interval 1d
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SPOT_BASE = "https://api.binance.com/api/v3"
FAPI_BASE = "https://fapi.binance.com/fapi/v1"
FUTURES_DATA = "https://fapi.binance.com/futures/data"

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "trades",
              "taker_buy_base", "taker_buy_quote", "ignore"]

ALT_SYMBOLS = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
DAY_MS = 86_400_000


def _get(session, url, params, attempts=5):
    for i in range(attempts):
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 2 ** i
            print(f"    retry {i + 1}/{attempts} after {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError(f"GET failed: {url} {params}")


def _paged_klines(session, base, symbol, interval, start_ms, end_ms):
    rows, cursor = [], start_ms
    step = INTERVAL_MS[interval] * 1000
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": cursor, "endTime": min(cursor + step, end_ms),
                  "limit": 1000}
        batch = _get(session, f"{base}/klines", params)
        if not batch:
            cursor += step
            continue
        rows.extend(batch)
        cursor = batch[-1][0] + INTERVAL_MS[interval]
        time.sleep(0.15)
    return rows


def klines_df(rows):
    df = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns=["ignore"])
    nums = ["open", "high", "low", "close", "volume", "quote_volume",
            "taker_buy_base", "taker_buy_quote"]
    df[nums] = df[nums].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def fetch_funding_rate(session, symbol, start_ms, end_ms):
    rows, cursor = [], start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        batch = _get(session, f"{FAPI_BASE}/fundingRate", params)
        if not batch:
            break
        rows.extend(batch)
        last_t = batch[-1]["fundingTime"]
        if last_t <= cursor:
            break
        cursor = last_t + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    if "markPrice" in df.columns:
        df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")
    return df.drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True)


def fetch_window_30d(session, endpoint, symbol, period, days=30, extra_keys=None):
    """Generic fetcher for 30-day-limited futures-data endpoints."""
    end = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start = end - days * DAY_MS
    rows, cursor = [], start
    pms = INTERVAL_MS[period]
    limit = 500
    while cursor < end:
        params = {"symbol": symbol, "period": period,
                  "startTime": cursor,
                  "endTime": min(cursor + pms * limit, end),
                  "limit": limit}
        batch = _get(session, f"{FUTURES_DATA}/{endpoint}", params)
        if not batch:
            cursor += pms * limit
            continue
        rows.extend(batch)
        cursor = batch[-1]["timestamp"] + pms
        if len(batch) < limit:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in df.columns:
        if c in {"symbol", "timestamp"}:
            continue
        try:
            df[c] = df[c].astype(float)
        except (ValueError, TypeError):
            pass
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--out_dir", default="btc_data")
    parser.add_argument("--futures_period", default="1h",
                        help="period for OI/long-short/taker (5m,15m,30m,1h,2h,4h,6h,12h,1d)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    end = datetime.now(tz=timezone.utc)
    start = end - pd.Timedelta(days=int(args.years * 365.25))
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    sess = requests.Session()

    print(f"Range: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')} UTC\n")

    for sym in ALT_SYMBOLS:
        print(f"[spot {sym} {args.interval}]")
        rows = _paged_klines(sess, SPOT_BASE, sym, args.interval, start_ms, end_ms)
        df = klines_df(rows)
        df.to_csv(out / f"{sym}_{args.interval}.csv", index=False)
        print(f"  -> {len(df)} rows\n")

    print(f"[futures BTCUSDT perp {args.interval}]")
    try:
        rows = _paged_klines(sess, FAPI_BASE, "BTCUSDT", args.interval, start_ms, end_ms)
        df = klines_df(rows)
        df.to_csv(out / f"BTCUSDT_perp_{args.interval}.csv", index=False)
        print(f"  -> {len(df)} rows\n")
    except Exception as e:
        print(f"  skipped: {e}\n")

    print(f"[futures BTCDOMUSDT {args.interval}]")
    try:
        rows = _paged_klines(sess, FAPI_BASE, "BTCDOMUSDT", args.interval, start_ms, end_ms)
        df = klines_df(rows)
        df.to_csv(out / f"BTCDOMUSDT_{args.interval}.csv", index=False)
        print(f"  -> {len(df)} rows\n")
    except Exception as e:
        print(f"  skipped: {e}\n")

    print(f"[funding rate BTCUSDT — full history]")
    df = fetch_funding_rate(sess, "BTCUSDT", start_ms, end_ms)
    df.to_csv(out / "funding_rate_BTCUSDT.csv", index=False)
    print(f"  -> {len(df)} rows\n")

    p = args.futures_period
    print(f"[OI hist BTCUSDT @ {p} — last 30d]")
    df = fetch_window_30d(sess, "openInterestHist", "BTCUSDT", p)
    df.to_csv(out / f"open_interest_BTCUSDT_{p}.csv", index=False)
    print(f"  -> {len(df)} rows\n")

    print(f"[long/short ratios BTCUSDT @ {p} — last 30d]")
    parts = []
    for kind, ep in [("top_account", "topLongShortAccountRatio"),
                     ("top_position", "topLongShortPositionRatio"),
                     ("global_account", "globalLongShortAccountRatio")]:
        d = fetch_window_30d(sess, ep, "BTCUSDT", p)
        if not d.empty:
            d["kind"] = kind
            parts.append(d)
    if parts:
        all_ls = pd.concat(parts, ignore_index=True)
        all_ls.to_csv(out / f"long_short_BTCUSDT_{p}.csv", index=False)
        print(f"  -> {len(all_ls)} rows (3 kinds)\n")

    print(f"[taker buy/sell BTCUSDT @ {p} — last 30d]")
    df = fetch_window_30d(sess, "takerlongshortRatio", "BTCUSDT", p)
    df.to_csv(out / f"taker_BTCUSDT_{p}.csv", index=False)
    print(f"  -> {len(df)} rows\n")

    print("Done.")


if __name__ == "__main__":
    main()
