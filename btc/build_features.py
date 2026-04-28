"""Build the BTC feature matrix from raw klines + auxiliary Binance data.

Combines BTCUSDT spot with correlated majors, futures perp + dominance,
funding/OI/long-short/taker derivatives data, and engineers technical and
market-wide factors. Saves a single CSV ready for modeling.

Usage:
    ./venv/Scripts/python -m btc.build_features
    ./venv/Scripts/python -m btc.build_features --interval 1d
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from btc import factors as F

ALT_SYMBOLS = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]


def load_klines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, format="mixed")
    return df.set_index("open_time").sort_index()


def add_technicals(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    p = prefix
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    log_c = np.log(close)
    for h in (1, 3, 5, 10, 20):
        df[f"{p}log_ret_{h}"] = log_c.diff(h)
    df[f"{p}rsi_14"] = F.rsi(close, 14)
    df[f"{p}rsi_21"] = F.rsi(close, 21)
    macd_l, macd_s, macd_h = F.macd(close)
    df[f"{p}macd"] = macd_l
    df[f"{p}macd_signal"] = macd_s
    df[f"{p}macd_hist"] = macd_h
    _, _, _, pctb, bw = F.bollinger(close)
    df[f"{p}bb_pctb"] = pctb
    df[f"{p}bb_bandwidth"] = bw
    df[f"{p}atr_14"] = F.atr(high, low, close, 14)
    df[f"{p}atr_pct"] = df[f"{p}atr_14"] / close
    k, d = F.stoch(high, low, close)
    df[f"{p}stoch_k"] = k
    df[f"{p}stoch_d"] = d
    a, pdi, mdi = F.adx(high, low, close, 14)
    df[f"{p}adx"] = a
    df[f"{p}plus_di"] = pdi
    df[f"{p}minus_di"] = mdi
    df[f"{p}cci_20"] = F.cci(high, low, close, 20)
    df[f"{p}wr_14"] = F.williams_r(high, low, close, 14)
    df[f"{p}obv_diff"] = F.obv_diff(close, vol)
    df[f"{p}rv_20"] = F.realized_vol(close, 20)
    df[f"{p}rv_60"] = F.realized_vol(close, 60)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--data_dir", default="btc_data")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    interval = args.interval

    btc = load_klines(data_dir / f"BTCUSDT_{interval}.csv")
    btc = btc[["open", "high", "low", "close", "volume", "quote_volume", "trades",
               "taker_buy_base", "taker_buy_quote"]].copy()
    btc = add_technicals(btc, prefix="btc_")
    btc["btc_taker_buy_ratio"] = btc["taker_buy_base"] / btc["volume"].replace(0, np.nan)
    btc["btc_avg_trade_size"] = btc["quote_volume"] / btc["trades"].replace(0, np.nan)

    print(f"[patterns] detecting W-bottom / M-top on {len(btc)} bars...")
    w, m = F.detect_double_bottom_top(btc["close"], window=60)
    btc["w_bottom"] = w
    btc["m_top"] = m
    btc["w_bottom_30d"] = btc["w_bottom"].rolling(30).sum()
    btc["m_top_30d"] = btc["m_top"].rolling(30).sum()

    intraday_path = data_dir / "BTCUSDT_5m.csv"
    if intraday_path.exists() and interval == "1d":
        print(f"[intraday] aggregating 5m -> daily features...")
        df5 = pd.read_csv(intraday_path)
        df5["open_time"] = pd.to_datetime(df5["open_time"], utc=True, format="mixed")
        for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
            df5[c] = pd.to_numeric(df5[c], errors="coerce")
        intraday = F.aggregate_5m_to_daily(df5)
        intraday = intraday.reindex(btc.index)
        for col in intraday.columns:
            btc[col] = intraday[col]
        print(f"  added {len(intraday.columns)} intraday features")

    alt_closes = {}
    for sym in ALT_SYMBOLS:
        path = data_dir / f"{sym}_{interval}.csv"
        if not path.exists():
            print(f"  skip {sym}: not found")
            continue
        a = load_klines(path)
        s = sym.lower().replace("usdt", "")
        a_close = a["close"].reindex(btc.index)
        alt_closes[s] = a_close
        log_a = np.log(a_close)
        btc[f"{s}_log_ret_1"] = log_a.diff()
        btc[f"{s}_log_ret_5"] = log_a.diff(5)
        btc[f"{s}_log_ret_20"] = log_a.diff(20)
        btc[f"{s}_rsi_14"] = F.rsi(a_close, 14)
        btc[f"{s}_rv_20"] = F.realized_vol(a_close, 20)

    if "eth" in alt_closes:
        btc["eth_btc_ratio"] = alt_closes["eth"] / btc["close"]
        btc["eth_btc_ratio_mom20"] = btc["eth_btc_ratio"].pct_change(20)
        btc["btc_eth_corr_30"] = btc["btc_log_ret_1"].rolling(30).corr(btc["eth_log_ret_1"])

    alt_ret_cols = [f"{s}_log_ret_1" for s in alt_closes if f"{s}_log_ret_1" in btc.columns]
    if alt_ret_cols:
        btc["alts_mean_ret_1"] = btc[alt_ret_cols].mean(axis=1)
        btc["btc_alts_spread_1"] = btc["btc_log_ret_1"] - btc["alts_mean_ret_1"]
        btc["alts_mean_ret_5"] = btc[[f"{s}_log_ret_5" for s in alt_closes]].mean(axis=1)
        btc["btc_alts_spread_5"] = btc["btc_log_ret_5"] - btc["alts_mean_ret_5"]
        btc["alts_disp_1"] = btc[alt_ret_cols].std(axis=1)

    dom_path = data_dir / f"BTCDOMUSDT_{interval}.csv"
    if dom_path.exists():
        dom = load_klines(dom_path)["close"].reindex(btc.index)
        btc["btcdom_close"] = dom
        btc["btcdom_log_ret_1"] = np.log(dom).diff()
        btc["btcdom_log_ret_5"] = np.log(dom).diff(5)
        btc["btcdom_rsi_14"] = F.rsi(dom, 14)
        btc["btcdom_mom20"] = dom.pct_change(20)

    perp_path = data_dir / f"BTCUSDT_perp_{interval}.csv"
    if perp_path.exists():
        perp = load_klines(perp_path)
        perp = perp.reindex(btc.index)
        btc["perp_close"] = perp["close"]
        btc["perp_volume"] = perp["volume"]
        btc["perp_basis"] = (perp["close"] - btc["close"]) / btc["close"]
        btc["perp_basis_z20"] = ((btc["perp_basis"] - btc["perp_basis"].rolling(20).mean())
                                 / btc["perp_basis"].rolling(20).std(ddof=0))
        btc["perp_taker_buy_ratio"] = perp["taker_buy_base"] / perp["volume"].replace(0, np.nan)
        btc["perp_spot_vol_ratio"] = perp["volume"] / btc["volume"].replace(0, np.nan)

    fpath = data_dir / "funding_rate_BTCUSDT.csv"
    if fpath.exists():
        fr = pd.read_csv(fpath)
        fr["fundingTime"] = pd.to_datetime(fr["fundingTime"], utc=True, format="mixed")
        fr = fr.set_index("fundingTime").sort_index()
        if not fr.empty:
            rule = "1D" if interval == "1d" else interval
            agg = fr["fundingRate"].resample(rule).mean()
            agg = agg.reindex(btc.index, method="ffill")
            btc["funding_rate"] = agg
            btc["funding_rate_3d"] = btc["funding_rate"].rolling(3).mean()
            btc["funding_rate_7d"] = btc["funding_rate"].rolling(7).mean()
            mu = btc["funding_rate"].rolling(30).mean()
            sd = btc["funding_rate"].rolling(30).std(ddof=0)
            btc["funding_rate_z30"] = (btc["funding_rate"] - mu) / sd.replace(0, np.nan)
            btc["funding_cumul_30d"] = btc["funding_rate"].rolling(30).sum()

    for fname, base in [("open_interest_BTCUSDT_1h.csv", "oi"),
                        ("taker_BTCUSDT_1h.csv", "taker")]:
        path = data_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        df = df.set_index("timestamp").sort_index()
        rule = "1D" if interval == "1d" else interval
        if base == "oi" and "sumOpenInterest" in df.columns:
            d = df["sumOpenInterest"].resample(rule).mean()
            btc["oi_mean"] = d.reindex(btc.index)
            btc["oi_change_1"] = btc["oi_mean"].pct_change()
            btc["oi_change_5"] = btc["oi_mean"].pct_change(5)
        elif base == "taker" and "buySellRatio" in df.columns:
            d = df["buySellRatio"].resample(rule).mean()
            btc["taker_buysell_ratio"] = d.reindex(btc.index)

    lspath = data_dir / "long_short_BTCUSDT_1h.csv"
    if lspath.exists():
        ls = pd.read_csv(lspath)
        ls["timestamp"] = pd.to_datetime(ls["timestamp"], utc=True, format="mixed")
        rule = "1D" if interval == "1d" else interval
        for kind in ls["kind"].unique():
            sub = ls[ls["kind"] == kind].set_index("timestamp").sort_index()
            if "longShortRatio" in sub.columns:
                d = sub["longShortRatio"].astype(float).resample(rule).mean()
                btc[f"ls_{kind}_ratio"] = d.reindex(btc.index)

    out_path = Path(args.out) if args.out else (data_dir / f"features_BTCUSDT_{interval}.csv")
    btc.to_csv(out_path)

    print(f"\nSaved {len(btc)} rows x {btc.shape[1]} cols -> {out_path}")
    print(f"Range: {btc.index[0]} -> {btc.index[-1]}")
    print(f"\nW-bottom signals: {int(btc['w_bottom'].sum())}, "
          f"M-top signals: {int(btc['m_top'].sum())}")
    cov = btc.notna().mean().sort_values(ascending=False) * 100
    print(f"\nFeature coverage (% non-null), bottom 10:")
    print(cov.tail(10).round(1).to_string())


if __name__ == "__main__":
    main()
