"""
Compute 30-day feature trends for the top-K portfolio stocks.

For each candidate, loads its OHLCV history and computes the recent-30-day
trajectory of the key model features. Used by the predictions dashboard's
"特征趋势" section so the user can visually validate why the model picked
each stock — e.g. "is its turnover spiking? is RSI elevated? is it riding
above MA20?"

Returns a JSON-serialisable dict with per-stock time-series per feature.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'stock_data'

# Per-stock features (no cross-sectional dependency). These match the names
# used in the model so users see the actual signal channels.
FEATURE_FORMULAS = {
    'pct_chg':           'daily % return',
    'turnover_rate_f':   '换手率(自由流通) %',
    'vol_ratio_5':       '当日量比/5日均',
    'vol_ratio_20':      '当日量比/20日均',
    'amt_ratio_5':       '当日额比/5日均',
    'hl_ratio':          '日内振幅 (high-low)/pre_close',
    'overnight_gap':     '隔夜跳空 (open-pre_close)/pre_close',
    'lowershadow':       '下影线/pre_close',
    'rsi_14':            'RSI(14)',
    'momentum_5':        '5日累计对数收益',
    'momentum_20':       '20日累计对数收益',
    'close_ma_20_ratio': 'close / 20日均线',
    'close_ma_60_ratio': 'close / 60日均线',
    'vol_pct_20':        '20日 pct_chg 标准差',
    'up_limit_ratio':    '距涨停距离',
}


def _safe_div(a, b):
    return np.where(b != 0, a / b, np.nan)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock features on a single OHLCV DataFrame indexed by date."""
    df = df.sort_values('trade_date').copy()
    out = pd.DataFrame(index=df.index)

    out['trade_date'] = df['trade_date']
    out['close']      = df['close']
    out['pct_chg']    = df['pct_chg']

    # Turnover from daily_basic — load lazily; fall back to NaN
    # (For now we just compute features available from OHLCV directly.)

    pre = df['pre_close'].astype(float)
    high = df['high'].astype(float)
    low  = df['low'].astype(float)
    close = df['close'].astype(float)
    open_ = df['open'].astype(float)
    vol  = df['vol'].astype(float)

    out['hl_ratio']      = _safe_div(high - low, pre)
    out['overnight_gap'] = _safe_div(open_ - pre, pre)
    out['lowershadow']   = _safe_div(np.minimum(open_, close) - low, pre)
    log_ret = np.log(close / pre)
    out['momentum_5']  = pd.Series(log_ret).rolling(5).sum().to_numpy()
    out['momentum_20'] = pd.Series(log_ret).rolling(20).sum().to_numpy()
    pct = df['pct_chg']
    out['vol_pct_20'] = pct.rolling(20).std().to_numpy()

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    lossv = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / lossv.replace(0, np.nan)
    out['rsi_14'] = (100 - 100/(1+rs)).to_numpy()

    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    out['close_ma_20_ratio'] = (close / ma20).to_numpy()
    out['close_ma_60_ratio'] = (close / ma60).to_numpy()

    vma5  = vol.rolling(5).mean()
    vma20 = vol.rolling(20).mean()
    out['vol_ratio_5']  = (vol / vma5).to_numpy()
    out['vol_ratio_20'] = (vol / vma20).to_numpy()
    amt = df['amount'].astype(float)
    out['amt_ratio_5']  = (amt / amt.rolling(5).mean()).to_numpy()

    # up_limit_ratio: requires up_limit price; approximate via 10%-band rule
    out['up_limit_ratio'] = ((pre * 1.10 - close) / close).to_numpy()

    return out


def _load_turnover(ts_code: str, dates: List[str]) -> Optional[List[float]]:
    """Pull turnover_rate_f from per-date daily_basic files for the given dates."""
    out = []
    cache = {}
    for d in dates:
        if d not in cache:
            fp = DATA_DIR / 'daily_basic' / f'daily_basic_{d}.csv'
            if fp.exists():
                try:
                    df = pd.read_csv(fp, usecols=['ts_code', 'turnover_rate_f']).set_index('ts_code')
                    cache[d] = df
                except Exception:
                    cache[d] = None
            else:
                cache[d] = None
        c = cache[d]
        if c is not None and ts_code in c.index:
            v = c.loc[ts_code, 'turnover_rate_f']
            out.append(float(v) if pd.notna(v) else None)
        else:
            out.append(None)
    return out


def build_trends_for(ts_codes: List[str],
                       feature_date: str,
                       window_days: int = 30) -> List[dict]:
    """For each ts_code, return its 30-day feature trajectory ending at
    feature_date.

    Output rows: {ts_code, dates[], features: {pct_chg: [...], ...}}.
    """
    fd = pd.Timestamp(feature_date)
    rows = []
    for ts in ts_codes:
        code, suf = ts.split('.')
        sub = 'sh' if suf == 'SH' else 'sz'
        fp = DATA_DIR / sub / f'{code}.csv'
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, usecols=['trade_date','open','high','low',
                                           'close','pre_close','pct_chg','vol','amount'])
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df[df['trade_date'] <= fd].sort_values('trade_date')
            # Keep an extra 60 days for rolling windows, then crop to last 30
            full = df.tail(window_days + 80)
            feats = _compute_features(full)
            recent = feats.tail(window_days).reset_index(drop=True)

            dates_str = [d.strftime('%Y-%m-%d') for d in recent['trade_date']]
            dates_yyyymmdd = [d.strftime('%Y%m%d') for d in recent['trade_date']]

            # Fetch turnover from daily_basic for these specific dates
            turnover = _load_turnover(ts, dates_yyyymmdd)

            features_payload = {}
            for col in ['pct_chg','hl_ratio','overnight_gap','lowershadow',
                        'rsi_14','momentum_5','momentum_20','vol_pct_20',
                        'close_ma_20_ratio','close_ma_60_ratio',
                        'vol_ratio_5','vol_ratio_20','amt_ratio_5','up_limit_ratio']:
                if col in recent.columns:
                    s = recent[col].astype(float)
                    features_payload[col] = [None if (pd.isna(v) or not np.isfinite(v))
                                              else float(round(v, 4)) for v in s]
            features_payload['turnover_rate_f'] = turnover
            features_payload['close']           = [float(round(v, 4)) for v in recent['close']]

            rows.append({
                'ts_code':  ts,
                'dates':    dates_str,
                'features': features_payload,
            })
        except Exception as e:
            print(f"[feature_trends] {ts} failed: {e}")
            continue
    return rows


SLIM_FEATURES = ['close', 'pct_chg', 'turnover_rate_f', 'rsi_14', 'vol_ratio_20']
SLIM_FEATURE_LABELS = {
    'close':            '收盘价',
    'pct_chg':          '日涨跌幅 %',
    'turnover_rate_f':  '换手率(自由流通) %',
    'rsi_14':           'RSI(14)',
    'vol_ratio_20':     '量比/20日均',
}


def build_slim_trends_all(ts_codes: List[str], feature_date: str,
                           window_days: int = 30,
                           progress_every: int = 500) -> Dict[str, dict]:
    """Compact 30-day trend for every stock — used by dashboard dropdown.

    Returns: {ts_code: {dates, close, pct_chg, turnover_rate_f, rsi_14,
                         vol_ratio_20}} for ts_codes that have data.

    Optimization: load each daily_basic file once into a shared cache, so
    turnover lookups across thousands of stocks don't re-read the same files.
    """
    fd = pd.Timestamp(feature_date)
    out: Dict[str, dict] = {}

    db_cache: Dict[str, pd.Series] = {}
    db_dates_to_load = pd.date_range(end=fd, periods=int(window_days * 1.6), freq='B')
    for d in db_dates_to_load:
        ymd = d.strftime('%Y%m%d')
        fp = DATA_DIR / 'daily_basic' / f'daily_basic_{ymd}.csv'
        if fp.exists():
            try:
                df = pd.read_csv(fp, usecols=['ts_code', 'turnover_rate_f'])
                db_cache[ymd] = df.set_index('ts_code')['turnover_rate_f']
            except Exception:
                pass

    for i, ts in enumerate(ts_codes):
        if i % progress_every == 0 and i:
            print(f"    [{i}/{len(ts_codes)}] slim trends built")
        try:
            code, suf = ts.split('.')
        except ValueError:
            continue
        sub = 'sh' if suf == 'SH' else 'sz'
        fp = DATA_DIR / sub / f'{code}.csv'
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, usecols=['trade_date', 'open', 'high', 'low',
                                           'close', 'pre_close', 'pct_chg', 'vol'])
        except Exception:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df = df[df['trade_date'] <= fd].sort_values('trade_date')
        if len(df) < 5:
            continue
        full = df.tail(window_days + 30).copy()
        close = full['close'].astype(float)
        pre = full['pre_close'].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        vol = full['vol'].astype(float)
        vma20 = vol.rolling(20).mean()
        vr20 = vol / vma20

        recent = full.tail(window_days).reset_index(drop=True)
        recent['rsi_14'] = rsi.tail(window_days).values
        recent['vol_ratio_20'] = vr20.tail(window_days).values

        dates_yyyymmdd = [d.strftime('%Y%m%d') for d in recent['trade_date']]
        turnover = []
        for ymd in dates_yyyymmdd:
            s = db_cache.get(ymd)
            if s is not None and ts in s.index:
                v = s.loc[ts]
                turnover.append(float(round(v, 3)) if pd.notna(v) else None)
            else:
                turnover.append(None)

        def _ser(name):
            s = recent[name].astype(float)
            return [None if (pd.isna(v) or not np.isfinite(v))
                    else float(round(v, 4)) for v in s]

        out[ts] = {
            'dates': [d.strftime('%Y-%m-%d') for d in recent['trade_date']],
            'close': _ser('close'),
            'pct_chg': _ser('pct_chg'),
            'turnover_rate_f': turnover,
            'rsi_14': _ser('rsi_14'),
            'vol_ratio_20': _ser('vol_ratio_20'),
        }
    return out


def feature_descriptions() -> dict:
    """Dict for dashboard rendering — feature_name → Chinese description."""
    return dict(FEATURE_FORMULAS)


if __name__ == '__main__':
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument('--ts_codes', nargs='+', required=True)
    p.add_argument('--feature_date', default=pd.Timestamp.now().strftime('%Y-%m-%d'))
    args = p.parse_args()
    out = build_trends_for(args.ts_codes, args.feature_date)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:2000])
