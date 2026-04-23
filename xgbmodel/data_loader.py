"""
Build a (stock, date) panel from the tushare stock_data directory.

Exposes build_panel(cfg) -> pandas.DataFrame with columns:
    ts_code, trade_date, <feature columns>, target

Data sources merged:
  - sh/{CODE}.csv, sz/{CODE}.csv        — per-stock OHLCV (required, base table)
  - daily_basic/daily_basic_{YMD}.csv   — pe/pb/turnover/mv, one file per trade date
  - moneyflow/moneyflow_{YMD}.csv       — sm/md/lg/elg net flows
  - stk_limit/stk_limit_{YMD}.csv       — up/down limit prices
  - index/index_dailybasic/<code>.csv   — macro indices (csi300, csi500, sse50, szse)
  - fina_indicator/<code>_<EX>.csv      — quarterly fundamentals (forward-fill to trade day)
  - stock_sectors.csv                   — SW sector/industry (label-encoded as static cat)

Target: raw next-day pct_chg (target_mode='raw') or next-day excess return vs
CSI300 (target_mode='excess').
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import features as F
from .cross_section import add_cross_section_features
from .config import (
    DAILY_BASIC_RAW_KEEP, FINA_COLUMNS, INDEX_CODES, MONEYFLOW_TIERS,
    GLOBAL_INDEX_CODES, IDX_FACTOR_COLUMNS, IDX_FACTOR_CODE,
)


# ─── Per-stock OHLCV loading ─────────────────────────────────────────────────

def _list_stock_files(data_dir: str, max_stocks: int = 0) -> List[Tuple[str, str]]:
    """Return list of (ts_code, path) for all stocks in sh/ + sz/."""
    out: List[Tuple[str, str]] = []
    for sub, suffix in [('sh', '.SH'), ('sz', '.SZ')]:
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.csv'):
                continue
            bare = fname[:-4]
            out.append((f"{bare}{suffix}", os.path.join(d, fname)))
    if max_stocks and max_stocks > 0:
        # Sample evenly across both exchanges (head of each)
        sh = [x for x in out if x[0].endswith('.SH')][: max_stocks // 2]
        sz = [x for x in out if x[0].endswith('.SZ')][: max_stocks - len(sh)]
        out = sh + sz
    return out


def _load_stock_csv(path: str, min_rows: int) -> Optional[pd.DataFrame]:
    """Read a single per-stock CSV; return None if too short/invalid."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if len(df) < min_rows:
        return None
    needed = {'ts_code', 'trade_date', 'open', 'high', 'low', 'close',
              'pre_close', 'pct_chg', 'vol', 'amount'}
    if not needed.issubset(df.columns):
        return None
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), errors='coerce')
    df = df.dropna(subset=['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    for c in ['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    df = df.dropna(subset=['close', 'pre_close', 'pct_chg']).reset_index(drop=True)
    # Drop duplicated dates (tushare sometimes re-emits corrections)
    df = df.drop_duplicates(subset=['trade_date'], keep='last').reset_index(drop=True)
    return df if len(df) >= min_rows else None


# ─── Cross-sectional daily files ────────────────────────────────────────────

def _load_cross_sectional_dir(
    dir_path: str,
    prefix: str,
    dtype_cols: List[str],
) -> pd.DataFrame:
    """
    Read every `<prefix>_YYYYMMDD.csv` in dir_path and concatenate.
    Returns a long DataFrame with ts_code, trade_date, and dtype_cols.
    """
    if not os.path.isdir(dir_path):
        print(f"  WARN: {dir_path} not found, skipping")
        return pd.DataFrame(columns=['ts_code', 'trade_date'] + dtype_cols)

    frames = []
    for fname in sorted(os.listdir(dir_path)):
        if not (fname.startswith(prefix) and fname.endswith('.csv')):
            continue
        path = os.path.join(dir_path, fname)
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except Exception:
            continue
        keep = ['ts_code', 'trade_date'] + [c for c in dtype_cols if c in df.columns]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=['ts_code', 'trade_date'] + dtype_cols)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out['trade_date'] = pd.to_datetime(out['trade_date'].astype(str), errors='coerce')
    out = out.dropna(subset=['trade_date', 'ts_code']).reset_index(drop=True)
    for c in dtype_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').astype('float32')
    return out


def load_daily_basic(data_dir: str) -> pd.DataFrame:
    return _load_cross_sectional_dir(
        os.path.join(data_dir, 'daily_basic'),
        'daily_basic_',
        DAILY_BASIC_RAW_KEEP,
    )


def load_moneyflow(data_dir: str) -> pd.DataFrame:
    raw_cols = []
    for tier in MONEYFLOW_TIERS:
        raw_cols += [f'buy_{tier}_amount', f'sell_{tier}_amount']
    raw_cols += ['net_mf_amount']
    df = _load_cross_sectional_dir(
        os.path.join(data_dir, 'moneyflow'), 'moneyflow_', raw_cols,
    )
    if df.empty:
        return df
    # Compute net flow per tier (amount terms, in 千元)
    for tier in MONEYFLOW_TIERS:
        df[f'net_{tier}_amount'] = (df[f'buy_{tier}_amount']
                                    - df[f'sell_{tier}_amount']).astype('float32')
    keep = ['ts_code', 'trade_date', 'net_mf_amount'] + \
           [f'net_{tier}_amount' for tier in MONEYFLOW_TIERS]
    return df[keep]


def load_stk_limit(data_dir: str) -> pd.DataFrame:
    df = _load_cross_sectional_dir(
        os.path.join(data_dir, 'stk_limit'), 'stk_limit_',
        ['up_limit', 'down_limit'],
    )
    return df


# ─── Index daily basics (macro) ─────────────────────────────────────────────

def load_index_panel(data_dir: str) -> pd.DataFrame:
    """Build a wide macro DataFrame: trade_date × per-index cols.

    For each index in INDEX_CODES we keep pct_chg + turnover + pe_ttm, renamed
    with the index alias prefix (e.g. csi300_pct_chg).
    """
    idx_dir = os.path.join(data_dir, 'index', 'index_dailybasic')
    per_idx: List[pd.DataFrame] = []

    for alias, bare in INDEX_CODES.items():
        path = os.path.join(idx_dir, f"{bare}.csv")
        if not os.path.exists(path):
            continue
        db = pd.read_csv(path, encoding='utf-8-sig')
        db['trade_date'] = pd.to_datetime(db['trade_date'].astype(str), errors='coerce')
        db = db.dropna(subset=['trade_date']).sort_values('trade_date')
        db = db.drop_duplicates(subset=['trade_date'], keep='last').reset_index(drop=True)

        # Also compute index pct_chg from total_mv — but tushare already has it
        # in the sh/<bare>.csv or sz/<bare>.csv base tables. We'll pull pct_chg
        # from the underlying index price CSV if available.
        kept = db[['trade_date', 'turnover_rate', 'pe_ttm', 'pb']].copy()
        kept.columns = ['trade_date',
                        f'{alias}_turnover',
                        f'{alias}_pe_ttm',
                        f'{alias}_pb']
        per_idx.append(kept)

        # Join in pct_chg from the raw price CSV (sh/ or sz/)
        for sub in ('sh', 'sz'):
            price_path = os.path.join(data_dir, sub, f"{bare.split('_')[0]}.csv")
            if os.path.exists(price_path):
                try:
                    p = pd.read_csv(price_path, usecols=['trade_date', 'pct_chg'])
                    p['trade_date'] = pd.to_datetime(p['trade_date'].astype(str), errors='coerce')
                    p = p.dropna(subset=['trade_date']).sort_values('trade_date')
                    p = p.drop_duplicates(subset=['trade_date'], keep='last')
                    p.columns = ['trade_date', f'{alias}_pct_chg']
                    per_idx.append(p)
                    break
                except Exception:
                    pass

    if not per_idx:
        return pd.DataFrame(columns=['trade_date'])

    out = per_idx[0]
    for df in per_idx[1:]:
        out = out.merge(df, on='trade_date', how='outer')
    out = out.sort_values('trade_date').reset_index(drop=True)

    # Rolling 20-day volatility of CSI300 as market-stress feature
    if 'csi300_pct_chg' in out.columns:
        out['csi300_vol_20'] = out['csi300_pct_chg'].rolling(20).std().astype('float32')

    for c in out.columns:
        if c != 'trade_date':
            out[c] = pd.to_numeric(out[c], errors='coerce').astype('float32')
    return out


# ─── Block trades (one file per date, aggregate per ts_code×date) ───────────

def load_block_trade_agg(data_dir: str) -> pd.DataFrame:
    """Aggregate block trades per (ts_code, trade_date).

    Returns DataFrame with columns:
      ts_code, trade_date, block_count, block_vol, block_amount
    which then get normalized against daily vol/amount during per-stock assembly.
    """
    dir_path = os.path.join(data_dir, 'block_trade')
    if not os.path.isdir(dir_path):
        return pd.DataFrame(columns=['ts_code', 'trade_date',
                                     'block_count', 'block_vol', 'block_amount'])

    frames = []
    use_cols = ['ts_code', 'trade_date', 'vol', 'amount']
    for fname in sorted(os.listdir(dir_path)):
        if not (fname.startswith('block_trade_') and fname.endswith('.csv')):
            continue
        path = os.path.join(dir_path, fname)
        try:
            df = pd.read_csv(path, usecols=use_cols, encoding='utf-8-sig')
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['ts_code', 'trade_date',
                                     'block_count', 'block_vol', 'block_amount'])

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw['trade_date'] = pd.to_datetime(raw['trade_date'].astype(str), errors='coerce')
    raw = raw.dropna(subset=['trade_date', 'ts_code']).reset_index(drop=True)
    raw['vol']    = pd.to_numeric(raw['vol'],    errors='coerce').astype('float32')
    raw['amount'] = pd.to_numeric(raw['amount'], errors='coerce').astype('float32')

    agg = raw.groupby(['ts_code', 'trade_date'], as_index=False).agg(
        block_count =('vol',    'size'),
        block_vol   =('vol',    'sum'),
        block_amount=('amount', 'sum'),
    )
    agg['block_count']  = agg['block_count'].astype('float32')
    agg['block_vol']    = agg['block_vol'].astype('float32')
    agg['block_amount'] = agg['block_amount'].astype('float32')
    return agg


# ─── Global indices (US / HK / Europe, lagged 1 day for causality) ──────────

def load_global_index_panel(data_dir: str) -> pd.DataFrame:
    """Load global index pct_chg and shift by one trading day.

    US/EU markets close after A-shares open, so using today's close would leak
    future information. We shift the series by one *calendar day* and then
    reindex to A-share trading days via merge_asof (backward).

    Returns a long DataFrame indexed by trade_date with one column per alias:
      <alias>_pct_chg_lag1, <alias>_vol_lag1_ratio
    """
    dir_path = os.path.join(data_dir, 'index', 'index_global')
    if not os.path.isdir(dir_path):
        return pd.DataFrame(columns=['trade_date'])

    merged_frames = []
    for alias, bare in GLOBAL_INDEX_CODES.items():
        path = os.path.join(dir_path, f"{bare}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=['trade_date', 'pct_chg', 'vol'],
                             encoding='utf-8-sig')
        except Exception:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), errors='coerce')
        df = df.dropna(subset=['trade_date']).sort_values('trade_date')
        df = df.drop_duplicates(subset=['trade_date'], keep='last').reset_index(drop=True)
        df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce').astype('float32')
        df['vol']     = pd.to_numeric(df['vol'],     errors='coerce').astype('float32')
        # Shift one step (global_pct at row i was yesterday's close overseas)
        df['pct_chg_lag1'] = df['pct_chg'].shift(1).astype('float32')
        vol_ma20 = df['vol'].rolling(20, min_periods=5).mean()
        df['vol_ratio_lag1'] = (df['vol'] / vol_ma20).shift(1).astype('float32')
        df = df[['trade_date', 'pct_chg_lag1', 'vol_ratio_lag1']]
        df.columns = ['trade_date',
                      f'{alias}_pct_chg_lag1',
                      f'{alias}_vol_ratio_lag1']
        merged_frames.append(df)

    if not merged_frames:
        return pd.DataFrame(columns=['trade_date'])

    out = merged_frames[0]
    for f in merged_frames[1:]:
        out = out.merge(f, on='trade_date', how='outer')
    return out.sort_values('trade_date').reset_index(drop=True)


# ─── Pre-computed TA factors for CSI300 (market regime) ─────────────────────

def load_idx_factor_panel(data_dir: str) -> pd.DataFrame:
    """Load a subset of TA factors for the chosen index (default CSI300).

    Returns DataFrame[trade_date, csi300_<factor>...] suitable for broadcast
    merging on trade_date.
    """
    path = os.path.join(data_dir, 'index', 'idx_factor_pro', f'{IDX_FACTOR_CODE}.csv')
    if not os.path.exists(path):
        return pd.DataFrame(columns=['trade_date'])

    try:
        head = pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns
        use  = ['trade_date'] + [c for c in IDX_FACTOR_COLUMNS if c in head]
        df   = pd.read_csv(path, usecols=use, encoding='utf-8-sig')
    except Exception:
        return pd.DataFrame(columns=['trade_date'])

    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date')
    df = df.drop_duplicates(subset=['trade_date'], keep='last').reset_index(drop=True)
    for c in df.columns:
        if c != 'trade_date':
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    rename = {c: f'idx_{c.replace("_bfq", "")}' for c in df.columns if c != 'trade_date'}
    return df.rename(columns=rename)


# ─── Fundamentals (quarterly, forward-filled) ───────────────────────────────

def _bare_code(ts_code: str) -> str:
    return ts_code.split('.')[0]


def load_fina_for_codes(data_dir: str, ts_codes: List[str]) -> Dict[str, pd.DataFrame]:
    """Load per-stock quarterly fundamentals. Returns {ts_code: df(ann_date, fina cols)}."""
    fina_dir = os.path.join(data_dir, 'fina_indicator')
    if not os.path.isdir(fina_dir):
        return {}

    out: Dict[str, pd.DataFrame] = {}
    needed = ['ts_code', 'ann_date'] + FINA_COLUMNS
    for ts_code in ts_codes:
        bare = _bare_code(ts_code)
        ex_suffix = '_SZ' if ts_code.endswith('.SZ') else '_SH'
        path = os.path.join(fina_dir, f"{bare}{ex_suffix}.csv")
        if not os.path.exists(path):
            continue
        try:
            head = pd.read_csv(path, nrows=0).columns
            use  = [c for c in needed if c in head]
            df   = pd.read_csv(path, usecols=use)
            df['ann_date'] = pd.to_datetime(df['ann_date'].astype(str),
                                             format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=['ann_date']).sort_values('ann_date')
            df = df.drop_duplicates(subset=['ann_date'], keep='last').reset_index(drop=True)
            for c in FINA_COLUMNS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
                else:
                    df[c] = np.float32(0.0)
            out[ts_code] = df[['ann_date'] + FINA_COLUMNS]
        except Exception:
            continue
    return out


def merge_fina_point_in_time(stock_df: pd.DataFrame,
                             fina_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach most recent fina row where ann_date <= trade_date (forward-fill).

    Adds a `has_fina_data` binary flag so the model can tell "no announcement
    yet" from genuine zeros.
    """
    if fina_df is None or len(fina_df) == 0:
        # Build all missing columns in one shot via concat — this avoids the
        # per-column inserts that fragment the block manager.
        idx = stock_df.index
        fill = {c: pd.Series(np.float32(0.0), index=idx) for c in FINA_COLUMNS}
        fill['has_fina_data'] = pd.Series(np.int8(0), index=idx)
        right = pd.DataFrame(fill, index=idx)
        return pd.concat([stock_df, right], axis=1, copy=False)

    merged = pd.merge_asof(
        stock_df.sort_values('trade_date'),
        fina_df.sort_values('ann_date')[['ann_date'] + FINA_COLUMNS],
        left_on='trade_date', right_on='ann_date',
        direction='backward', allow_exact_matches=True,
    )
    # Build a single replacement frame column-wise, then rebuild via pd.concat.
    # pd.concat(axis=1) joins all columns at once instead of inserting them into
    # an already-fragmented block manager — this is what the pandas warning
    # explicitly recommends.
    has_flag = (~merged['ann_date'].isna()).astype('int8').values
    merged   = merged.drop(columns=['ann_date'])
    left_cols = [c for c in merged.columns if c not in FINA_COLUMNS]
    right_pieces = {c: merged[c].fillna(0.0).astype('float32') for c in FINA_COLUMNS}
    right_pieces['has_fina_data'] = pd.Series(has_flag, index=merged.index, name='has_fina_data')
    right_df = pd.DataFrame(right_pieces, index=merged.index)
    return pd.concat([merged[left_cols], right_df], axis=1, copy=False)


# ─── Sector mapping (static categorical) ────────────────────────────────────

def load_sector_map(data_dir: str) -> Dict[str, int]:
    """Return {ts_code: sector_id} using SW L1 sector. Unknown → 0."""
    path = os.path.join(data_dir, 'stock_sectors.csv')
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, usecols=['ts_code', 'sw_l1_name'])
    except Exception:
        return {}
    df = df.dropna(subset=['ts_code']).fillna({'sw_l1_name': 'unknown'})
    cats = {name: i + 1 for i, name in enumerate(sorted(df['sw_l1_name'].unique()))}
    return {row.ts_code: cats.get(row.sw_l1_name, 0) for row in df.itertuples(index=False)}


# ─── Per-stock assembly worker ──────────────────────────────────────────────

def _assemble_one_stock(
    stock_df: pd.DataFrame,
    daily_basic_stock: Optional[pd.DataFrame],
    moneyflow_stock: Optional[pd.DataFrame],
    stk_limit_stock: Optional[pd.DataFrame],
    block_trade_stock: Optional[pd.DataFrame],
    index_panel: pd.DataFrame,
    global_panel: pd.DataFrame,
    idx_factor_panel: pd.DataFrame,
    fina_df: Optional[pd.DataFrame],
    sector_id: int,
    target_mode: str,
    forward_window: int,
    clip_target_pct: float,
) -> Optional[pd.DataFrame]:
    """Build a single-stock feature frame. Returns None if too short after NaN drop."""
    if len(stock_df) < 80:
        return None

    # 1) Technical features (price-derived)
    df = F.compute_price_features(stock_df)
    df = F.compute_calendar_features(df)

    # 2) Daily basic (valuation, mv, turnover). Drop any overlapping columns
    # from the right side so we keep the base OHLCV stream untouched.
    if daily_basic_stock is not None and not daily_basic_stock.empty:
        overlap = [c for c in daily_basic_stock.columns
                   if c in df.columns and c != 'trade_date']
        right = daily_basic_stock.drop(columns=overlap) if overlap else daily_basic_stock
        df = df.merge(right, on='trade_date', how='left')
        # Log-scale market cap (more stationary than raw)
        for c in ('total_mv', 'circ_mv'):
            if c in df.columns:
                df[f'log_{c}'] = np.log1p(df[c].clip(lower=0)).astype('float32')
        df = df.drop(columns=[c for c in ('total_mv', 'circ_mv') if c in df.columns])

    # 3) Money flow (tier-level) — normalize by daily amount
    if moneyflow_stock is not None and not moneyflow_stock.empty:
        df = df.merge(moneyflow_stock, on='trade_date', how='left')
        for tier in MONEYFLOW_TIERS:
            col = f'net_{tier}_amount'
            if col in df.columns:
                df[f'{col}_ratio'] = (df[col] / df['amount'].replace(0, np.nan)).astype('float32')
        if 'net_mf_amount' in df.columns:
            df['net_mf_amount_ratio'] = (df['net_mf_amount']
                                         / df['amount'].replace(0, np.nan)).astype('float32')
        # Drop the raw absolute-amount columns to avoid scale-leak
        drop = ['net_mf_amount'] + [f'net_{t}_amount' for t in MONEYFLOW_TIERS]
        df = df.drop(columns=[c for c in drop if c in df.columns])

    # 4) stk_limit → continuous limit ratios
    if stk_limit_stock is not None and not stk_limit_stock.empty:
        df = df.merge(stk_limit_stock, on='trade_date', how='left')
        if 'up_limit' in df.columns:
            df['up_limit_ratio'] = ((df['up_limit'] / df['close']) - 1).astype('float32')
        if 'down_limit' in df.columns:
            df['down_limit_ratio'] = (1 - (df['down_limit'] / df['close'])).astype('float32')
        df = df.drop(columns=[c for c in ('up_limit', 'down_limit') if c in df.columns])

    # 5) Block trades → ratios vs daily volume & amount
    if block_trade_stock is not None and not block_trade_stock.empty:
        df = df.merge(block_trade_stock, on='trade_date', how='left')
        df['block_count']       = df['block_count'].fillna(0.0).astype('float32')
        df['block_vol_ratio']   = (df['block_vol']    / df['vol']   .replace(0, np.nan)) \
                                    .fillna(0.0).astype('float32')
        df['block_amount_ratio'] = (df['block_amount'] / df['amount'].replace(0, np.nan)) \
                                    .fillna(0.0).astype('float32')
        df = df.drop(columns=[c for c in ('block_vol', 'block_amount') if c in df.columns])

    # 6) Macro indices (broadcast market-wide features to every row by trade_date)
    if not index_panel.empty:
        df = df.merge(index_panel, on='trade_date', how='left')

    # 7) Global indices (lagged 1 day). merge_asof with backward direction so
    # A-share dates with no overseas match pick up the most recent one.
    if not global_panel.empty:
        df = pd.merge_asof(
            df.sort_values('trade_date'),
            global_panel.sort_values('trade_date'),
            on='trade_date', direction='backward', allow_exact_matches=True,
        )

    # 8) Pre-computed TA factors on CSI300 (market regime)
    if not idx_factor_panel.empty:
        df = df.merge(idx_factor_panel, on='trade_date', how='left')

    # 9) Quarterly fundamentals (point-in-time forward-fill). Defragment after
    # all macro merges since the frame now has ~150 columns.
    df = merge_fina_point_in_time(df, fina_df).copy()

    # 7) Target: next-day pct_chg (or excess vs csi300)
    if target_mode == 'excess' and 'csi300_pct_chg' in df.columns:
        tgt = df['pct_chg'] - df['csi300_pct_chg']
    else:
        # Raw pct_chg (or excess fallback when csi300 missing)
        tgt = df['pct_chg']
    target = tgt.shift(-forward_window).astype('float32')
    if clip_target_pct and clip_target_pct > 0:
        target = target.clip(-clip_target_pct, clip_target_pct).astype('float32')

    # Add sector_id + target in one assign so we don't re-fragment after the defrag above.
    df = df.assign(sector_id=np.int16(sector_id), target=target)

    # 8) Drop rows with no target (last `forward_window` rows)
    df = df.dropna(subset=['target'])

    # 9) Drop the warm-up rows whose critical long-window TA features are NaN
    # (rather than using a fixed offset — lets the loader recover if windows change).
    core_features = ['close_ma_60_ratio', 'dist_from_high_60', 'atr_14_pct']
    df = df.dropna(subset=[c for c in core_features if c in df.columns])

    if df.empty:
        return None

    # Final NaN→0 fill for any remaining feature cells (e.g. missing daily_basic)
    feat_cols = [c for c in df.columns if c not in ('ts_code', 'trade_date', 'target')]
    for c in feat_cols:
        if df[c].dtype.kind == 'f':
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype('float32')
    # Defragment: many per-column assignments above fragment the block manager;
    # one copy consolidates the internal blocks and silences PerformanceWarning.
    return df.reset_index(drop=True).copy()


# ─── Public entry point ─────────────────────────────────────────────────────

def build_panel(cfg: dict) -> pd.DataFrame:
    """Build the full (stock × date) feature + target panel.

    Parameters
    ----------
    cfg : dict
        See xgbmodel.config.get_config(); keys used: data_dir, min_rows_per_stock,
        max_stocks, target_mode, forward_window, clip_target_pct.

    Returns
    -------
    pd.DataFrame with columns [ts_code, trade_date, <features...>, target]
    """
    data_dir         = cfg['data_dir']
    min_rows         = cfg['min_rows_per_stock']
    max_stocks       = cfg.get('max_stocks', 0)
    target_mode      = cfg.get('target_mode', 'raw')
    forward_window   = cfg.get('forward_window', 1)
    clip_target_pct  = cfg.get('clip_target_pct', 11.0)

    print(f"[xgbmodel] Building panel from {data_dir}")
    print(f"  target_mode={target_mode}  forward_window={forward_window}  max_stocks={max_stocks or 'all'}")

    # Listing + filter
    files = _list_stock_files(data_dir, max_stocks=max_stocks)
    print(f"  {len(files)} candidate stock files")

    # Cross-sectional daily files — load once, then groupby ts_code
    print("  loading daily_basic/ ...")
    db_all = load_daily_basic(data_dir)
    print(f"    {len(db_all):,} rows across {db_all['ts_code'].nunique() if not db_all.empty else 0} codes")

    print("  loading moneyflow/ ...")
    mf_all = load_moneyflow(data_dir)
    print(f"    {len(mf_all):,} rows")

    print("  loading stk_limit/ ...")
    lim_all = load_stk_limit(data_dir)
    print(f"    {len(lim_all):,} rows")

    print("  loading block_trade/ ...")
    bt_all = load_block_trade_agg(data_dir)
    print(f"    {len(bt_all):,} aggregated block-trade rows")

    print("  loading index/index_dailybasic/ ...")
    idx_panel = load_index_panel(data_dir)
    print(f"    {len(idx_panel):,} macro rows with {len(idx_panel.columns) - 1} cols")

    print("  loading index/index_global/ ...")
    global_panel = load_global_index_panel(data_dir)
    print(f"    {len(global_panel):,} rows with {len(global_panel.columns) - 1} lagged cols")

    print("  loading index/idx_factor_pro/ ...")
    idx_factor_panel = load_idx_factor_panel(data_dir)
    print(f"    {len(idx_factor_panel):,} rows with {len(idx_factor_panel.columns) - 1} TA cols")

    # Pre-group cross-sectional data by ts_code for O(1) lookup per stock.
    # Drop ts_code and dedupe on trade_date so the per-stock merges don't
    # carry string cols or duplicate rows into the panel.
    def _split_groups(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if frame.empty:
            return {}
        frame = (frame.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
                      .sort_values(['ts_code', 'trade_date']))
        out = {}
        for code, grp in frame.groupby('ts_code'):
            out[code] = grp.drop(columns=['ts_code']).reset_index(drop=True)
        return out

    db_groups  = _split_groups(db_all)
    mf_groups  = _split_groups(mf_all)
    lim_groups = _split_groups(lim_all)
    bt_groups  = _split_groups(bt_all)

    # Fundamentals + sectors
    ts_codes = [tc for tc, _ in files]
    print(f"  loading fina_indicator/ for {len(ts_codes)} stocks ...")
    fina_map = load_fina_for_codes(data_dir, ts_codes)
    print(f"    fina available for {len(fina_map)} stocks")
    sector_map = load_sector_map(data_dir)

    # Free the big cross-sectional frames early — keep only per-code groups
    del db_all, mf_all, lim_all, bt_all
    gc.collect()

    assembled: List[pd.DataFrame] = []
    skipped = 0
    for i, (ts_code, path) in enumerate(files):
        if i % 200 == 0 and i > 0:
            print(f"    assembled {i}/{len(files)} ...  kept={len(assembled)}  skipped={skipped}")
        raw = _load_stock_csv(path, min_rows=min_rows)
        if raw is None:
            skipped += 1
            continue
        raw['ts_code'] = ts_code
        out = _assemble_one_stock(
            stock_df          = raw,
            daily_basic_stock = db_groups.get(ts_code),
            moneyflow_stock   = mf_groups.get(ts_code),
            stk_limit_stock   = lim_groups.get(ts_code),
            block_trade_stock = bt_groups.get(ts_code),
            index_panel       = idx_panel,
            global_panel      = global_panel,
            idx_factor_panel  = idx_factor_panel,
            fina_df           = fina_map.get(ts_code),
            sector_id         = sector_map.get(ts_code, 0),
            target_mode       = target_mode,
            forward_window    = forward_window,
            clip_target_pct   = clip_target_pct,
        )
        if out is None or out.empty:
            skipped += 1
            continue
        assembled.append(out)

    print(f"  assembled {len(assembled)} stocks  skipped={skipped}")
    if not assembled:
        raise RuntimeError("No stocks assembled — check data_dir / min_rows_per_stock")

    panel = pd.concat(assembled, ignore_index=True, sort=False)

    # Compute cross-sectional daily rank / de-meaned features on the merged
    # panel — must happen after assembly so ranks cover the full cross-section.
    print("  computing cross-sectional rank + de-mean features ...")
    panel = add_cross_section_features(panel)

    # Cross-sectional target demean (or z-score).
    # Rationale: without this, per-day-constant features (macro indices, global
    # lags, calendar) dominate total_gain because they correctly predict the
    # per-day MEAN of the target across the universe — but a per-day-constant
    # prediction contributes zero cross-sectional rank IC, which is the metric
    # we actually optimize. Demeaning the target per trade_date removes this
    # degree of freedom from the objective, forcing splits on stock-discriminating
    # features. `target_mode='excess'` only partially fixes this because CSI300
    # is 300 of ~5100 stocks and the excess-return mean over the full universe
    # is still nonzero.
    cs_norm = cfg.get('cs_target_norm', 'demean')
    if cs_norm in ('demean', 'zscore'):
        before_std = float(panel['target'].std())
        mean_by_date = panel.groupby('trade_date')['target'].transform('mean')
        demeaned = panel['target'] - mean_by_date
        if cs_norm == 'zscore':
            std_by_date = panel.groupby('trade_date')['target'].transform('std')
            demeaned = demeaned / std_by_date.replace(0, np.nan)
            demeaned = demeaned.fillna(0.0)
        panel['target'] = demeaned.astype('float32')
        after_std = float(panel['target'].std())
        print(f"  applied cs_target_norm={cs_norm}  "
              f"target std {before_std:.4f} → {after_std:.4f}")

    # Canonical column order: ts_code, trade_date, ..., target
    cols = ['ts_code', 'trade_date'] + \
           [c for c in panel.columns if c not in ('ts_code', 'trade_date', 'target')] + \
           ['target']
    panel = panel[cols]

    print(f"  panel shape: {panel.shape}  "
          f"{panel['ts_code'].nunique()} stocks, "
          f"{panel['trade_date'].min().date()} → {panel['trade_date'].max().date()}")
    return panel


def list_feature_columns(panel: pd.DataFrame) -> List[str]:
    """Feature columns = numeric/bool cols except identifiers + target.

    Drops any accidental object-dtype columns so we never hand xgboost a
    string column.
    """
    drop = {'ts_code', 'trade_date', 'target'}
    out = []
    for c in panel.columns:
        if c in drop:
            continue
        if panel[c].dtype.kind in ('f', 'i', 'u', 'b'):
            out.append(c)
    return out
