"""
Feature engineering for the deeptime regression pipeline.

Extends dl/data_processing.py with:
  - Quarterly fundamentals from fina_indicator/ (forward-filled from ann_date)
  - Block trade aggregates from block_trade/
  - Extended money flow (SM/MD tiers)
  - Continuous price limit ratios (up_limit_ratio, down_limit_ratio)
  - Regression targets: excess return pct_chg - csi300_pct_chg for days 1..5
"""

import gc
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dl.data_processing import (
    load_sector_data,
    load_stock_data,
    load_daily_basic_data,
    load_market_context_data,
    load_index_membership_data,
    load_stk_limit_data,
    load_moneyflow_data,
    normalize_data,
    calculate_technical_features,
    calculate_date_features,
    apply_cs_normalization,
    compute_daily_cs_stats,
    compute_cross_section_tech_stats,
    merge_daily_basic,
    merge_market_context,
    merge_index_membership,
    merge_stk_limit,
    merge_moneyflow,
)

from .config import (
    DT_FEATURE_COLUMNS, DT_KNOWN_FUTURE_COLUMNS, DT_OBSERVED_PAST_COLUMNS,
    _DT_FUTURE_FEAT_IDX, _DT_OBS_PAST_FEAT_IDX,
    FINA_INDICATOR_COLUMNS, BLOCK_TRADE_COLUMNS, EXTRA_MONEYFLOW_COLUMNS,
    PRICE_LIMIT_RATIO_COLUMNS,
    FORWARD_WINDOWS, NUM_HORIZONS, SEQUENCE_LENGTH, MAX_FORWARD_WINDOW,
    NUM_DT_FEATURES, NUM_DT_KNOWN_FUTURE, NUM_DT_OBSERVED_PAST,
    DT_CS_NORMALIZE_TECH_FEATURES,
    INTERLEAVED_TEST_START,
    ROLLING_TRAIN_MONTHS, ROLLING_VAL_MONTHS, ROLLING_TEST_MONTHS,
    ROLLING_STEP_MONTHS, PURGE_GAP_DAYS,
)


# ─── Fina indicator loading ────────────────────────────────────────────────────

def load_fina_indicator_data(data_dir: str, ts_codes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load quarterly financial indicators for each stock.

    Returns:
        {bare_code: DataFrame sorted by ann_date ascending}
        Columns: ann_date, roe, roa, grossprofit_margin, netprofit_margin,
                 current_ratio, quick_ratio, debt_to_assets,
                 assets_yoy, equity_yoy, op_yoy, ebt_yoy, eps
    """
    fina_dir = os.path.join(data_dir, 'fina_indicator')
    if not os.path.isdir(fina_dir):
        print(f"  Warning: fina_indicator/ not found at {fina_dir}")
        return {}

    result: Dict[str, pd.DataFrame] = {}
    use_cols = ['ts_code', 'ann_date', 'end_date'] + FINA_INDICATOR_COLUMNS

    for ts_code in ts_codes:
        bare = str(ts_code).split('.')[0]
        # Files named as {bare}_{EXCHANGE}.csv  e.g. 000001_SZ.csv, 600000_SH.csv
        found = None
        for suffix in ['_SZ', '_SH', '']:
            fname = f"{bare}{suffix}.csv"
            path  = os.path.join(fina_dir, fname)
            if os.path.exists(path):
                found = path
                break
        if found is None:
            # Fallback: try ts_code with dot-exchange e.g. 000001.SZ
            for dot_sfx in ['.SZ', '.SH']:
                fname = f"{bare}{dot_sfx}.csv"
                path  = os.path.join(fina_dir, fname)
                if os.path.exists(path):
                    found = path
                    break
        if found is None:
            continue
        try:
            df = pd.read_csv(found, usecols=[c for c in use_cols if c in
                             pd.read_csv(found, nrows=0).columns])
            df['ann_date'] = pd.to_datetime(df['ann_date'].astype(str), errors='coerce')
            df = df.dropna(subset=['ann_date'])
            # Deduplicate: same ann_date may appear twice in raw data
            df = df.sort_values('ann_date').drop_duplicates(subset=['ann_date'], keep='last')
            df = df.reset_index(drop=True)
            result[bare] = df
        except Exception:
            continue

    print(f"  Loaded fina_indicator for {len(result)}/{len(ts_codes)} stocks")
    return result


def forward_fill_fundamentals(
    stock_df: pd.DataFrame,
    fina_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Point-in-time join: for each trading day, use the most recent quarterly
    report whose ann_date <= trade_date. Forward-fill until next announcement.
    """
    # Ensure all fina columns exist (filled with NaN initially)
    for col in FINA_INDICATOR_COLUMNS:
        if col not in stock_df.columns:
            stock_df[col] = np.nan

    if fina_df is None or len(fina_df) == 0:
        return stock_df

    if not pd.api.types.is_datetime64_any_dtype(stock_df['trade_date']):
        stock_df = stock_df.copy()
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'].astype(str))

    # Merge: for each trading date, find the last ann_date <= trade_date
    fina_sorted = fina_df.sort_values('ann_date').reset_index(drop=True)
    ann_dates   = fina_sorted['ann_date'].values
    trade_dates = stock_df['trade_date'].values

    # Use searchsorted to find the last applicable announcement
    idx = np.searchsorted(ann_dates, trade_dates, side='right') - 1   # -1 if before first ann

    for col in FINA_INDICATOR_COLUMNS:
        if col not in fina_sorted.columns:
            continue
        vals = fina_sorted[col].values
        col_values = np.where(idx >= 0, vals[np.clip(idx, 0, len(vals) - 1)], np.nan)
        stock_df[col] = col_values.astype('float32')

    return stock_df


# ─── Block trade loading and feature engineering ───────────────────────────────

def load_block_trade_data(
    data_dir: str,
    date_range: Optional[Tuple[int, int]] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Load block trade CSVs and aggregate per stock per day.

    Returns:
        {date_int: DataFrame with columns [ts_code, block_vol, block_amt,
                                           block_count, block_net_amt]}
    """
    block_dir = os.path.join(data_dir, 'block_trade')
    if not os.path.isdir(block_dir):
        print(f"  Warning: block_trade/ not found at {block_dir}")
        return {}

    result: Dict[int, pd.DataFrame] = {}
    files = sorted([f for f in os.listdir(block_dir) if f.endswith('.csv')])

    for fname in files:
        try:
            date_str = fname.replace('block_trade_', '').replace('.csv', '')
            date_int = int(date_str)
        except ValueError:
            continue

        if date_range is not None:
            if date_int < date_range[0] or date_int > date_range[1]:
                continue

        try:
            df = pd.read_csv(os.path.join(block_dir, fname),
                             usecols=['ts_code', 'price', 'vol', 'amount', 'buyer', 'seller'])
            if len(df) == 0:
                continue
            # Institutional sell = seller is '机构专用'
            df['is_inst_sell'] = df['seller'].str.contains('机构', na=False).astype(float)
            df['is_inst_buy']  = df['buyer'].str.contains('机构', na=False).astype(float)

            agg = df.groupby('ts_code').agg(
                block_vol   = ('vol',    'sum'),
                block_amt   = ('amount', 'sum'),
                block_count = ('amount', 'count'),
                inst_sell_amt = ('amount', lambda x: (x * df.loc[x.index, 'is_inst_sell']).sum()),
                inst_buy_amt  = ('amount', lambda x: (x * df.loc[x.index, 'is_inst_buy']).sum()),
            ).reset_index()
            agg['block_net_amt'] = agg['inst_buy_amt'] - agg['inst_sell_amt']
            result[date_int] = agg[['ts_code', 'block_vol', 'block_amt', 'block_count', 'block_net_amt']]
        except Exception:
            continue

    print(f"  Loaded block_trade for {len(result)} trading days")
    return result


def _pregroup_block_trade(block_trade_daily: Dict[int, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Reorganise block trade data by stock code for O(1) per-stock lookup."""
    if not block_trade_daily:
        return {}

    rows = []
    for date_int, df in block_trade_daily.items():
        df2 = df.copy()
        df2['trade_date'] = date_int
        rows.append(df2)

    combined = pd.concat(rows, ignore_index=True)
    by_stock: Dict[str, pd.DataFrame] = {}
    for key, grp in combined.groupby('ts_code'):
        bare = str(key).split('.')[0]
        by_stock[bare] = grp.drop(columns=['ts_code']).reset_index(drop=True)
    return by_stock


def merge_block_trade_features(
    stock_df: pd.DataFrame,
    block_trade_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute block_vol_ratio, block_amt_ratio, block_count_log, block_buy_sell_ratio
    and merge into stock_df.
    """
    for col in BLOCK_TRADE_COLUMNS:
        stock_df[col] = 0.0

    if block_trade_df is None or len(block_trade_df) == 0:
        return stock_df

    if not pd.api.types.is_datetime64_any_dtype(stock_df['trade_date']):
        stock_df = stock_df.copy()
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'].astype(str))

    # Compute 20-day average daily volume for normalization
    avg_vol = stock_df['vol'].rolling(20, min_periods=1).mean().replace(0, np.nan)
    avg_amt = stock_df['amount'].rolling(20, min_periods=1).mean().replace(0, np.nan)

    # Map block trade date_int → trade_date int
    trade_date_ints = (
        stock_df['trade_date'].dt.year  * 10000 +
        stock_df['trade_date'].dt.month * 100   +
        stock_df['trade_date'].dt.day
    ).values

    # Create a lookup from date_int → row in block_trade_df
    bt = block_trade_df.copy()
    bt_date_col = 'trade_date' if 'trade_date' in bt.columns else bt.columns[0]
    bt = bt.set_index(bt_date_col)

    vol_ratio   = np.zeros(len(stock_df), dtype='float32')
    amt_ratio   = np.zeros(len(stock_df), dtype='float32')
    count_log   = np.zeros(len(stock_df), dtype='float32')
    buy_sell    = np.zeros(len(stock_df), dtype='float32')

    for i, d in enumerate(trade_date_ints):
        if d not in bt.index:
            continue
        row = bt.loc[d]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        bvol  = float(row.get('block_vol',     0) or 0)
        bamt  = float(row.get('block_amt',     0) or 0)
        bcnt  = float(row.get('block_count',   0) or 0)
        bnet  = float(row.get('block_net_amt', 0) or 0)
        avol  = float(avg_vol.iloc[i]) if not np.isnan(avg_vol.iloc[i]) else 1.0
        aamt  = float(avg_amt.iloc[i]) if not np.isnan(avg_amt.iloc[i]) else 1.0

        vol_ratio[i] = bvol / (avol + 1e-8)
        amt_ratio[i] = bamt / (aamt + 1e-8)
        count_log[i] = np.log1p(bcnt)
        buy_sell[i]  = np.clip(bnet / (bamt + 1e-8), -1.0, 1.0) if bamt > 0 else 0.0

    stock_df['block_vol_ratio']      = vol_ratio
    stock_df['block_amt_ratio']      = amt_ratio
    stock_df['block_count_log']      = count_log
    stock_df['block_buy_sell_ratio'] = buy_sell
    return stock_df


# ─── Extended money flow ───────────────────────────────────────────────────────

def compute_extended_moneyflow(
    stock_df: pd.DataFrame,
    moneyflow_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Add net_sm_flow_ratio and net_md_flow_ratio to stock_df."""
    for col in EXTRA_MONEYFLOW_COLUMNS:
        stock_df[col] = 0.0

    if moneyflow_df is None or len(moneyflow_df) == 0:
        return stock_df

    if not pd.api.types.is_datetime64_any_dtype(stock_df['trade_date']):
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'].astype(str))

    mf_cols = ['trade_date', 'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount']
    available = [c for c in mf_cols if c in moneyflow_df.columns]
    if len(available) < 3:
        return stock_df

    mf = moneyflow_df[available].copy()
    if not pd.api.types.is_datetime64_any_dtype(mf['trade_date']):
        mf['trade_date'] = pd.to_datetime(mf['trade_date'].astype(str))

    merged = stock_df[['trade_date', 'amount']].merge(mf, on='trade_date', how='left')
    total_amt = merged['amount'].values

    if 'buy_sm_amount' in mf.columns and 'sell_sm_amount' in mf.columns:
        net_sm = (merged['buy_sm_amount'].values - merged['sell_sm_amount'].values)
        stock_df['net_sm_flow_ratio'] = np.where(
            total_amt > 0, net_sm / (total_amt + 1e-8), 0.0
        ).astype('float32')

    if 'buy_md_amount' in mf.columns and 'sell_md_amount' in mf.columns:
        net_md = (merged['buy_md_amount'].values - merged['sell_md_amount'].values)
        stock_df['net_md_flow_ratio'] = np.where(
            total_amt > 0, net_md / (total_amt + 1e-8), 0.0
        ).astype('float32')

    return stock_df


# ─── Price limit ratio features ───────────────────────────────────────────────

def compute_price_limit_ratios(
    stock_df: pd.DataFrame,
    stk_limit_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Add up_limit_ratio and down_limit_ratio to stock_df.
    Default: ±10% for normal stocks.
    """
    stock_df['up_limit_ratio']   = 0.10
    stock_df['down_limit_ratio'] = -0.10

    if stk_limit_df is None or len(stk_limit_df) == 0:
        return stock_df

    if not pd.api.types.is_datetime64_any_dtype(stock_df['trade_date']):
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'].astype(str))

    lim = stk_limit_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(lim['trade_date']):
        lim['trade_date'] = pd.to_datetime(lim['trade_date'].astype(str))

    if 'up_limit' not in lim.columns or 'down_limit' not in lim.columns:
        return stock_df

    merged = stock_df[['trade_date', 'pre_close']].merge(
        lim[['trade_date', 'up_limit', 'down_limit']], on='trade_date', how='left'
    )
    pre_close = merged['pre_close'].values
    up_lim    = merged['up_limit'].values
    dn_lim    = merged['down_limit'].values

    valid = (pre_close > 0) & np.isfinite(pre_close)
    up_ratio = np.where(valid & np.isfinite(up_lim),
                        up_lim / (pre_close + 1e-8) - 1.0, 0.10)
    dn_ratio = np.where(valid & np.isfinite(dn_lim),
                        dn_lim / (pre_close + 1e-8) - 1.0, -0.10)

    # Clip to reasonable bounds (STAR market = ±20%, ST = ±5%)
    stock_df['up_limit_ratio']   = np.clip(up_ratio, 0.04, 0.22).astype('float32')
    stock_df['down_limit_ratio'] = np.clip(dn_ratio, -0.22, -0.04).astype('float32')
    return stock_df


# ─── CSI300 forward return lookup ─────────────────────────────────────────────

def build_csi300_forward_returns_regression(data_dir: str) -> Dict[int, List[float]]:
    """
    Build {date_int → [ret_1, ret_2, ret_3, ret_4, ret_5]} from CSI300 close prices.
    Used to compute excess returns for regression targets.
    """
    path = os.path.join(data_dir, 'index', 'idx_factor_pro', '000300_SH.csv')
    if not os.path.exists(path):
        # Fallback: try index_dailybasic
        path = os.path.join(data_dir, 'index', 'index_dailybasic', '000300_SH.csv')
    if not os.path.exists(path):
        print("  Warning: CSI300 data not found for target computation")
        return {}

    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df = df[['trade_date', 'close']].dropna().sort_values('trade_date').reset_index(drop=True)

    closes = df['close'].values
    date_ints = (
        df['trade_date'].dt.year  * 10000 +
        df['trade_date'].dt.month * 100   +
        df['trade_date'].dt.day
    ).values.astype(np.int32)

    lookup: Dict[int, List[float]] = {}
    n = len(closes)
    max_fw = max(FORWARD_WINDOWS)
    for i in range(n - max_fw):
        d = int(date_ints[i])
        rets = []
        for fw in FORWARD_WINDOWS:
            if closes[i] > 0:
                ret = 100.0 * (closes[i + fw] - closes[i]) / closes[i]
            else:
                ret = 0.0
            rets.append(ret)
        lookup[d] = rets

    print(f"  CSI300 forward returns: {len(lookup)} dates")
    return lookup


# ─── Regression target computation ────────────────────────────────────────────

def compute_regression_targets(
    closes: np.ndarray,
    valid_indices: List[int],
    dates_arr: np.ndarray,
    csi300_fw_rets: Dict[int, List[float]],
    target_mode: str = 'excess',
) -> np.ndarray:
    """
    Compute regression targets for each sequence anchor.

    Args:
        closes:         close price array (N stocks × T days) — 1D array for one stock
        valid_indices:  sequence anchor indices (the 'current' day index)
        dates_arr:      YYYYMMDD int for each row
        csi300_fw_rets: {date_int → [ret_1..ret_5]}
        target_mode:    'excess' or 'raw'

    Returns:
        targets (len(valid_indices), NUM_HORIZONS) float32
    """
    targets = np.zeros((len(valid_indices), NUM_HORIZONS), dtype='float32')
    _neutral = [0.0] * NUM_HORIZONS

    for si, i in enumerate(valid_indices):
        date_int = int(dates_arr[i - 1])
        csi_rets = csi300_fw_rets.get(date_int, _neutral)
        for h, fw in enumerate(FORWARD_WINDOWS):
            if i + fw - 1 >= len(closes) or closes[i - 1] <= 0:
                targets[si, h] = 0.0
                continue
            raw_ret = 100.0 * (closes[i + fw - 1] - closes[i - 1]) / closes[i - 1]
            if target_mode == 'excess':
                targets[si, h] = raw_ret - (csi_rets[h] if h < len(csi_rets) else 0.0)
            else:
                targets[si, h] = raw_ret

    return targets


# ─── Size decile computation ───────────────────────────────────────────────────

def compute_size_decile(
    ts_code: str,
    daily_basic_dict: Dict[str, pd.DataFrame],
) -> int:
    """Return market-cap decile (0-9) for this stock, or 10 (unknown)."""
    bare = str(ts_code).split('.')[0]
    df   = daily_basic_dict.get(bare)
    if df is None or 'circ_mv' not in df.columns:
        return 10
    median_mv = df['circ_mv'].median()
    if np.isnan(median_mv) or median_mv <= 0:
        return 10
    # Will be replaced by cross-sectional decile in the caller; return raw value
    return median_mv


# ─── Main dataset builder ─────────────────────────────────────────────────────

def prepare_dataset_regression(
    stock_files:          List[Tuple[str, str]],
    sector_data:          pd.DataFrame,
    daily_basic:          pd.DataFrame,
    output_dir:           str,
    data_dir:             str,
    config:               dict,
    fina_data:            Optional[Dict[str, pd.DataFrame]] = None,
    block_trade_by_stock: Optional[Dict[str, pd.DataFrame]] = None,
    market_context:       Optional[pd.DataFrame] = None,
    index_membership:     Optional[Dict] = None,
    stk_limit:            Optional[pd.DataFrame] = None,
    moneyflow:            Optional[pd.DataFrame] = None,
    cs_tech_stats:        Optional[Dict] = None,
    split_mode:           str = 'rolling_window',
) -> dict:
    """
    Process stocks streaming one-at-a-time and write to memmap cache.
    Returns metadata dict.
    """
    from .memmap_dataset import RegressionDataWriter

    seq_len       = config.get('sequence_length', SEQUENCE_LENGTH)
    max_fw        = MAX_FORWARD_WINDOW
    target_mode   = config.get('target_mode', 'excess')
    max_seqs      = config.get('max_sequences_per_stock', None)

    from datetime import datetime as _dt

    def _build_enum(series, col_name):
        """Build {value: int_id} mapping; 'Unknown' always last."""
        vals = [v for v in series.dropna().unique() if str(v) != 'Unknown']
        m = {str(v): i for i, v in enumerate(sorted(vals))}
        m['Unknown'] = len(m)
        return m

    # ── Sector / industry encodings — use SW L1/L2 from enriched sector_data ──
    # Prefer sw_l1_name (SW classification) over legacy 'sector' (9-category)
    sw_l1_col = 'sw_l1_name' if 'sw_l1_name' in sector_data.columns else 'sector'
    sw_l2_col = 'sw_l2_name' if 'sw_l2_name' in sector_data.columns else 'industry'

    sector_to_id   = _build_enum(sector_data[sw_l1_col], sw_l1_col) if len(sector_data) else {'Unknown': 0}
    industry_to_id = _build_enum(sector_data[sw_l2_col], sw_l2_col) if len(sector_data) else {'Unknown': 0}
    sub_ind_to_id  = {'Unknown': 0}   # SW L3 not downloaded; placeholder

    # New static dimensions
    area_to_id     = _build_enum(sector_data['area'],   'area')   if 'area'   in sector_data.columns else {'Unknown': 0}
    board_to_id    = _build_enum(sector_data['market'], 'market') if 'market' in sector_data.columns else {'Unknown': 0}
    # IPO age buckets: 0=<1yr, 1=1-2yr, 2=2-3yr, 3=3-5yr, 4=5-10yr, 5=10yr+, 6=unknown
    def _ipo_age_bucket(list_date_str) -> int:
        try:
            ld = pd.to_datetime(str(list_date_str), errors='coerce')
            if pd.isna(ld): return 6
            years = (pd.Timestamp.today() - ld).days / 365.25
            if years <  1: return 0
            if years <  2: return 1
            if years <  3: return 2
            if years <  5: return 3
            if years < 10: return 4
            return 5
        except Exception:
            return 6

    # Build lookup dicts (O(1) per stock)
    sector_dict   = {}
    industry_dict = {}
    sub_ind_dict  = {}
    area_dict     = {}
    board_dict    = {}
    ipo_age_dict  = {}
    if len(sector_data):
        for _, row in sector_data.iterrows():
            code = str(row['ts_code'])
            bare = code.split('.')[0]
            for d in [code, bare]:
                sector_dict[d]   = str(row.get(sw_l1_col, 'Unknown'))
                industry_dict[d] = str(row.get(sw_l2_col, 'Unknown'))
                sub_ind_dict[d]  = 'Unknown'
                area_dict[d]     = str(row.get('area',   'Unknown'))
                board_dict[d]    = str(row.get('market', 'Unknown'))
                ipo_age_dict[d]  = _ipo_age_bucket(row.get('list_date', None))

    # Pre-group daily_basic
    daily_basic_dict: Dict[str, pd.DataFrame] = {}
    if daily_basic is not None and len(daily_basic) > 0:
        print("Pre-grouping daily_basic...")
        for key, grp in daily_basic.groupby('ts_code'):
            bare = str(key).split('.')[0]
            daily_basic_dict[bare] = grp.drop(columns=['ts_code'], errors='ignore').reset_index(drop=True)
        del daily_basic
        gc.collect()
        print(f"  Pre-grouped {len(daily_basic_dict)} stocks")

    # Cross-section valuation stats
    daily_cs_stats: Dict = {}
    if daily_basic_dict:
        print("Computing cross-section valuation stats...")
        daily_cs_stats = compute_daily_cs_stats(daily_basic_dict)

    # Pre-group stk_limit, moneyflow — only if already filtered/small
    # If None is passed, per-stock loading is skipped (features default to 0)
    stk_limit_dict: Dict[str, pd.DataFrame] = {}
    if stk_limit is not None and len(stk_limit) > 0:
        for key, grp in stk_limit.groupby('ts_code'):
            stk_limit_dict[str(key).split('.')[0]] = grp.reset_index(drop=True)
        del stk_limit; gc.collect()
        print(f"  stk_limit grouped: {len(stk_limit_dict)} stocks")

    moneyflow_dict: Dict[str, pd.DataFrame] = {}
    if moneyflow is not None and len(moneyflow) > 0:
        for key, grp in moneyflow.groupby('ts_code'):
            moneyflow_dict[str(key).split('.')[0]] = grp.reset_index(drop=True)
        del moneyflow; gc.collect()
        print(f"  moneyflow grouped: {len(moneyflow_dict)} stocks")

    # CSI300 forward returns for target computation
    csi300_fw_rets = build_csi300_forward_returns_regression(data_dir)

    # Compute median circ_mv per stock for size decile assignment
    median_mvs: Dict[str, float] = {}
    for bare, df in daily_basic_dict.items():
        if 'circ_mv' in df.columns:
            mv = df['circ_mv'].median()
            if np.isfinite(mv) and mv > 0:
                median_mvs[bare] = float(mv)

    # Assign size decile (0-9 by percentile of median circ_mv)
    size_decile_map: Dict[str, int] = {}
    if median_mvs:
        mvs    = np.array(list(median_mvs.values()))
        codes  = list(median_mvs.keys())
        decile = pd.qcut(mvs, 10, labels=False, duplicates='drop')
        for code, d in zip(codes, decile):
            size_decile_map[code] = int(d)

    # Initialize writer
    os.makedirs(output_dir, exist_ok=True)
    writer = RegressionDataWriter(
        output_dir    = output_dir,
        seq_length    = seq_len,
        n_features    = NUM_DT_FEATURES,
        n_past        = NUM_DT_OBSERVED_PAST,
        n_future      = NUM_DT_KNOWN_FUTURE,
        num_horizons  = NUM_HORIZONS,
        max_fw        = max_fw,
    )

    # ── PASS 1: Lightweight date scan — O(1 stock RAM) ────────────────────────
    # Read only trade_date column from each stock to collect all anchor dates
    # and count sequences per split WITHOUT loading features into memory.
    print(f"\nPass 1: scanning {len(stock_files)} stocks for anchor dates...")
    min_rows = config.get('min_data_points', 100)
    all_anchor_dates_list: List[int] = []
    viable_stocks: List[Tuple[str, str]] = []

    for ts_code, filepath in stock_files:
        try:
            dates_df = pd.read_csv(filepath, usecols=['trade_date'])
            if len(dates_df) < min_rows:
                continue
            dates_df['trade_date'] = pd.to_datetime(dates_df['trade_date'].astype(str))
            dates_df = dates_df.sort_values('trade_date').reset_index(drop=True)
            n = len(dates_df)
            if n < seq_len + max_fw:
                continue
            # Anchor date = last date of the seq_len window
            date_ints = (
                dates_df['trade_date'].dt.year  * 10000 +
                dates_df['trade_date'].dt.month * 100   +
                dates_df['trade_date'].dt.day
            ).values.astype(np.int32)
            valid_indices = range(seq_len, n - max_fw + 1)
            if max_seqs:
                valid_indices = list(valid_indices)[:max_seqs]
            for i in valid_indices:
                all_anchor_dates_list.append(int(date_ints[i - 1]))
            viable_stocks.append((ts_code, filepath))
        except Exception:
            continue

    all_anchor_dates = np.array(all_anchor_dates_list, dtype=np.int32)
    del all_anchor_dates_list
    print(f"  {len(viable_stocks)} viable stocks, {len(all_anchor_dates):,} potential sequences")

    # Build split map from the complete set of anchor dates
    writer.build_split_map(all_anchor_dates, split_mode, config)
    dsm = writer.date_split_map

    # Count sequences per split to pre-allocate memmap files
    split_counts: Dict[str, int] = {'train': 0, 'val': 0, 'test': 0}
    for d in all_anchor_dates:
        sp = dsm.get(int(d), 'gap')
        if sp in split_counts:
            split_counts[sp] += 1
    # Add 5% margin to handle slight over-counting from stocks that fail feature eng.
    for sp in split_counts:
        split_counts[sp] = int(split_counts[sp] * 1.05) + 100

    print(f"  Pre-allocating memmaps (with 5% margin):")
    for sp, n in split_counts.items():
        obs_gb = n * seq_len * NUM_DT_OBSERVED_PAST * 4 / 1e9
        print(f"    {sp}: {n:,} slots  ({obs_gb:.1f} GB obs)")
    writer.setup(split_counts)
    del all_anchor_dates; gc.collect()

    # ── PASS 2: Full feature engineering — writes directly to memmaps ─────────
    print(f"\nPass 2: processing {len(viable_stocks)} stocks to deeptime cache...")
    processed = skipped = total_seqs = 0
    skip_reasons: Dict[str, int] = {}

    _future_idx = np.array(_DT_FUTURE_FEAT_IDX, dtype=np.intp)
    _obs_idx    = np.array(_DT_OBS_PAST_FEAT_IDX, dtype=np.intp)

    stock_files_pass2 = viable_stocks  # only process stocks that passed the date scan

    for ts_code, filepath in stock_files_pass2:
        try:
            df = pd.read_csv(filepath)
            if len(df) < min_rows:
                skip_reasons['too_few_rows'] = skip_reasons.get('too_few_rows', 0) + 1
                skipped += 1
                continue

            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('trade_date').reset_index(drop=True)

            # ── Daily basic merge ───────────────────────────────────────────
            bare = str(ts_code).split('.')[0]
            stock_basic = daily_basic_dict.get(bare, pd.DataFrame())
            df = merge_daily_basic(df, stock_basic, ts_code=None)

            if daily_cs_stats:
                df = apply_cs_normalization(df, daily_cs_stats)

            # ── Market context ─────────────────────────────────────────────
            if market_context is not None:
                df = merge_market_context(df, market_context)
            else:
                from dl.config import MARKET_CONTEXT_FEATURES
                for col in MARKET_CONTEXT_FEATURES:
                    df[col] = 0.0

            # ── Index membership ───────────────────────────────────────────
            if index_membership is not None:
                df = merge_index_membership(df, index_membership, ts_code)
            else:
                from dl.config import INDEX_MEMBERSHIP_FEATURES
                for col in INDEX_MEMBERSHIP_FEATURES:
                    df[col] = 0.0

            # ── Price limits ───────────────────────────────────────────────
            df = merge_stk_limit(df, stk_limit_dict.get(bare), None)
            df = compute_price_limit_ratios(df, stk_limit_dict.get(bare))

            # ── Money flow ─────────────────────────────────────────────────
            df = merge_moneyflow(df, moneyflow_dict.get(bare), None)
            df = compute_extended_moneyflow(df, moneyflow_dict.get(bare))

            # ── Technical features ─────────────────────────────────────────
            df = calculate_technical_features(df)

            if cs_tech_stats:
                df = apply_cs_normalization(df, cs_tech_stats, DT_CS_NORMALIZE_TECH_FEATURES)

            # ── Quarterly fundamentals ─────────────────────────────────────
            fina_df = fina_data.get(bare) if fina_data else None
            df = forward_fill_fundamentals(df, fina_df)

            # ── Block trades ───────────────────────────────────────────────
            bt_df = block_trade_by_stock.get(bare) if block_trade_by_stock else None
            df = merge_block_trade_features(df, bt_df)

            # ── Ensure all DT_FEATURE_COLUMNS exist ────────────────────────
            for col in DT_FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0

            n_before = len(df)
            df = df.dropna(subset=DT_FEATURE_COLUMNS)
            n_dropped = n_before - len(df)
            if len(df) < seq_len + max_fw:
                reason = (f'too_short_after_dropna(dropped {n_dropped}/{n_before} rows)'
                          if n_dropped else 'too_short')
                skip_reasons[reason.split('(')[0]] = skip_reasons.get(reason.split('(')[0], 0) + 1
                skipped += 1
                del df
                continue

            features  = df[DT_FEATURE_COLUMNS].values.astype('float32')
            closes    = df['close'].values
            date_ints = (
                df['trade_date'].dt.year  * 10000 +
                df['trade_date'].dt.month * 100   +
                df['trade_date'].dt.day
            ).values.astype(np.int32)

            valid_indices = list(range(seq_len, len(df) - max_fw + 1))
            if max_seqs and len(valid_indices) > max_seqs:
                valid_indices = list(np.random.choice(valid_indices, max_seqs, replace=False))

            if not valid_indices:
                skipped += 1
                continue

            # Extract feature arrays
            obs_seqs      = np.array([features[i - seq_len:i][:, _obs_idx]  for i in valid_indices], dtype='float32')
            future_seqs   = np.array([features[i:i + max_fw][:, _future_idx] for i in valid_indices], dtype='float32')
            anchor_dates  = np.array([int(date_ints[i - 1]) for i in valid_indices], dtype=np.int32)

            targets = compute_regression_targets(
                closes, valid_indices, date_ints, csi300_fw_rets, target_mode
            )
            # Clamp targets to ±30% and replace any stray NaN/inf with 0
            targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
            targets = np.clip(targets, -30.0, 30.0).astype('float32')

            # Static IDs — all 7 covariates for TFT static context
            def _get(d, key, fallback): return d.get(ts_code, d.get(bare, fallback))
            N = len(valid_indices)
            sec_id   = sector_to_id .get(_get(sector_dict,   ts_code, 'Unknown'), sector_to_id ['Unknown'])
            ind_id   = industry_to_id.get(_get(industry_dict, ts_code, 'Unknown'), industry_to_id['Unknown'])
            sub_id   = 0
            size_id  = size_decile_map.get(bare, 10)
            area_id  = area_to_id.get( _get(area_dict,  ts_code, 'Unknown'), area_to_id ['Unknown'])
            board_id = board_to_id.get(_get(board_dict, ts_code, 'Unknown'), board_to_id['Unknown'])
            ipo_id   = ipo_age_dict.get(ts_code, ipo_age_dict.get(bare, 6))

            def _fill(v): return np.full(N, v, dtype=np.int64)

            # Write directly to pre-allocated split memmaps (no RAM buffering)
            writer.write_batch(
                obs_seqs, future_seqs, targets,
                _fill(sec_id), _fill(ind_id), _fill(sub_id), _fill(size_id),
                anchor_dates,
                area_ids    = _fill(area_id),
                board_ids   = _fill(board_id),
                ipo_age_ids = _fill(ipo_id),
            )
            total_seqs += len(valid_indices)
            processed  += 1

        except Exception as e:
            skip_reasons['exception'] = skip_reasons.get('exception', 0) + 1
            skipped += 1
            continue

        del df
        gc.collect()

        if processed % 100 == 0 and processed > 0:
            print(f"  {processed}/{len(stock_files_pass2)} stocks, {total_seqs:,} sequences")

    print(f"\n  Done: {processed} processed, {skipped} skipped, {total_seqs:,} total sequences")
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    Skip reason '{reason}': {count} stocks")

    # Close: flush memmaps and write metadata with ACTUAL counts
    print(f"\nFinalizing cache (flushing {total_seqs:,} sequences)...")
    metadata = writer.close(split_mode=split_mode, config=config)
    return metadata
