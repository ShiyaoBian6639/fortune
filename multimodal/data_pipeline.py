"""
Data pipeline: align price/technical sequences with daily news embeddings.

Produces disk-based numpy arrays for fast DataLoader access:
    {cache_dir}/price_sequences.npy   (N, 30, 106)  float32
    {cache_dir}/news_sequences.npy    (N,  3, 768)  float32
    {cache_dir}/labels.npy            (N,)           int64
    {cache_dir}/dates.npy             (N,)           '<U8' str  (YYYYMMDD)

After build_aligned_dataset(), call split_and_normalize() to produce:
    {cache_dir}/{split}_price.npy
    {cache_dir}/{split}_news.npy
    {cache_dir}/{split}_labels.npy
    {cache_dir}/price_scaler.npz      (mean + scale for StandardScaler)
    {cache_dir}/metadata.json

Time-based split is mandatory: news is market-wide and temporally correlated,
so random splitting would allow future macro-events to leak into training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Reuse from dl/ ────────────────────────────────────────────────────────────
from dl.data_processing import (
    calculate_technical_features,
    merge_daily_basic,
    load_daily_basic_data,
)
from dl.config import FEATURE_COLUMNS

from .config import (
    MM_CLASS_BUCKETS,
    MM_NUM_CLASSES,
    NEWS_WINDOW,
    NEWS_START_DATE,
    NEWS_END_DATE,
)


# ─── Label assignment ─────────────────────────────────────────────────────────

def pct_to_label(pct_change: float) -> int:
    """Map a forward return percentage to a bucket label index.

    Uses MM_CLASS_BUCKETS (10 buckets, same as dl/CHANGE_BUCKETS).
    Formula: pct = (max(high[t:t+5]) - close[t-1]) / close[t-1] * 100
    """
    for i, (lo, hi, _) in enumerate(MM_CLASS_BUCKETS):
        if lo <= pct_change < hi:
            return i
    return MM_NUM_CLASSES - 1   # fallback to last bucket


# ─── News window lookup ───────────────────────────────────────────────────────

def load_news_window(
    date_str:         str,
    daily_news_cache: Dict[str, np.ndarray],
    window:           int,
    trading_calendar: List[str],
    _cal_index:       Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Return (window, 768) float32 — one embedding per trading day in the window.

    The window covers the `window` trading days ending on date_str inclusive:
      [T-(window-1), ..., T-1, T]

    Walk back through `trading_calendar` (sorted YYYYMMDD strings) to find
    previous trading days — this correctly skips weekends and holidays.

    Missing days (no cache entry) → zero vector of shape (768,).

    Pass `_cal_index` (a pre-built {date: position} dict) for O(1) lookup
    instead of O(N) list.index() — critical when called millions of times.
    """
    if _cal_index is not None:
        idx = _cal_index.get(date_str, -1)
        if idx == -1:
            return np.zeros((window, 768), dtype=np.float32)
    else:
        try:
            idx = trading_calendar.index(date_str)
        except ValueError:
            return np.zeros((window, 768), dtype=np.float32)

    vecs = []
    zero = np.zeros(768, dtype=np.float32)
    for offset in range(window - 1, -1, -1):   # window-1, window-2, ..., 0
        cal_idx = idx - offset
        if cal_idx < 0:
            vecs.append(zero)
        else:
            day = trading_calendar[cal_idx]
            vecs.append(daily_news_cache.get(day, zero))

    return np.stack(vecs).astype(np.float32)   # (window, 768)


# ─── Per-stock sequence builder ───────────────────────────────────────────────

def build_stock_sequences(
    stock_file:    str,
    config:        dict,
    daily_basic:   Optional[pd.DataFrame] = None,
    _prefiltered:  bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load one stock CSV, compute 106 features, create 30-day sequences.

    Args:
        daily_basic:  Pre-loaded daily basic DataFrame.  When ``_prefiltered``
                      is True, ``daily_basic`` already contains only rows for
                      this stock (ts_code filter already applied by the caller),
                      so ``merge_daily_basic`` is called with ``ts_code=None``
                      to skip the O(9.8M) scan.
        _prefiltered: Set True when the caller passes a pre-grouped per-stock
                      slice of daily_basic (avoids redundant full-table scan).

    Returns:
        sequences  : (N, 30, 106) float32
        labels     : (N,)          int64
        date_strs  : (N,)          str — trade_date of the LAST day in each seq

    Returns None if the stock has insufficient data.
    """
    seq_len     = config.get('sequence_length', 30)
    fwd_window  = config.get('forward_window',   5)
    news_start  = config.get('news_start_date',  NEWS_START_DATE)
    news_end    = config.get('news_end_date',    NEWS_END_DATE)
    news_only   = config.get('news_only_mode',   True)

    try:
        df = pd.read_csv(stock_file)
    except Exception:
        return None

    if len(df) < config.get('min_data_points', 100):
        return None

    # Normalise date column
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    df = df.sort_values('trade_date').reset_index(drop=True)

    ts_code = os.path.basename(stock_file).replace('.csv', '')

    # Merge pre-loaded daily fundamental metrics
    if daily_basic is not None and not daily_basic.empty:
        try:
            # When _prefiltered=True the caller already extracted the per-stock
            # slice — pass ts_code=None to skip the O(9.8M) scan inside merge.
            df = merge_daily_basic(df, daily_basic, ts_code if not _prefiltered else None)
        except Exception:
            pass   # continue without fundamentals

    # Compute 106 technical features (reuse dl/ pipeline)
    try:
        df = calculate_technical_features(df)
    except Exception:
        return None

    # Fill any FEATURE_COLUMNS that are still absent (e.g. daily_basic merge
    # failed) with 0 so dropna doesn't raise KeyError.
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        df[missing_cols] = 0.0

    df = df.dropna(subset=FEATURE_COLUMNS)
    if len(df) < seq_len + fwd_window:
        return None

    features = df[FEATURE_COLUMNS].values.astype(np.float32)
    closes   = df['close'].values.astype(np.float64)
    highs    = df['high'].values.astype(np.float64)

    # Build YYYYMMDD string column for news lookup
    df['date_str'] = df['trade_date'].dt.strftime('%Y%m%d')

    sequences  = []
    labels     = []
    date_strs  = []

    valid_end = len(df) - fwd_window  # last index we can label

    for i in range(seq_len, valid_end + 1):
        date_str = df['date_str'].iloc[i - 1]   # last day of the window

        # Optionally restrict to news-covered dates
        if news_only and not (news_start <= date_str <= news_end):
            continue

        seq = features[i - seq_len: i]           # (30, 106)
        pct = 100.0 * (highs[i: i + fwd_window].max() - closes[i - 1]) / closes[i - 1]
        lbl = pct_to_label(float(pct))

        sequences.append(seq)
        labels.append(lbl)
        date_strs.append(date_str)

    if not sequences:
        return None

    return (
        np.array(sequences,  dtype=np.float32),
        np.array(labels,     dtype=np.int64),
        np.array(date_strs,  dtype='<U8'),
    )


# ─── Dataset builder ──────────────────────────────────────────────────────────

def build_aligned_dataset(
    stock_files:      List[str],
    daily_basic_dir:  str,
    daily_news_cache: Dict[str, np.ndarray],
    config:           dict,
    output_cache_dir: str,
    trading_calendar: Optional[List[str]] = None,
) -> None:
    """
    Process all stocks, attach news windows, write sorted arrays to disk.

    Memory-efficient two-pass implementation — avoids loading all ~5M+ samples
    into RAM simultaneously (would require ~126 GB for price + news arrays).

    Pass 1 — count valid samples per stock; no large arrays kept in memory.
    Pass 2 — write price/news sequences directly to pre-allocated memory-mapped
              .npy files on disk, one stock at a time.
    Sort   — labels and dates (~230 MB) are sorted in RAM; price and news
              (~126 GB) are sorted via chunked memmap I/O so only ~600 MB
              is in RAM at any point during the sort.

    Output files (sorted by date):
        price_sequences.npy  (N, seq_len, n_feat)  float32  — memmap-friendly
        news_sequences.npy   (N, news_window, 768)  float32  — memmap-friendly
        labels.npy           (N,)                   int64
        dates.npy            (N,)                   '<U8' str (YYYYMMDD)
    """
    os.makedirs(output_cache_dir, exist_ok=True)

    news_window = config.get('news_window', NEWS_WINDOW)
    seq_len     = config.get('sequence_length', 30)
    n_feat      = len(FEATURE_COLUMNS)

    if trading_calendar is None:
        trading_calendar = sorted(daily_news_cache.keys())
    cal_index: Dict[str, int] = {d: i for i, d in enumerate(trading_calendar)}

    # Load daily_basic once, then pre-group by ts_code for O(1) per-stock lookup.
    # Without pre-grouping, merge_daily_basic scans all 9.8M rows per stock —
    # that's 9.8M × 3752 = 36.7B row comparisons for the full dataset.
    daily_basic_df: Optional[pd.DataFrame] = None
    daily_basic_by_ts: Dict[str, pd.DataFrame] = {}
    if daily_basic_dir and os.path.isdir(daily_basic_dir):
        try:
            daily_basic_df = load_daily_basic_data(daily_basic_dir)
            if daily_basic_df is not None and not daily_basic_df.empty:
                print("[data_pipeline] Pre-grouping daily_basic by ts_code ...")
                daily_basic_by_ts = {
                    code: grp.reset_index(drop=True)
                    for code, grp in daily_basic_df.groupby('ts_code', sort=False)
                }
                del daily_basic_df   # free 9.8M-row DataFrame; dict is enough
        except Exception as e:
            print(f"[data_pipeline] Warning: could not load daily_basic: {e}")

    # ── Fast upper-bound estimation for memmap pre-allocation ────────────────
    # Count CSV lines (no parsing) to get a cheap upper bound on total samples.
    # Actual sample count is lower (NaN drops, news_only filter, etc.); we
    # trim the memmaps to the real cursor position after the single full pass.
    print(f"[data_pipeline] Estimating dataset size ({len(stock_files)} stocks) ...")
    seq_len_cfg   = config.get('sequence_length', 30)
    fwd_win_cfg   = config.get('forward_window',   5)
    min_pts_cfg   = config.get('min_data_points', 100)
    max_N = 0
    for sf in stock_files:
        try:
            with open(sf, 'r', errors='ignore') as fh:
                rows = sum(1 for _ in fh) - 1   # subtract header
            if rows >= min_pts_cfg:
                max_N += max(0, rows - seq_len_cfg - fwd_win_cfg)
        except Exception:
            pass
    max_N = int(max_N * 1.05)   # 5% safety margin
    if max_N == 0:
        print("[data_pipeline] No estimable samples. Aborting.")
        return
    print(f"[data_pipeline] Pre-allocated size: {max_N:,} samples (upper bound)")

    # ── Allocate memory-mapped temp files ────────────────────────────────────
    raw_price_path = os.path.join(output_cache_dir, '_raw_price.npy')
    raw_news_path  = os.path.join(output_cache_dir, '_raw_news.npy')

    price_mm = np.lib.format.open_memmap(
        raw_price_path, mode='w+', dtype=np.float32,
        shape=(max_N, seq_len, n_feat),
    )
    news_mm  = np.lib.format.open_memmap(
        raw_news_path,  mode='w+', dtype=np.float32,
        shape=(max_N, news_window, 768),
    )

    # ── Single pass: build sequences and write to memmaps ───────────────────
    print(f"[data_pipeline] Building sequences and writing to disk ...")
    all_labels: List[np.ndarray] = []
    all_dates:  List[np.ndarray] = []
    cursor    = 0
    ok        = 0
    skipped   = 0

    for sf in tqdm(stock_files, desc='  building', unit='stock', leave=False):
        ts_code   = os.path.basename(sf).replace('.csv', '')
        stk_basic = daily_basic_by_ts.get(ts_code)   # O(1) dict lookup

        result = build_stock_sequences(
            sf, config,
            daily_basic=stk_basic,
            _prefiltered=(stk_basic is not None),
        )
        if result is None:
            skipped += 1
            continue
        seqs, lbls, dates = result
        n = len(lbls)

        news_windows = np.stack([
            load_news_window(d, daily_news_cache, news_window, trading_calendar, cal_index)
            for d in dates
        ]).astype(np.float32)

        price_mm[cursor:cursor + n] = seqs
        news_mm[cursor:cursor + n]  = news_windows
        all_labels.append(lbls)
        all_dates.append(dates)
        cursor += n
        ok     += 1

    price_mm.flush(); del price_mm
    news_mm.flush();  del news_mm

    print(f"[data_pipeline] {ok} stocks OK, {skipped} skipped → {cursor:,} total samples")

    labels_arr = np.concatenate(all_labels)   # (cursor,)  ~46 MB
    dates_arr  = np.concatenate(all_dates)    # (cursor,)  ~185 MB
    del all_labels, all_dates

    # ── Sort by date ─────────────────────────────────────────────────────────
    # labels and dates fit in RAM → sort directly.
    # price and news are sorted via chunked memmap reads (~600 MB per chunk).
    print(f"[data_pipeline] Sorting {cursor:,} samples by date ...")
    sort_idx   = np.argsort(dates_arr, kind='stable')
    labels_arr = labels_arr[sort_idx]
    dates_arr  = dates_arr[sort_idx]

    price_raw = np.load(raw_price_path, mmap_mode='r')
    news_raw  = np.load(raw_news_path,  mmap_mode='r')

    final_price_path = os.path.join(output_cache_dir, 'price_sequences.npy')
    final_news_path  = os.path.join(output_cache_dir, 'news_sequences.npy')

    price_out = np.lib.format.open_memmap(
        final_price_path, mode='w+', dtype=np.float32,
        shape=(cursor, seq_len, n_feat),
    )
    news_out  = np.lib.format.open_memmap(
        final_news_path,  mode='w+', dtype=np.float32,
        shape=(cursor, news_window, 768),
    )

    SORT_CHUNK = 10_000
    for start in tqdm(range(0, cursor, SORT_CHUNK), desc='  sorting', unit='chunk', leave=False):
        end = min(start + SORT_CHUNK, cursor)
        idx = sort_idx[start:end]
        price_out[start:end] = price_raw[idx]
        news_out[start:end]  = news_raw[idx]

    price_out.flush(); del price_out
    news_out.flush();  del news_out
    del price_raw, news_raw

    # Remove temp files
    for p in [raw_price_path, raw_news_path]:
        try:
            os.remove(p)
        except Exception:
            pass

    np.save(os.path.join(output_cache_dir, 'labels.npy'), labels_arr)
    np.save(os.path.join(output_cache_dir, 'dates.npy'),  dates_arr)
    print(f"[data_pipeline] Sorted arrays saved to {output_cache_dir}")


def build_predict_sequences(
    stock_file:    str,
    config:        dict,
    scaler_mean:   np.ndarray,
    scaler_scale:  np.ndarray,
    daily_basic:   Optional[pd.DataFrame] = None,
    _prefiltered:  bool = False,
) -> Optional[Tuple[np.ndarray, str, str]]:
    """
    Extract the most recent 30-day sequence for one stock (inference only).

    Applies the training StandardScaler so the model sees normalized features,
    identical to what was used during training.

    Returns:
        price_seq : (seq_len, n_feat) float32  — scaler-normalized + clipped
        ts_code   : stock code string  (e.g. '600000.SH')
        date_str  : YYYYMMDD of the last day in the sequence (the 'prediction date')

    Returns None if the stock has insufficient data.
    """
    seq_len = config.get('sequence_length', 30)

    try:
        df = pd.read_csv(stock_file)
    except Exception:
        return None

    if len(df) < seq_len:
        return None

    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    df = df.sort_values('trade_date').reset_index(drop=True)

    ts_code = os.path.basename(stock_file).replace('.csv', '')

    if daily_basic is not None and not daily_basic.empty:
        try:
            df = merge_daily_basic(df, daily_basic, ts_code if not _prefiltered else None)
        except Exception:
            pass

    try:
        df = calculate_technical_features(df)
    except Exception:
        return None

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        df[missing_cols] = 0.0

    df = df.dropna(subset=FEATURE_COLUMNS)
    if len(df) < seq_len:
        return None

    # Take the last seq_len rows
    last     = df.tail(seq_len).reset_index(drop=True)
    date_str = last['trade_date'].iloc[-1].strftime('%Y%m%d')

    features = last[FEATURE_COLUMNS].values.astype(np.float32)   # (seq_len, n_feat)

    # Apply the training StandardScaler (same transform as split_and_normalize)
    n_feat = features.shape[1]
    scale  = np.where(scaler_scale == 0, 1.0, scaler_scale)
    flat   = (features.reshape(-1, n_feat) - scaler_mean) / scale
    flat   = np.clip(flat, -10.0, 10.0)
    features = flat.reshape(seq_len, n_feat).astype(np.float32)

    return features, ts_code, date_str


def split_and_normalize(
    cache_dir:   str,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> None:
    """
    Time-based split + StandardScaler on price features.

    Memory-efficient: price and news arrays are accessed as read-only memmaps
    (never fully loaded into RAM).  StandardScaler is fit incrementally via
    partial_fit so the full training price array (~50 GB) is never resident.
    Each split is written to disk in chunks via memory-mapped output files.

    StandardScaler is applied to price features ONLY.
    News embeddings (BERT outputs) are not re-scaled: BERT layer-norms
    already centre and scale them; additional normalisation degrades quality.
    """
    price  = np.load(os.path.join(cache_dir, 'price_sequences.npy'), mmap_mode='r')
    news   = np.load(os.path.join(cache_dir, 'news_sequences.npy'),  mmap_mode='r')
    labels = np.load(os.path.join(cache_dir, 'labels.npy'))   # (N,)  ~46 MB — fine in RAM
    dates  = np.load(os.path.join(cache_dir, 'dates.npy'))    # (N,) ~185 MB — fine in RAM

    N       = len(labels)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)

    splits = {
        'train': (0,               n_train),
        'val':   (n_train,         n_train + n_val),
        'test':  (n_train + n_val, N),
    }

    # Fit StandardScaler incrementally — avoids loading full training price
    # array (~50 GB) into RAM at once.
    CHUNK = 50_000
    print(f"[data_pipeline] Fitting StandardScaler on {n_train:,} training samples ...")
    scaler = StandardScaler()
    for start in range(0, n_train, CHUNK):
        end   = min(start + CHUNK, n_train)
        scaler.partial_fit(np.array(price[start:end]).reshape(-1, price.shape[-1]))

    metadata: dict = {
        'n_total':    N,
        'seq_length': price.shape[1],
        'n_features': price.shape[2],
        'bert_dim':   news.shape[2],
        'news_window': news.shape[1],
        'splits':     {},
    }

    for split_name, (start, end) in splits.items():
        n = end - start
        if n == 0:
            metadata['splits'][split_name] = {
                'n_samples': 0, 'date_start': '', 'date_end': '', 'class_counts': []
            }
            continue

        # Write split files as memmaps (avoids holding full split in RAM)
        p_out  = np.lib.format.open_memmap(
            os.path.join(cache_dir, f'{split_name}_price.npy'),
            mode='w+', dtype=np.float32, shape=(n, price.shape[1], price.shape[2]),
        )
        nv_out = np.lib.format.open_memmap(
            os.path.join(cache_dir, f'{split_name}_news.npy'),
            mode='w+', dtype=np.float32, shape=(n, news.shape[1], news.shape[2]),
        )

        for c_start in range(0, n, CHUNK):
            c_end  = min(c_start + CHUNK, n)
            src_s  = start + c_start
            src_e  = start + c_end

            # np.array() copies the memmap slice into RAM for transform
            p_chunk = np.array(price[src_s:src_e])
            shape   = p_chunk.shape
            p_norm  = scaler.transform(
                p_chunk.reshape(-1, shape[-1])
            ).reshape(shape).astype(np.float32)
            p_norm  = np.clip(p_norm, -10.0, 10.0)

            p_out[c_start:c_end]  = p_norm
            nv_out[c_start:c_end] = news[src_s:src_e]

        p_out.flush();  del p_out
        nv_out.flush(); del nv_out

        # labels and dates are already in RAM — slice and save directly
        np.save(os.path.join(cache_dir, f'{split_name}_labels.npy'), labels[start:end])
        np.save(os.path.join(cache_dir, f'{split_name}_dates.npy'),  dates[start:end])

        class_counts = np.bincount(labels[start:end], minlength=MM_NUM_CLASSES).tolist()
        metadata['splits'][split_name] = {
            'n_samples':    n,
            'date_start':   str(dates[start]),
            'date_end':     str(dates[end - 1]),
            'class_counts': class_counts,
        }
        print(
            f"  {split_name:5s}: {n:7,} samples  "
            f"dates {dates[start]} → {dates[end - 1]}  "
            f"classes {class_counts}"
        )

    np.savez(
        os.path.join(cache_dir, 'price_scaler.npz'),
        mean=scaler.mean_,
        scale=scaler.scale_,
    )

    with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[data_pipeline] Split + normalisation complete → {cache_dir}")
