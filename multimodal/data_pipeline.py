"""
Compact-cache data pipeline for the multimodal stock prediction package.

Storage strategy
----------------
Earlier versions materialised per-sample arrays of shape ``(N, 30, F)`` for
price and ``(N, 3, 768)`` for news.  With ~7.7 M samples that consumed
hundreds of GB on disk despite massive redundancy: news vectors are
market-wide (identical across stocks for the same date) and price windows
overlap by 29 of 30 rows between consecutive samples.

This module instead stores **one row per (stock, date) pair** plus a tiny
per-sample index that points into that matrix.  Slicing recovers the 30-day
window at __getitem__ time.  News embeddings are stored once per date and
looked up via a per-sample ``date_idx``.

Output files (in ``cache_dir``):

    price_matrix.npy      (T_total, F)  float32  — concatenated per-stock daily
                                                   features, normalised in place
    price_date_idx.npy    (T_total,)    int32    — date_idx per row (used to
                                                   pick training rows for the
                                                   StandardScaler fit)
    price_scaler.npz                              — mean + scale (per-feature)
    stock_offsets.npy     (S+1,)        int64    — cumulative row counts;
                                                   stock k owns rows
                                                   [stock_offsets[k] :
                                                    stock_offsets[k+1])
    stock_codes.npy       (S,)          str      — ts_code per stock, parallel
                                                   to stock_offsets
    sample_end.npy        (N,)          int32    — last row (inclusive) of the
                                                   30-day window for each
                                                   sample, in price_matrix
    sample_date_idx.npy   (N,)          int32    — index into trading_calendar
    sample_labels.npy     (N,)          int32    — bucket label
    trading_calendar.npy  (D,)          str      — sorted YYYYMMDD calendar
                                                   used for date_idx semantics
    metadata.json                                 — n_features, sequence_length,
                                                   forward_window, news_window,
                                                   splits[ {start,end,
                                                   date_start,date_end,
                                                   class_counts,n_samples} ],
                                                   news_cache_path,
                                                   token_cache_path

Sample arrays are sorted by ``date_idx`` so split = contiguous slice.
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
    """Map a forward return percentage to a bucket label index."""
    for i, (lo, hi, _) in enumerate(MM_CLASS_BUCKETS):
        if lo <= pct_change < hi:
            return i
    return MM_NUM_CLASSES - 1


# ─── News window lookup (used by run_predict) ────────────────────────────────

def load_news_window(
    date_str:         str,
    daily_news_cache: Dict[str, np.ndarray],
    window:           int,
    trading_calendar: List[str],
    _cal_index:       Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Return (window, 768) float32 — embeddings for the trailing trading days."""
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
    for offset in range(window - 1, -1, -1):
        cal_idx = idx - offset
        if cal_idx < 0:
            vecs.append(zero)
        else:
            day = trading_calendar[cal_idx]
            vecs.append(daily_news_cache.get(day, zero))
    return np.stack(vecs).astype(np.float32)


# ─── Per-stock feature builder ────────────────────────────────────────────────

def _build_stock_features(
    stock_file:   str,
    config:       dict,
    cal_index:    Dict[str, int],
    daily_basic:  Optional[pd.DataFrame] = None,
    _prefiltered: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]]:
    """
    Process one stock CSV and return:

        features    : (T_stock, F)  float32   — daily feature matrix
        date_idx    : (T_stock,)    int32     — index into trading_calendar per row
        samples     : list of (rel_end_row, date_idx, label)
                      — one entry per emittable training sample

    ``rel_end_row`` is the row index *within this stock* (0-based) of the last
    day in a 30-day window.  The caller adds the stock's ``stock_offset`` to
    convert it to an absolute row in ``price_matrix``.

    Returns ``None`` if the stock has insufficient data after dropna +
    news-window filtering.
    """
    seq_len    = config.get('sequence_length', 30)
    fwd_window = config.get('forward_window',   5)
    news_start = config.get('news_start_date',  NEWS_START_DATE)
    news_end   = config.get('news_end_date',    NEWS_END_DATE)
    news_only  = config.get('news_only_mode',   True)
    min_pts    = config.get('min_data_points',  100)

    try:
        df = pd.read_csv(stock_file)
    except Exception:
        return None

    if len(df) < min_pts:
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

    # Add any missing FEATURE_COLUMNS as zero — assemble in one concat to avoid
    # the pandas "highly fragmented" PerformanceWarning that the old per-column
    # ``df[missing_cols] = 0.0`` triggered for hundreds of inserts.
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        zeros = pd.DataFrame(
            0.0, index=df.index, columns=missing_cols, dtype=np.float32
        )
        df = pd.concat([df, zeros], axis=1)

    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if len(df) < seq_len + fwd_window:
        return None

    features  = df[FEATURE_COLUMNS].values.astype(np.float32)
    closes    = df['close'].values.astype(np.float64)
    highs     = df['high'].values.astype(np.float64)
    date_strs = df['trade_date'].dt.strftime('%Y%m%d').values

    date_idx_arr = np.fromiter(
        (cal_index.get(d, -1) for d in date_strs),
        count=len(date_strs), dtype=np.int32,
    )

    samples: List[Tuple[int, int, int]] = []
    T = len(features)
    # End-of-window position is inclusive: window = features[end-seq_len+1 : end+1]
    # Forward returns use highs[end+1 : end+1+fwd_window] vs closes[end].
    for end in range(seq_len - 1, T - fwd_window):
        ds = date_strs[end]
        if news_only and not (news_start <= ds <= news_end):
            continue
        d_idx = int(date_idx_arr[end])
        if d_idx < 0:
            continue
        pct = 100.0 * (highs[end + 1: end + 1 + fwd_window].max() - closes[end]) / closes[end]
        samples.append((end, d_idx, pct_to_label(float(pct))))

    if not samples:
        return None

    return features, date_idx_arr, samples


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
    Build the compact cache (see module docstring).

    Disk usage at peak: roughly ``T_total × F × 4`` bytes for the price matrix
    (≈6 GB for the full dataset).  All other arrays are tiny (<200 MB combined).
    """
    os.makedirs(output_cache_dir, exist_ok=True)

    # Reclaim disk from the previous (per-sample-array) cache layout if the
    # user is re-running after the storage refactor.  Each of these old files
    # could be tens or hundreds of GB on a full-dataset run.
    _OLD_FORMAT_FILES = (
        'price_sequences.npy', 'news_sequences.npy',
        '_raw_price.npy',      '_raw_news.npy',
        'train_price.npy', 'train_news.npy', 'train_labels.npy', 'train_dates.npy',
        'val_price.npy',   'val_news.npy',   'val_labels.npy',   'val_dates.npy',
        'test_price.npy',  'test_news.npy',  'test_labels.npy',  'test_dates.npy',
        'labels.npy',      'dates.npy',
    )
    for fname in _OLD_FORMAT_FILES:
        p = os.path.join(output_cache_dir, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"[data_pipeline] removed legacy cache file {fname}")
            except Exception as e:
                print(f"[data_pipeline] could not remove legacy {fname}: {e}")

    seq_len = config.get('sequence_length', 30)
    n_feat  = len(FEATURE_COLUMNS)

    if trading_calendar is None:
        trading_calendar = sorted(daily_news_cache.keys())
    cal_index: Dict[str, int] = {d: i for i, d in enumerate(trading_calendar)}

    # ── Load daily_basic (optional) ──────────────────────────────────────────
    daily_basic_by_ts: Dict[str, pd.DataFrame] = {}
    if daily_basic_dir and os.path.isdir(daily_basic_dir):
        try:
            db = load_daily_basic_data(daily_basic_dir)
            if db is not None and not db.empty:
                print("[data_pipeline] Pre-grouping daily_basic by ts_code ...")
                daily_basic_by_ts = {
                    code: grp.reset_index(drop=True)
                    for code, grp in db.groupby('ts_code', sort=False)
                }
                del db
        except Exception as e:
            print(f"[data_pipeline] Warning: could not load daily_basic: {e}")

    # ── Estimate upper bound on price_matrix rows ───────────────────────────
    # Sum CSV row counts (no parse).  Real T_total is lower (NaN drops) but
    # this gives a safe pre-allocation size that we trim later.
    print(f"[data_pipeline] Estimating dataset size ({len(stock_files)} stocks) ...")
    min_pts = config.get('min_data_points', 100)
    upper_T = 0
    for sf in stock_files:
        try:
            with open(sf, 'r', errors='ignore') as fh:
                rows = sum(1 for _ in fh) - 1
            if rows >= min_pts:
                upper_T += rows
        except Exception:
            pass
    upper_T = int(upper_T * 1.05)
    if upper_T == 0:
        print("[data_pipeline] No estimable data. Aborting.")
        return
    est_gb = upper_T * n_feat * 4 / 1e9
    print(f"[data_pipeline] Pre-allocating price matrix: "
          f"{upper_T:,} rows × {n_feat} features ≈ {est_gb:.1f} GB")

    raw_price_path = os.path.join(output_cache_dir, '_price_matrix_raw.npy')
    price_mm = np.lib.format.open_memmap(
        raw_price_path, mode='w+', dtype=np.float32, shape=(upper_T, n_feat),
    )
    price_date_idx_buf = np.zeros(upper_T, dtype=np.int32)

    # ── Build per-stock features and samples ────────────────────────────────
    print(f"[data_pipeline] Building per-stock features and samples ...")
    stock_codes_kept: List[str] = []
    stock_offsets:    List[int] = [0]
    sample_end_buf:        List[int] = []
    sample_date_idx_buf:   List[int] = []
    sample_label_buf:      List[int] = []

    cursor   = 0
    skipped  = 0
    for sf in tqdm(stock_files, desc='  building', unit='stock', leave=False):
        ts_code   = os.path.basename(sf).replace('.csv', '')
        stk_basic = daily_basic_by_ts.get(ts_code)

        result = _build_stock_features(
            sf, config, cal_index,
            daily_basic=stk_basic, _prefiltered=(stk_basic is not None),
        )
        if result is None:
            skipped += 1
            continue
        features, date_idx_arr, samples = result
        T_stock = len(features)

        if cursor + T_stock > upper_T:
            print(f"[data_pipeline] WARNING: upper bound exceeded at {ts_code}; skipping")
            skipped += 1
            continue

        price_mm[cursor: cursor + T_stock] = features
        price_date_idx_buf[cursor: cursor + T_stock] = date_idx_arr

        for rel_end, d_idx, lbl in samples:
            sample_end_buf.append(cursor + rel_end)
            sample_date_idx_buf.append(d_idx)
            sample_label_buf.append(lbl)

        cursor += T_stock
        stock_codes_kept.append(ts_code)
        stock_offsets.append(cursor)

    price_mm.flush()
    del price_mm

    if cursor == 0 or not sample_label_buf:
        print("[data_pipeline] No valid samples produced.")
        try:
            os.remove(raw_price_path)
        except Exception:
            pass
        return

    n_samples = len(sample_label_buf)
    print(f"[data_pipeline] {len(stock_codes_kept)} stocks OK, {skipped} skipped → "
          f"{cursor:,} feature rows, {n_samples:,} samples")

    # ── Trim price_matrix to the actual cursor size ─────────────────────────
    final_price_path = os.path.join(output_cache_dir, 'price_matrix.npy')
    print(f"[data_pipeline] Trimming price matrix to {cursor:,} rows ...")
    raw   = np.load(raw_price_path, mmap_mode='r')
    final = np.lib.format.open_memmap(
        final_price_path, mode='w+', dtype=np.float32, shape=(cursor, n_feat),
    )
    CHUNK = 200_000
    for s in tqdm(range(0, cursor, CHUNK), desc='  trim', unit='chunk', leave=False):
        e = min(s + CHUNK, cursor)
        final[s:e] = raw[s:e]
    final.flush()
    del final
    del raw
    try:
        os.remove(raw_price_path)
    except Exception:
        pass

    # ── Sample arrays: convert to numpy + sort by date_idx ──────────────────
    sample_end      = np.asarray(sample_end_buf,      dtype=np.int32)
    sample_date_idx = np.asarray(sample_date_idx_buf, dtype=np.int32)
    sample_labels   = np.asarray(sample_label_buf,    dtype=np.int32)

    print(f"[data_pipeline] Sorting {n_samples:,} samples by date_idx ...")
    sort_idx        = np.argsort(sample_date_idx, kind='stable')
    sample_end      = sample_end[sort_idx]
    sample_date_idx = sample_date_idx[sort_idx]
    sample_labels   = sample_labels[sort_idx]

    np.save(os.path.join(output_cache_dir, 'price_date_idx.npy'),
            price_date_idx_buf[:cursor])
    np.save(os.path.join(output_cache_dir, 'stock_offsets.npy'),
            np.asarray(stock_offsets, dtype=np.int64))
    np.save(os.path.join(output_cache_dir, 'stock_codes.npy'),
            np.array(stock_codes_kept))
    np.save(os.path.join(output_cache_dir, 'sample_end.npy'),      sample_end)
    np.save(os.path.join(output_cache_dir, 'sample_date_idx.npy'), sample_date_idx)
    np.save(os.path.join(output_cache_dir, 'sample_labels.npy'),   sample_labels)
    np.save(os.path.join(output_cache_dir, 'trading_calendar.npy'),
            np.array(trading_calendar))

    meta = {
        'n_features':       n_feat,
        'sequence_length':  seq_len,
        'forward_window':   config.get('forward_window', 5),
        'news_window':      config.get('news_window', NEWS_WINDOW),
        'n_total_samples':  int(n_samples),
        'n_price_rows':     int(cursor),
        'n_stocks':         len(stock_codes_kept),
        'n_calendar_days':  len(trading_calendar),
        'news_cache_path':  config.get('news_cache_path'),
        'token_cache_path': config.get('token_cache_path'),
    }
    with open(os.path.join(output_cache_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[data_pipeline] Compact cache saved → {output_cache_dir}")


# ─── Predict-only sequence builder (unchanged contract) ──────────────────────

def build_predict_sequences(
    stock_file:    str,
    config:        dict,
    scaler_mean:   np.ndarray,
    scaler_scale:  np.ndarray,
    daily_basic:   Optional[pd.DataFrame] = None,
    _prefiltered:  bool = False,
) -> Optional[Tuple[np.ndarray, str, str]]:
    """
    Most-recent (seq_len, n_feat) window for one stock, scaled with the
    training StandardScaler.  Returns (price_seq, ts_code, date_str) or None.
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
        zeros = pd.DataFrame(0.0, index=df.index, columns=missing_cols, dtype=np.float32)
        df = pd.concat([df, zeros], axis=1)

    df = df.dropna(subset=FEATURE_COLUMNS)
    if len(df) < seq_len:
        return None

    last     = df.tail(seq_len).reset_index(drop=True)
    date_str = last['trade_date'].iloc[-1].strftime('%Y%m%d')

    features = last[FEATURE_COLUMNS].values.astype(np.float32)

    n_feat = features.shape[1]
    scale  = np.where(scaler_scale == 0, 1.0, scaler_scale)
    flat   = (features.reshape(-1, n_feat) - scaler_mean) / scale
    flat   = np.clip(flat, -10.0, 10.0)
    features = flat.reshape(seq_len, n_feat).astype(np.float32)

    return features, ts_code, date_str


# ─── Split + scaler (compact-cache version) ──────────────────────────────────

def split_and_normalize(
    cache_dir:   str,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> None:
    """
    Record split boundaries in metadata, fit StandardScaler on training-period
    rows of ``price_matrix.npy``, and transform the matrix in place.

    No per-split copies are written.  The Dataset reads the master file and
    indexes into it using the ``[start, end)`` ranges saved in metadata.
    """
    meta_path = os.path.join(cache_dir, 'metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    sample_date_idx  = np.load(os.path.join(cache_dir, 'sample_date_idx.npy'))
    sample_labels    = np.load(os.path.join(cache_dir, 'sample_labels.npy'))
    trading_calendar = np.load(os.path.join(cache_dir, 'trading_calendar.npy'))
    price_date_idx   = np.load(os.path.join(cache_dir, 'price_date_idx.npy'))

    N       = int(meta['n_total_samples'])
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)
    splits  = {
        'train': (0,               n_train),
        'val':   (n_train,         n_train + n_val),
        'test':  (n_train + n_val, N),
    }

    # Training-period boundary: any price row whose date_idx is strictly less
    # than the first val sample's date_idx is "training-period".  This avoids
    # using val/test feature distributions in the scaler fit.
    if n_train >= N:
        val_start_date_idx = int(sample_date_idx[-1]) + 1
    else:
        val_start_date_idx = int(sample_date_idx[n_train])

    print(f"[data_pipeline] Fitting StandardScaler "
          f"(rows with date_idx < {val_start_date_idx}) ...")
    train_mask    = price_date_idx < val_start_date_idx
    n_train_rows  = int(train_mask.sum())
    if n_train_rows == 0:
        raise ValueError("No training-period rows found for scaler fit.")
    train_indices = np.where(train_mask)[0]

    price = np.load(os.path.join(cache_dir, 'price_matrix.npy'), mmap_mode='r+')
    n_rows, n_feat = price.shape

    scaler = StandardScaler()
    CHUNK  = 200_000
    for s in tqdm(range(0, n_train_rows, CHUNK), desc='  scaler fit', unit='chunk', leave=False):
        e        = min(s + CHUNK, n_train_rows)
        idx_chunk = train_indices[s:e]
        rows = np.asarray(price[idx_chunk])   # gather (chunk, n_feat)
        scaler.partial_fit(rows)

    mean  = scaler.mean_.astype(np.float32)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)

    # Apply scaler in place so the matrix is ready for direct __getitem__ reads.
    print(f"[data_pipeline] Applying scaler in place to {n_rows:,} rows ...")
    for s in tqdm(range(0, n_rows, CHUNK), desc='  normalise', unit='chunk', leave=False):
        e     = min(s + CHUNK, n_rows)
        chunk = np.asarray(price[s:e])
        chunk = (chunk - mean) / scale
        np.clip(chunk, -10.0, 10.0, out=chunk)
        price[s:e] = chunk.astype(np.float32)
    price.flush()
    del price

    np.savez(os.path.join(cache_dir, 'price_scaler.npz'), mean=mean, scale=scale)

    # ── Per-split metadata (n_samples, date range, class_counts) ─────────────
    splits_meta: Dict[str, dict] = {}
    for name, (start, end) in splits.items():
        n = end - start
        if n == 0:
            splits_meta[name] = {
                'n_samples':    0,
                'start':        int(start),
                'end':          int(end),
                'date_start':   '',
                'date_end':     '',
                'class_counts': [],
            }
            continue
        labels_split = sample_labels[start:end]
        class_counts = np.bincount(labels_split, minlength=MM_NUM_CLASSES).tolist()
        d_start = str(trading_calendar[int(sample_date_idx[start])])
        d_end   = str(trading_calendar[int(sample_date_idx[end - 1])])
        splits_meta[name] = {
            'n_samples':    int(n),
            'start':        int(start),
            'end':          int(end),
            'date_start':   d_start,
            'date_end':     d_end,
            'class_counts': class_counts,
        }
        print(f"  {name:5s}: {n:7,} samples  "
              f"dates {d_start} → {d_end}  classes {class_counts}")

    meta['splits']             = splits_meta
    meta['val_start_date_idx'] = int(val_start_date_idx)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[data_pipeline] Split + normalisation complete → {cache_dir}")
