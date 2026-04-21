#!/usr/bin/env python3
"""
Backtest: DL Transformer → Bucket Portfolio Strategy
=====================================================

Test window  : 2026-04-06 → 2026-04-10  (5 trading days)
Predict date : 2026-04-03  (last trading day before window)

Strategy
--------
1. Extend every stock's OHLCV data to 2026-04-10 (idempotent).
2. Run model predictions *as of* 2026-04-03 for ALL stocks.
3. Group stocks by their model-predicted bucket.
4. Drop buckets with negative max-gain  (buckets 0-2: < -5%, -5→-2%, -2→0%).
5. From each remaining (positive) bucket randomly draw up to N stocks.
6. Portfolio weight for each stock ∝ model confidence (probability of its bucket).
7. Entry = close on 2026-04-03; Exit = close on 2026-04-10.
8. Report per-stock P&L, bucket breakdown, and benchmark comparison.

Usage
-----
    # From project root:
    python -m backtest.strategy

    # Skip the data-extension step (data already current):
    python -m backtest.strategy --no-extend

    # Tune parameters:
    python -m backtest.strategy --n-per-bucket 5 --seed 7 --device cpu

    # Save predictions & results to CSV:
    python -m backtest.strategy --save-csv
"""

import os
import sys
import random
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ── Project path ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tushare as ts
from dl.config import CHANGE_BUCKETS, NUM_CLASSES, FEATURE_COLUMNS, TUSHARE_TOKEN
from dl.models import TransformerClassifier
from dl.data_processing import (
    calculate_technical_features,
    load_sector_data,
    load_daily_basic_data,
    merge_daily_basic,
    get_stock_files,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR        = ROOT / 'stock_data'
MODEL_PATH      = DATA_DIR / 'transformer_classifier.pth'

PREDICT_DATE    = '20260403'          # last trading day before test window
TEST_START      = '20260406'          # first day of test window (inclusive)
TEST_END        = '20260410'          # last day of test window  (inclusive)
EXTEND_TARGET   = TEST_END

N_STOCKS_PER_BUCKET = 10             # stocks to randomly draw from each bucket
CALL_INTERVAL   = 0.15               # seconds between tushare API calls

# Positive-gain buckets: those whose LOW end >= 0
# CHANGE_BUCKETS indices: 0='<-5%', 1='-5→-2%', 2='-2→0%', 3='0→1%' … 9='>15%'
POSITIVE_BUCKET_IDX = [
    i for i, (lo, hi, _) in enumerate(CHANGE_BUCKETS) if lo >= 0
]

CLASS_NAMES = [name for _, _, name in CHANGE_BUCKETS]

# Bucket midpoints used for expected-return calculation
BUCKET_MIDPOINTS = []
for lo, hi, _ in CHANGE_BUCKETS:
    if lo == -float('inf'):
        mid = hi - 5.0
    elif hi == float('inf'):
        mid = lo + 5.0
    else:
        mid = (lo + hi) / 2.0
    BUCKET_MIDPOINTS.append(mid)
BUCKET_MIDPOINTS = np.array(BUCKET_MIDPOINTS, dtype=np.float32)


# ── 1. Data extension ─────────────────────────────────────────────────────────

def _extend_one_stock(pro, ts_code: str, filepath: Path, target: str) -> str:
    """
    Fetch trading days from the stock's current latest date to `target`.
    Returns 'updated', 'up_to_date', or 'failed'.
    """
    try:
        df = pd.read_csv(filepath)
        df['trade_date'] = df['trade_date'].astype(str)
        latest = df['trade_date'].max()
    except Exception:
        return 'failed'

    if latest >= target:
        return 'up_to_date'

    # next calendar day after latest
    from datetime import datetime, timedelta
    nxt = (datetime.strptime(latest, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')

    try:
        new_df = pro.daily(ts_code=ts_code, start_date=nxt, end_date=target)
    except Exception as e:
        err = str(e)
        if any(k in err for k in ('exceed', 'limit', '频率')):
            time.sleep(60)
            try:
                new_df = pro.daily(ts_code=ts_code, start_date=nxt, end_date=target)
            except Exception:
                return 'failed'
        else:
            return 'failed'

    if new_df is None or new_df.empty:
        return 'up_to_date'

    combined = pd.concat([df, new_df], ignore_index=True)
    combined['trade_date'] = combined['trade_date'].astype(str)
    combined = combined.drop_duplicates(subset=['trade_date'])
    combined = combined.sort_values('trade_date', ascending=False)
    combined.to_csv(filepath, index=False)
    return 'updated'


def extend_data_to(target: str = EXTEND_TARGET):
    """
    Extend all sh/sz stock CSVs to include data through `target` (YYYYMMDD).
    Uses the same Tushare token as the rest of the project.
    """
    print(f"\n{'='*60}")
    print(f"Extending stock data → {target}")
    print(f"{'='*60}")

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)

    updated = failed = skipped = 0
    all_files = []
    for market in ['sh', 'sz']:
        d = DATA_DIR / market
        if d.exists():
            all_files.extend([(market, f) for f in d.glob('*.csv')])

    print(f"  Files to check: {len(all_files)}")

    for i, (market, fp) in enumerate(all_files, 1):
        code   = fp.stem
        suffix = 'SH' if market == 'sh' else 'SZ'
        ts_code = f"{code}.{suffix}"

        result = _extend_one_stock(pro, ts_code, fp, target)
        if result == 'updated':
            updated += 1
        elif result == 'failed':
            failed += 1
        else:
            skipped += 1

        if i % 200 == 0:
            print(f"  {i}/{len(all_files)}  updated={updated} skip={skipped} fail={failed}")

        time.sleep(CALL_INTERVAL)

    print(f"\n  Done — updated={updated} up-to-date={skipped} failed={failed}")


# ── 2. Prediction as-of a specific date ───────────────────────────────────────

def _load_checkpoint(device: str):
    """Load model, scaler, config, and sector data from the saved checkpoint."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    ckpt       = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    saved_cfg  = ckpt['config']

    scaler         = StandardScaler()
    scaler.mean_   = ckpt['scaler_mean']
    scaler.scale_  = ckpt['scaler_scale']

    sector_data    = load_sector_data(str(DATA_DIR))
    num_sectors    = len(sector_data['sector'].unique()) if len(sector_data) > 0 else 0

    model = TransformerClassifier(
        input_dim       = len(scaler.mean_),
        num_classes     = NUM_CLASSES,
        d_model         = saved_cfg['d_model'],
        nhead           = saved_cfg['nhead'],
        num_layers      = saved_cfg['num_layers'],
        dim_feedforward = saved_cfg['dim_feedforward'],
        dropout         = saved_cfg['dropout'],
        num_sectors     = num_sectors,
        use_sector      = (num_sectors > 0),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, scaler, saved_cfg, sector_data


def predict_as_of(
    as_of_date: str,
    device: str = 'cpu',
    batch_size: int = 512,
    save_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run model predictions for ALL stocks using data up to and including
    `as_of_date` (YYYYMMDD).  Returns a DataFrame sorted by expected_return.
    """
    print(f"\n{'='*60}")
    print(f"Batch Prediction  [as of {as_of_date}]")
    print(f"{'='*60}")

    model, scaler, saved_cfg, sector_data = _load_checkpoint(device)
    seq_len = saved_cfg['sequence_length']

    sector_to_id = {s: i for i, s in enumerate(sector_data['sector'].unique())} \
                   if len(sector_data) > 0 else {}
    sector_to_id['Unknown'] = len(sector_to_id)

    daily_basic = load_daily_basic_data(str(DATA_DIR))

    cutoff = pd.Timestamp(as_of_date)

    all_files: list[tuple] = []
    for market in ['sh', 'sz']:
        for code, fp in get_stock_files(str(DATA_DIR), market, max_stocks=None):
            all_files.append((market, code, fp))

    print(f"  Stock files found: {len(all_files)}")

    seqs, sec_ids, meta_rows, skipped = [], [], [], 0

    t0 = time.perf_counter()
    for i, (market, code, fp) in enumerate(all_files):
        if (i + 1) % 1000 == 0:
            print(f"  Feature extraction {i+1}/{len(all_files)} ...")

        ts_code = f"{code}.{'SH' if market == 'sh' else 'SZ'}"
        try:
            df = pd.read_csv(fp)
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('trade_date').reset_index(drop=True)
            df = df[df['trade_date'] <= cutoff]          # ← date cutoff
            if df.empty:
                skipped += 1
                continue
            df = merge_daily_basic(df, daily_basic, ts_code)
            df = calculate_technical_features(df)
            df = df.dropna(subset=FEATURE_COLUMNS)
        except Exception:
            skipped += 1
            continue

        if len(df) < seq_len:
            skipped += 1
            continue

        feat = df[FEATURE_COLUMNS].values[-seq_len:]
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        feat = scaler.transform(feat)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        sr   = sector_data[sector_data['ts_code'] == ts_code] \
               if len(sector_data) > 0 else pd.DataFrame()
        sname = sr['sector'].values[0] if len(sr) > 0 else 'Unknown'
        sid   = sector_to_id.get(sname, sector_to_id['Unknown'])

        seqs.append(feat)
        sec_ids.append(sid)
        meta_rows.append({
            'stock_code':   code,
            'ts_code':      ts_code,
            'market':       market.upper(),
            'sector':       sname,
            'predict_date': df['trade_date'].iloc[-1].strftime('%Y-%m-%d'),
            'entry_close':  float(df['close'].iloc[-1]),
        })

    print(f"  Feature extraction: {time.perf_counter()-t0:.1f}s  "
          f"valid={len(seqs)} skipped={skipped}")

    if not seqs:
        print("  No valid sequences — check DATA_DIR and as_of_date.")
        return pd.DataFrame()

    # ── Batched GPU inference ──────────────────────────────────────────────────
    seq_arr = np.array(seqs,    dtype=np.float32)   # (N, T, F)
    sec_arr = np.array(sec_ids, dtype=np.int64)      # (N,)
    all_probs = []

    use_amp = device.startswith('cuda')
    print(f"  Running inference: {len(seqs)} stocks, batch={batch_size} ...")
    t1 = time.perf_counter()

    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            end   = min(start + batch_size, len(seqs))
            seq_t = torch.from_numpy(seq_arr[start:end]).to(device, non_blocking=True)
            sec_t = torch.from_numpy(sec_arr[start:end]).to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(':')[0],
                                dtype=torch.float16, enabled=use_amp):
                logits = model(seq_t, sec_t)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs     = np.vstack(all_probs)                        # (N, C)
    pred_cls      = all_probs.argmax(axis=1)                    # (N,)
    exp_ret       = (all_probs * BUCKET_MIDPOINTS).sum(axis=1)  # (N,)
    print(f"  Inference: {time.perf_counter()-t1:.2f}s")

    # ── Build DataFrame ────────────────────────────────────────────────────────
    rows = []
    for i, meta in enumerate(meta_rows):
        row = dict(meta)
        row['predicted_class'] = int(pred_cls[i])
        row['predicted_label'] = CLASS_NAMES[pred_cls[i]]
        row['confidence']      = float(all_probs[i, pred_cls[i]])
        row['expected_return'] = float(exp_ret[i])
        row['bull_prob']       = float(all_probs[i, POSITIVE_BUCKET_IDX].sum())
        for j, cn in enumerate(CLASS_NAMES):
            row[f'prob_{cn}'] = float(all_probs[i, j])
        rows.append(row)

    df_out = (pd.DataFrame(rows)
              .sort_values('expected_return', ascending=False)
              .reset_index(drop=True))

    if save_csv:
        df_out.to_csv(save_csv, index=False, encoding='utf-8-sig')
        print(f"  Predictions saved → {save_csv}")

    return df_out


# ── 3. Portfolio construction ─────────────────────────────────────────────────

def build_portfolio(
    preds: pd.DataFrame,
    n_per_bucket: int = N_STOCKS_PER_BUCKET,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply the bucket portfolio strategy to the predictions DataFrame.

    For each positive-gain bucket:
      - Collect all stocks predicted to be in that bucket.
      - Randomly sample min(n_per_bucket, available) stocks.
      - Assign raw weight = model confidence of each selected stock.

    Final portfolio weight = raw_weight / sum(all raw weights).
    """
    rng = random.Random(seed)

    selected_rows = []
    bucket_summary = {}

    for bucket_idx in POSITIVE_BUCKET_IDX:
        bucket_label = CLASS_NAMES[bucket_idx]
        candidates   = preds[preds['predicted_class'] == bucket_idx].copy()

        if candidates.empty:
            bucket_summary[bucket_label] = {'candidates': 0, 'selected': 0}
            continue

        n_select = min(n_per_bucket, len(candidates))
        chosen   = rng.sample(range(len(candidates)), n_select)
        chosen_df = candidates.iloc[sorted(chosen)].copy()
        chosen_df['bucket_idx'] = bucket_idx

        selected_rows.append(chosen_df)
        bucket_summary[bucket_label] = {
            'candidates': len(candidates),
            'selected':   n_select,
        }

    if not selected_rows:
        print("  [WARNING] No stocks selected — all positive buckets are empty.")
        return pd.DataFrame()

    portfolio = pd.concat(selected_rows, ignore_index=True)

    # Normalize weights to sum to 1
    total_conf = portfolio['confidence'].sum()
    portfolio['weight'] = portfolio['confidence'] / total_conf

    # ── Print selection summary ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Portfolio Construction  (seed={seed}, n_per_bucket={n_per_bucket})")
    print(f"{'─'*60}")
    print(f"{'Bucket':<18} {'Candidates':>10} {'Selected':>9} {'Weight%':>8}")
    print(f"{'─'*60}")
    for bucket_idx in POSITIVE_BUCKET_IDX:
        label = CLASS_NAMES[bucket_idx]
        info  = bucket_summary.get(label, {})
        cands = info.get('candidates', 0)
        sel   = info.get('selected', 0)
        subset = portfolio[portfolio['bucket_idx'] == bucket_idx]
        w_pct  = subset['weight'].sum() * 100 if len(subset) > 0 else 0.0
        print(f"  {label:<16} {cands:>10} {sel:>9} {w_pct:>7.1f}%")

    print(f"{'─'*60}")
    print(f"  Total stocks in portfolio: {len(portfolio)}")
    print(f"  Sum of weights: {portfolio['weight'].sum():.6f}")

    return portfolio


# ── 4. Actual returns over the test window ────────────────────────────────────

def compute_returns(
    portfolio: pd.DataFrame,
    test_start: str = TEST_START,
    test_end:   str = TEST_END,
) -> pd.DataFrame:
    """
    Load actual OHLCV data for the test window and compute realized returns.

    Returns
    -------
    portfolio enriched with columns:
      exit_close          : close price on test_end
      open_entry          : open price on first trading day of window
      max_high            : max(high) over window
      actual_return_pct   : (exit_close − entry_close) / entry_close × 100
      open_return_pct     : (exit_close − open_entry)  / open_entry  × 100
      max_gain_pct        : (max_high   − entry_close) / entry_close × 100
      hit_bucket          : True if actual max_gain is within the predicted bucket
    """
    start_dt = pd.Timestamp(test_start)
    end_dt   = pd.Timestamp(test_end)

    records = []
    missing = []

    for _, row in portfolio.iterrows():
        code    = row['stock_code']
        market  = row['market'].lower()
        entry   = row['entry_close']
        lo, hi, _ = CHANGE_BUCKETS[int(row['predicted_class'])]

        fp = DATA_DIR / market / f"{code}.csv"
        if not fp.exists():
            missing.append(code)
            continue

        try:
            df = pd.read_csv(fp)
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('trade_date')
            window = df[(df['trade_date'] >= start_dt) & (df['trade_date'] <= end_dt)]

            if window.empty:
                missing.append(code)
                continue

            exit_close  = float(window['close'].iloc[-1])
            open_entry  = float(window['open'].iloc[0])
            max_high    = float(window['high'].max())

            actual_ret  = (exit_close - entry) / entry * 100
            open_ret    = (exit_close - open_entry) / open_entry * 100
            max_gain    = (max_high   - entry) / entry * 100

            # True if the actual max_gain falls inside the predicted bucket
            bucket_lo = lo if lo != -float('inf') else -9999
            bucket_hi = hi if hi !=  float('inf') else  9999
            hit = (bucket_lo <= max_gain < bucket_hi)

            rec = dict(row)
            rec.update({
                'exit_close':         exit_close,
                'open_entry':         open_entry,
                'max_high':           max_high,
                'actual_return_pct':  actual_ret,
                'open_return_pct':    open_ret,
                'max_gain_pct':       max_gain,
                'hit_bucket':         hit,
                'test_days':          len(window),
            })
            records.append(rec)

        except Exception as e:
            missing.append(f"{code}({e})")

    if missing:
        print(f"\n  [WARNING] {len(missing)} stocks had no test-window data "
              f"(shown first 10): {missing[:10]}")

    return pd.DataFrame(records)


# ── 5. Benchmark (equal-weight universe) ──────────────────────────────────────

def compute_benchmark(test_start: str = TEST_START, test_end: str = TEST_END,
                       max_stocks: Optional[int] = None) -> dict:
    """
    Equal-weight all stocks that have data for both the predict date and the
    test window.  Returns a dict with avg_return and median_return.
    """
    start_dt  = pd.Timestamp(test_start)
    end_dt    = pd.Timestamp(test_end)
    cutoff_dt = pd.Timestamp(PREDICT_DATE)

    returns = []
    all_files = []
    for market in ['sh', 'sz']:
        d = DATA_DIR / market
        if d.exists():
            all_files.extend([(market, f) for f in d.glob('*.csv')])

    if max_stocks:
        all_files = all_files[:max_stocks]

    for market, fp in all_files:
        try:
            df = pd.read_csv(fp, usecols=['trade_date', 'close', 'high'])
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('trade_date')

            entry_row = df[df['trade_date'] <= cutoff_dt]
            if entry_row.empty:
                continue
            entry = float(entry_row['close'].iloc[-1])

            window = df[(df['trade_date'] >= start_dt) & (df['trade_date'] <= end_dt)]
            if window.empty:
                continue

            exit_close = float(window['close'].iloc[-1])
            returns.append((exit_close - entry) / entry * 100)
        except Exception:
            continue

    if not returns:
        return {'avg_return': float('nan'), 'median_return': float('nan'), 'n': 0}

    return {
        'avg_return':    float(np.mean(returns)),
        'median_return': float(np.median(returns)),
        'n':             len(returns),
    }


# ── 6. Reporting ──────────────────────────────────────────────────────────────

def print_report(result: pd.DataFrame, bench: dict):
    """Print a formatted backtest performance report."""
    if result.empty:
        print("  [ERROR] No results to report.")
        return

    # Portfolio-level metrics (weighted)
    w = result['weight'].values
    actual_ret   = result['actual_return_pct'].values
    max_gain     = result['max_gain_pct'].values
    open_ret     = result['open_return_pct'].values

    port_ret     = float((w * actual_ret).sum())
    port_max     = float((w * max_gain).sum())
    port_open    = float((w * open_ret).sum())
    hit_rate     = float(result['hit_bucket'].mean() * 100)

    n_positive   = int((actual_ret > 0).sum())
    n_negative   = int((actual_ret < 0).sum())

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS  [{TEST_START} → {TEST_END}]")
    print(f"{'='*60}")
    print(f"  Predict date      : {PREDICT_DATE}")
    print(f"  Portfolio stocks  : {len(result)}")
    print(f"  Stocks with gain  : {n_positive}  |  Stocks with loss: {n_negative}")

    print(f"\n{'─'*60}")
    print(f"  PORTFOLIO RETURNS  (weighted by model confidence)")
    print(f"{'─'*60}")
    print(f"  Close-to-close return : {port_ret:+.2f}%")
    print(f"  Open-to-close return  : {port_open:+.2f}%")
    print(f"  Max achievable gain   : {port_max:+.2f}%   "
          f"(model target: max high / entry)")
    print(f"  Bucket hit rate       : {hit_rate:.1f}%   "
          f"(actual max_gain in predicted range)")

    print(f"\n{'─'*60}")
    print(f"  BENCHMARK  (equal-weight all stocks, n={bench['n']})")
    print(f"{'─'*60}")
    print(f"  Avg close-to-close    : {bench['avg_return']:+.2f}%")
    print(f"  Median close-to-close : {bench['median_return']:+.2f}%")
    print(f"  Alpha vs avg bench    : {port_ret - bench['avg_return']:+.2f}%")

    # ── Per-bucket summary ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PER-BUCKET BREAKDOWN")
    print(f"{'─'*60}")
    print(f"  {'Bucket':<18} {'N':>3} {'Wt%':>5} {'PortRet':>8} "
          f"{'MaxGain':>8} {'HitRate':>8}")

    for bidx in POSITIVE_BUCKET_IDX:
        sub = result[result['bucket_idx'] == bidx]
        if sub.empty:
            continue
        sw = sub['weight'].sum()
        sr = float((sub['weight'] * sub['actual_return_pct']).sum() / sw) if sw > 0 else 0.0
        sm = float((sub['weight'] * sub['max_gain_pct']).sum()       / sw) if sw > 0 else 0.0
        sh = float(sub['hit_bucket'].mean() * 100)
        print(f"  {CLASS_NAMES[bidx]:<18} {len(sub):>3} "
              f"{sw*100:>5.1f}% {sr:>+8.2f}% {sm:>+8.2f}% {sh:>7.1f}%")

    # ── Top / bottom performers ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  TOP 10 PERFORMERS")
    print(f"{'─'*60}")
    top_cols = ['stock_code', 'sector', 'predicted_label', 'confidence',
                'actual_return_pct', 'max_gain_pct', 'weight']
    top = result.nlargest(10, 'actual_return_pct')[top_cols]
    print(top.to_string(index=False,
                        float_format=lambda x: f"{x:+.2f}" if abs(x) < 100 else f"{x:.1f}"))

    print(f"\n{'─'*60}")
    print(f"  BOTTOM 10 PERFORMERS")
    print(f"{'─'*60}")
    bot = result.nsmallest(10, 'actual_return_pct')[top_cols]
    print(bot.to_string(index=False,
                        float_format=lambda x: f"{x:+.2f}" if abs(x) < 100 else f"{x:.1f}"))

    print(f"\n{'='*60}")


# ── 7. Main orchestrator ──────────────────────────────────────────────────────

def run_backtest(
    extend:       bool = True,
    n_per_bucket: int  = N_STOCKS_PER_BUCKET,
    seed:         int  = 42,
    device:       str  = 'cpu',
    save_csv:     bool = False,
):
    """
    Full backtest pipeline:
      1. (Optionally) extend stock data to TEST_END.
      2. Generate predictions as of PREDICT_DATE.
      3. Build portfolio using the bucket strategy.
      4. Compute actual returns over the test window.
      5. Print report + compare to benchmark.
    """
    # ── Step 1: Data extension ─────────────────────────────────────────────────
    if extend:
        extend_data_to(EXTEND_TARGET)
    else:
        print(f"\n[Skipping data extension]  Using existing data.")

    # ── Step 2: Predictions ────────────────────────────────────────────────────
    pred_csv = (DATA_DIR / 'backtest_predictions.csv') if save_csv else None
    preds = predict_as_of(
        as_of_date = PREDICT_DATE,
        device     = device,
        batch_size = 512,
        save_csv   = pred_csv,
    )

    if preds.empty:
        print("  Prediction step returned empty DataFrame — aborting.")
        return

    # ── Step 3: Portfolio selection ────────────────────────────────────────────
    portfolio = build_portfolio(preds, n_per_bucket=n_per_bucket, seed=seed)

    if portfolio.empty:
        print("  Portfolio is empty — aborting.")
        return

    # ── Step 4: Actual returns ─────────────────────────────────────────────────
    print(f"\n[Computing actual returns  {TEST_START} → {TEST_END}]")
    result = compute_returns(portfolio, test_start=TEST_START, test_end=TEST_END)

    if result.empty:
        print("  No return data available — check that data was extended.")
        return

    if save_csv:
        out_path = DATA_DIR / 'backtest_results.csv'
        result.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"  Results saved → {out_path}")

    # ── Step 5: Benchmark + report ─────────────────────────────────────────────
    print("\n[Computing equal-weight benchmark ...]")
    bench = compute_benchmark(TEST_START, TEST_END)

    print_report(result, bench)

    return result, bench


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Backtest DL transformer bucket portfolio strategy.'
    )
    parser.add_argument('--no-extend', action='store_true',
                        help='Skip data extension step (data already up to date)')
    parser.add_argument('--n-per-bucket', type=int, default=N_STOCKS_PER_BUCKET,
                        help=f'Stocks to pick per positive bucket (default {N_STOCKS_PER_BUCKET})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for stock selection (default 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Inference device: cpu | cuda | mps (default cpu)')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save predictions and results to CSV files')

    args, _ = parser.parse_known_args()

    run_backtest(
        extend       = not args.no_extend,
        n_per_bucket = args.n_per_bucket,
        seed         = args.seed,
        device       = args.device,
        save_csv     = args.save_csv,
    )
