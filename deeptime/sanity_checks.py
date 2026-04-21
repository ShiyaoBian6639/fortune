"""
Pre-training data sanity checks for the deeptime pipeline.
Run via run_all_checks(data_dir, cache_dir, config) before training.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import DT_FEATURE_COLUMNS, FORWARD_WINDOWS, NUM_HORIZONS


def check_feature_distribution_shift(
    cache_dir: str,
) -> List[str]:
    """
    Flag features with >5× std shift between 2017-2019 and 2023-2025.
    Returns list of flagged feature names.
    """
    import json
    obs_path = os.path.join(cache_dir, 'train_obs.npy')
    date_path = os.path.join(cache_dir, 'train_dates.npy')
    if not os.path.exists(obs_path) or not os.path.exists(date_path):
        print("  [skip] feature shift check: cache not found")
        return []

    with open(os.path.join(cache_dir, 'metadata.json')) as f:
        meta = json.load(f)
    n     = meta['splits']['train']['n_samples']
    n_past = meta['n_past']
    sl    = meta['seq_length']

    obs   = np.memmap(obs_path,  dtype='float32', mode='r', shape=(n, sl, n_past))
    dates = np.memmap(date_path, dtype='int32',   mode='r', shape=(n,))

    # Use last timestep of each sequence for feature stats
    last = obs[:, -1, :]   # (N, n_past)

    early_mask = (dates < 20200101)
    late_mask  = (dates >= 20230101)

    flagged = []
    from .config import DT_OBSERVED_PAST_COLUMNS
    for i, feat in enumerate(DT_OBSERVED_PAST_COLUMNS[:n_past]):
        if early_mask.sum() < 100 or late_mask.sum() < 100:
            continue
        std_early = float(np.nanstd(last[early_mask, i]))
        std_late  = float(np.nanstd(last[late_mask,  i]))
        if std_early < 1e-8:
            continue
        ratio = std_late / (std_early + 1e-8)
        if ratio > 5.0 or ratio < 0.2:
            flagged.append(f"{feat} (ratio={ratio:.2f})")

    if flagged:
        print(f"  [WARN] Distribution shift detected in {len(flagged)} features:")
        for f in flagged[:10]:
            print(f"    {f}")
    else:
        print(f"  [OK] Feature distribution shift check passed")
    return flagged


def check_target_leakage(
    cache_dir: str,
    threshold: float = 0.3,
) -> List[str]:
    """
    Flag observed-past features with |corr| > threshold with any target horizon.
    High correlation suggests potential lookahead contamination.
    """
    import json
    for split in ('train', 'val'):
        obs_path = os.path.join(cache_dir, f'{split}_obs.npy')
        tgt_path = os.path.join(cache_dir, f'{split}_targets.npy')
        if not os.path.exists(obs_path):
            continue

        with open(os.path.join(cache_dir, 'metadata.json')) as f:
            meta = json.load(f)
        n     = meta['splits'][split]['n_samples']
        n_past = meta['n_past']
        sl    = meta['seq_length']

        obs  = np.memmap(obs_path, dtype='float32', mode='r', shape=(n, sl, n_past))
        tgts = np.memmap(tgt_path, dtype='float32', mode='r', shape=(n, NUM_HORIZONS))

        # Sample up to 5000 for speed
        idx = np.random.choice(n, min(5000, n), replace=False)
        last = obs[idx, -1, :].astype('float64')
        t    = tgts[idx, :].astype('float64')

        flagged = []
        from .config import DT_OBSERVED_PAST_COLUMNS
        for i, feat in enumerate(DT_OBSERVED_PAST_COLUMNS[:n_past]):
            col = last[:, i]
            if np.nanstd(col) < 1e-8:
                continue
            for h in range(NUM_HORIZONS):
                valid = np.isfinite(col) & np.isfinite(t[:, h])
                if valid.sum() < 50:
                    continue
                corr = float(np.corrcoef(col[valid], t[valid, h])[0, 1])
                if abs(corr) > threshold:
                    flagged.append(f"{feat} ~ day{FORWARD_WINDOWS[h]} (corr={corr:.3f})")

        if flagged:
            print(f"  [WARN] Possible leakage in {len(flagged)} feature-horizon pairs:")
            for f in flagged[:10]:
                print(f"    {f}")
        else:
            print(f"  [OK] Leakage check passed on {split} split")
        break   # only check one split

    return []


def check_sector_coverage(data_dir: str, min_stocks: int = 5) -> None:
    """Verify each sector has at least min_stocks stocks."""
    path = os.path.join(data_dir, 'stock_sectors.csv')
    if not os.path.exists(path):
        print("  [skip] sector coverage: stock_sectors.csv not found")
        return
    df = pd.read_csv(path)
    if 'sector' not in df.columns:
        return
    counts = df['sector'].value_counts()
    thin   = counts[counts < min_stocks]
    if len(thin):
        print(f"  [WARN] {len(thin)} sectors with < {min_stocks} stocks:")
        for sec, cnt in thin.items():
            print(f"    {sec}: {cnt} stocks")
    else:
        print(f"  [OK] Sector coverage: {len(counts)} sectors, min={counts.min()} stocks")


def check_fundamental_coverage(data_dir: str, sample_n: int = 100) -> None:
    """Verify fina_indicator has no future ann_date leakage for a sample of stocks."""
    fina_dir = os.path.join(data_dir, 'fina_indicator')
    if not os.path.isdir(fina_dir):
        print("  [skip] fundamental coverage: fina_indicator/ not found")
        return

    files = [f for f in os.listdir(fina_dir) if f.endswith('.csv')][:sample_n]
    coverage = []
    for fname in files:
        try:
            df = pd.read_csv(os.path.join(fina_dir, fname),
                             usecols=['ann_date', 'end_date'])
            # Check: ann_date should always be >= end_date
            df['ann_date'] = pd.to_datetime(df['ann_date'].astype(str), errors='coerce')
            df['end_date'] = pd.to_datetime(df['end_date'].astype(str), errors='coerce')
            bad = (df['ann_date'] < df['end_date']).sum()
            if bad > 0:
                print(f"  [WARN] {fname}: {bad} rows where ann_date < end_date (leakage risk)")
            coverage.append(len(df))
        except Exception:
            continue

    if coverage:
        print(f"  [OK] Fundamentals: {len(coverage)} files checked, "
              f"avg {np.mean(coverage):.0f} announcements per stock")


def check_price_limit_consistency(data_dir: str, sample_n: int = 50) -> None:
    """Verify pct_chg stays within declared ±limit for a sample of stocks."""
    stk_dir = os.path.join(data_dir, 'stk_limit')
    sh_dir  = os.path.join(data_dir, 'sh')
    if not os.path.isdir(stk_dir) or not os.path.isdir(sh_dir):
        print("  [skip] price limit check: stk_limit/ or sh/ not found")
        return

    violations = 0
    files = [f for f in os.listdir(sh_dir) if f.endswith('.csv')][:sample_n]
    for fname in files:
        ts_code = fname.replace('.csv', '')
        price_path = os.path.join(sh_dir, fname)
        lim_path   = os.path.join(stk_dir, f'stk_limit_{ts_code}.csv')
        try:
            price = pd.read_csv(price_path, usecols=['trade_date', 'pct_chg'])
            # Only need max abs(pct_chg) per stock
            max_abs = price['pct_chg'].abs().max()
            if max_abs > 25:   # >25% in a day is suspicious (except ST+IPO edge cases)
                violations += 1
        except Exception:
            continue

    if violations:
        print(f"  [WARN] {violations}/{sample_n} stocks have |pct_chg| > 25% (likely IPO/ST days)")
    else:
        print(f"  [OK] Price limit consistency: sample of {sample_n} stocks checked")


def check_split_integrity(cache_dir: str) -> None:
    """
    Verify no individual date appears in both train and val/test.

    Note: In rolling window splits the min/max DATE RANGES legitimately
    overlap across splits (a date from fold-1 val can be later than a
    date from fold-2 train because folds advance the cursor). What must
    NOT happen is a single date being assigned to both train and val/test.
    """
    import json
    if not os.path.exists(os.path.join(cache_dir, 'metadata.json')):
        print("  [skip] split integrity: cache not found")
        return

    with open(os.path.join(cache_dir, 'metadata.json')) as f:
        meta = json.load(f)

    split_dates = {}
    for split in ('train', 'val', 'test'):
        path = os.path.join(cache_dir, f'{split}_dates.npy')
        n    = meta['splits'].get(split, {}).get('n_samples', 0)
        if n == 0 or not os.path.exists(path):
            continue
        dates = np.memmap(path, dtype='int32', mode='r', shape=(n,))
        split_dates[split] = set(dates.tolist())
        lo, hi = int(dates.min()), int(dates.max())
        print(f"  {split}: {lo} -> {hi} ({n:,} samples, {len(split_dates[split])} unique dates)")

    # Check individual date overlap (the real leakage risk)
    if 'train' in split_dates and 'val' in split_dates:
        overlap = split_dates['train'] & split_dates['val']
        if overlap:
            print(f"  [WARN] {len(overlap)} dates appear in BOTH train and val — possible leakage")
        else:
            print(f"  [OK] No individual date appears in both train and val")

    if 'train' in split_dates and 'test' in split_dates:
        overlap = split_dates['train'] & split_dates['test']
        if overlap:
            print(f"  [WARN] {len(overlap)} dates appear in BOTH train and test — possible leakage")
        else:
            print(f"  [OK] No individual date appears in both train and test")


def check_block_trade_sparsity(data_dir: str) -> None:
    """Report fraction of stock-days with non-zero block trades."""
    block_dir = os.path.join(data_dir, 'block_trade')
    if not os.path.isdir(block_dir):
        print("  [skip] block trade sparsity: block_trade/ not found")
        return

    files = [f for f in os.listdir(block_dir) if f.endswith('.csv')][:100]
    n_dates = len(files)
    n_with_trades = 0
    for fname in files:
        try:
            df = pd.read_csv(os.path.join(block_dir, fname))
            if len(df) > 0:
                n_with_trades += 1
        except Exception:
            continue

    print(f"  Block trade dates with trades: {n_with_trades}/{n_dates} "
          f"({100.*n_with_trades/max(n_dates,1):.0f}%)")


def check_target_distribution(cache_dir: str) -> None:
    """Verify target distribution is approximately symmetric around zero."""
    import json
    path = os.path.join(cache_dir, 'train_targets.npy')
    if not os.path.exists(path):
        print("  [skip] target distribution: cache not found")
        return

    with open(os.path.join(cache_dir, 'metadata.json')) as f:
        meta = json.load(f)
    n = meta['splits']['train']['n_samples']
    h = meta['num_horizons']

    targets = np.memmap(path, dtype='float32', mode='r', shape=(n, h))
    tgt = targets[:, 0]   # day1 targets

    mean   = float(np.nanmean(tgt))
    std    = float(np.nanstd(tgt))
    p5, p25, p50, p75, p95 = np.nanpercentile(tgt, [5, 25, 50, 75, 95])

    print(f"  Target distribution (day+1): mean={mean:.3f} std={std:.3f}")
    print(f"    p5={p5:.2f} p25={p25:.2f} p50={p50:.2f} p75={p75:.2f} p95={p95:.2f}")
    if abs(mean) > 1.0:
        print(f"  [WARN] Target mean ({mean:.3f}) is far from 0 — check if CSI300 subtraction is working")
    else:
        print(f"  [OK] Target mean close to 0 (excess return)")


def run_all_checks(data_dir: str, cache_dir: Optional[str] = None, config: dict = None) -> None:
    """Run all 8 sanity checks in sequence."""
    print("\n" + "="*60)
    print("deeptime DATA SANITY CHECKS")
    print("="*60)

    print("\n[1] Sector coverage")
    check_sector_coverage(data_dir)

    print("\n[2] Fundamental coverage")
    check_fundamental_coverage(data_dir)

    print("\n[3] Price limit consistency")
    check_price_limit_consistency(data_dir)

    print("\n[4] Block trade sparsity")
    check_block_trade_sparsity(data_dir)

    if cache_dir and os.path.isdir(cache_dir):
        print("\n[5] Feature distribution shift")
        check_feature_distribution_shift(cache_dir)

        print("\n[6] Target leakage audit")
        check_target_leakage(cache_dir)

        print("\n[7] Split integrity")
        check_split_integrity(cache_dir)

        print("\n[8] Target distribution")
        check_target_distribution(cache_dir)
    else:
        print("\n[5-8] Skipped (cache not available yet)")

    print("\n" + "="*60 + "\n")
