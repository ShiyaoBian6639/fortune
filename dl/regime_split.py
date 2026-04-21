"""
Regime-aware temporal train/val/test split for financial time-series.

Uses CSI300 close vs MA-250 to define bull/bear market regimes, then assigns
entire regime blocks to splits — preserving temporal order and ensuring val/test
samples span distinct market periods.

Design goals:
  1. No look-ahead: val/test always lie in the future relative to (most of) train.
  2. Regime diversity: val covers the major bear period; test covers the recent bull.
  3. Purge gap: purge_gap_days trading days between adjacent blocks of different
     splits to prevent feature-window and label-window contamination.

Purge gap math:
  TRAIN last sequence ends at day T_last = TRAIN_boundary - purge_gap_days.
  TRAIN last sequence features: [T_last - seq_len + 1, T_last]
  VAL   first sequence features: [VAL_start - seq_len + 1, VAL_start]
     where VAL_start = TRAIN_boundary + 1.
  Zero feature overlap requires: T_last < VAL_start - seq_len + 1
     → TRAIN_boundary - purge_gap_days < TRAIN_boundary + 1 - seq_len + 1
     → purge_gap_days > seq_len - 2  (satisfied by gap=30 for seq_len=30)
  The extra +5 covers label contamination (max_forward_window = 5).
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd


# ── Signal loading ────────────────────────────────────────────────────────────

def load_csi300_signal(data_dir: str) -> pd.DataFrame:
    """
    Load CSI300 close price and MA-250 from idx_factor_pro.

    Returns:
        DataFrame with columns ['trade_date', 'close', 'ma_bfq_250']
        sorted by trade_date ascending. trade_date is datetime.
    """
    path = os.path.join(data_dir, 'index', 'idx_factor_pro', '000300_SH.csv')
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df = (df[['trade_date', 'close', 'ma_bfq_250']]
          .dropna()
          .sort_values('trade_date')
          .reset_index(drop=True))
    return df


# ── Regime detection ──────────────────────────────────────────────────────────

def detect_regime_blocks(
    signal_df: pd.DataFrame,
    min_block_days: int = 40,
) -> List[Dict]:
    """
    Detect bull/bear regime blocks from CSI300 close vs MA-250.

    Steps:
      1. Label each day: 'bull' (close > ma_bfq_250) or 'bear'.
      2. Group consecutive same-label days into raw blocks.
      3. Iteratively merge the shortest block below min_block_days into its
         smaller-sized neighbor until all blocks >= min_block_days.

    Args:
        signal_df: DataFrame from load_csi300_signal().
        min_block_days: Minimum block size after merging.

    Returns:
        List of dicts sorted by start date:
        [{'regime': 'bull'|'bear', 'start': datetime, 'end': datetime,
          'n_days': int}, ...]
    """
    df = signal_df.sort_values('trade_date').reset_index(drop=True)
    df['regime'] = np.where(df['close'] > df['ma_bfq_250'], 'bull', 'bear')

    # Build raw consecutive blocks
    blocks: List[Dict] = []
    for date, regime in zip(df['trade_date'], df['regime']):
        if not blocks or blocks[-1]['regime'] != regime:
            blocks.append({'regime': regime, 'start': date, 'end': date, 'n_days': 1})
        else:
            blocks[-1]['end'] = date
            blocks[-1]['n_days'] += 1

    # Iteratively merge the shortest sub-threshold block into its smaller neighbor
    while True:
        short_idxs = [i for i, b in enumerate(blocks) if b['n_days'] < min_block_days]
        if not short_idxs:
            break

        # Pick the shortest block; ties broken by index (first wins)
        i = min(short_idxs, key=lambda x: blocks[x]['n_days'])

        has_prev = i > 0
        has_next = i < len(blocks) - 1

        if not has_prev and not has_next:
            break  # Single block — nothing to merge into

        if has_prev and has_next:
            merge_with = (i - 1) if blocks[i - 1]['n_days'] <= blocks[i + 1]['n_days'] else (i + 1)
        elif has_prev:
            merge_with = i - 1
        else:
            merge_with = i + 1

        left  = min(i, merge_with)
        right = max(i, merge_with)

        merged = {
            'regime': blocks[merge_with]['regime'],   # neighbor regime wins
            'start':  blocks[left]['start'],
            'end':    blocks[right]['end'],
            'n_days': blocks[left]['n_days'] + blocks[right]['n_days'],
        }
        blocks[left:right + 1] = [merged]

        # Coalesce any now-adjacent same-regime blocks
        new_blocks: List[Dict] = []
        for b in blocks:
            if new_blocks and new_blocks[-1]['regime'] == b['regime']:
                new_blocks[-1]['end']    = b['end']
                new_blocks[-1]['n_days'] += b['n_days']
            else:
                new_blocks.append(b.copy())
        blocks = new_blocks

    return blocks


# ── Split assignment ───────────────────────────────────────────────────────────

def assign_blocks_to_splits(
    blocks: List[Dict],
    val_days: int = 200,
    test_days: int = 190,
) -> List[Dict]:
    """
    Assign each block to 'train', 'val', or 'test'.

    Strategy:
      - VAL:  the last `val_days` trading days of the longest bear block.
              The remaining (earlier) days of that block become TRAIN.
              This caps val size so training data stays dominant (~60-70%).
      - TEST: the last `test_days` of the final chronological block.
      - TRAIN: everything else.

    Blocks that are partially used for VAL or TEST get 'train_then_val' or
    'train_then_test' split labels; build_date_split_map handles the date-
    level boundary within those blocks.

    Returns:
        Blocks list with 'split' key added.
        Partial-split blocks include 'val_days' or 'test_days' keys.
    """
    # VAL: last val_days of the longest bear block
    bear_blocks = [(i, b) for i, b in enumerate(blocks) if b['regime'] == 'bear']
    if bear_blocks:
        val_idx = max(bear_blocks, key=lambda x: x[1]['n_days'])[0]
    else:
        # No bear period found — use penultimate block as val fallback
        val_idx = max(0, len(blocks) - 2)

    result = []
    for i, block in enumerate(blocks):
        b = block.copy()
        if i == val_idx:
            if b['n_days'] > val_days:
                # Partial: earlier part → TRAIN, last val_days → VAL
                b['split'] = 'train_then_val'
                b['val_days'] = val_days
            else:
                b['split'] = 'val'
        elif i == len(blocks) - 1:
            if b['n_days'] > test_days:
                # Partial: earlier part → TRAIN, last test_days → TEST
                b['split'] = 'train_then_test'
                b['test_days'] = test_days
            else:
                b['split'] = 'test'
        else:
            b['split'] = 'train'
        result.append(b)

    return result


# ── Date → split map ──────────────────────────────────────────────────────────

def build_date_split_map(
    blocks: List[Dict],
    signal_df: pd.DataFrame,
    purge_gap_days: int = 35,
    date_min: int = None,
    date_max: int = None,
) -> Dict[int, str]:
    """
    Build a mapping from trading date (int YYYYMMDD) to split assignment.

    Purge gap strategy: at each transition from split A → split B, mark the
    last `purge_gap_days` dates of block A as 'gap'.  This guarantees:
      · No feature-window overlap between the last TRAIN sequence and the
        first VAL/TEST sequence  (gap >= seq_len).
      · No label-window overlap (gap >= seq_len + max_forward_window).

    Args:
        blocks:          Output of assign_blocks_to_splits().
        signal_df:       Full CSI300 signal from load_csi300_signal().
        purge_gap_days:  Trading days to mark as gap at each block boundary.
        date_min:        If given (int YYYYMMDD), filter signal to >= this date.
        date_max:        If given (int YYYYMMDD), filter signal to <= this date.

    Returns:
        dict: int date (YYYYMMDD) → 'train' | 'val' | 'test' | 'gap'
    """
    # Filter signal to the sample date range if specified
    sig = signal_df[['trade_date']].copy()
    sig['date_int'] = (sig['trade_date'].dt.year  * 10000
                     + sig['trade_date'].dt.month * 100
                     + sig['trade_date'].dt.day).astype(int)
    if date_min is not None:
        sig = sig[sig['date_int'] >= date_min]
    if date_max is not None:
        sig = sig[sig['date_int'] <= date_max]
    sig = sig.sort_values('trade_date').reset_index(drop=True)

    # Initial assignment from blocks
    date_to_split: Dict[int, str] = {}
    for block in blocks:
        mask = ((signal_df['trade_date'] >= block['start']) &
                (signal_df['trade_date'] <= block['end']))
        dates_in_block = (signal_df.loc[mask, 'trade_date']
                          .sort_values().reset_index(drop=True))
        split = block['split']

        if split == 'train_then_test':
            tail = block.get('test_days', 190)
            n = len(dates_in_block)
            split_idx = max(0, n - tail)
            for idx, d in enumerate(dates_in_block):
                dint = int(d.strftime('%Y%m%d'))
                date_to_split[dint] = 'train' if idx < split_idx else 'test'
        elif split == 'train_then_val':
            tail = block.get('val_days', 200)
            n = len(dates_in_block)
            split_idx = max(0, n - tail)
            for idx, d in enumerate(dates_in_block):
                dint = int(d.strftime('%Y%m%d'))
                date_to_split[dint] = 'train' if idx < split_idx else 'val'
        else:
            for d in dates_in_block:
                date_to_split[int(d.strftime('%Y%m%d'))] = split

    # Apply purge gaps: for dates in our sample range only
    # Build sorted list restricted to sig dates
    sorted_sig_dates = sig['date_int'].tolist()
    splits_arr = [date_to_split.get(d, 'train') for d in sorted_sig_dates]
    n = len(sorted_sig_dates)

    # Walk forward; at each A→B transition, mark the last purge_gap_days
    # positions of block A as 'gap'.
    final_splits = list(splits_arr)
    i = 1
    while i < n:
        prev = splits_arr[i - 1]
        curr = splits_arr[i]
        if prev != curr and prev not in ('gap',) and curr not in ('gap',):
            # Boundary: splits[i-1] → splits[i]
            gap_start = max(0, i - purge_gap_days)
            for j in range(gap_start, i):
                final_splits[j] = 'gap'
        i += 1

    result: Dict[int, str] = {}
    for date_int, split in zip(sorted_sig_dates, final_splits):
        result[date_int] = split

    return result


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_split_stats(date_split_map: Dict[int, str]) -> None:
    """Print regime-split statistics: days per split and gap data loss."""
    from collections import Counter
    counts = Counter(date_split_map.values())
    total  = len(date_split_map)

    print("\nRegime-aware split (trading-day distribution):")
    for sp in ('train', 'val', 'test', 'gap'):
        n   = counts.get(sp, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f"  {sp:6s}: {n:5d} days ({pct:.1f}%)")
    print(f"  {'total':6s}: {total:5d} days")

    gap_n = counts.get('gap', 0)
    if gap_n:
        non_gap = total - gap_n
        print(f"\n  Purge gaps remove {gap_n} trading days "
              f"({100.*gap_n/total:.1f}% of calendar; "
              f"{100.*gap_n/non_gap:.1f}% relative to usable data)")
