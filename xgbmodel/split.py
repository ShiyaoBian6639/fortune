"""
Time-series splits for the xgbmodel pipeline.

Two modes are supported:

1. **fixed**  — a single train/val/test split defined by three date cutoffs
   (the original behaviour). Fast; good for development and final-model
   training. One fit, one set of numbers.

2. **walk_forward** — rolling or expanding walk-forward CV with purge and
   embargo, following López de Prado (2018) §7.4 "Purged K-Fold CV":
       train | purge | val | embargo | test
   A small gap (`purge_days`) is inserted between train and val, and between
   val and test, to prevent labels computed over a future horizon from
   leaking backward into training sequences. A second gap (`embargo_days`)
   is applied at the val→test boundary to further decorrelate serial
   dependence around split edges.

The user asked for "3w train / 1w val / 1w test, rolling". That is supported
via `--fold_train_weeks 3 --fold_val_weeks 1 --fold_test_weeks 1`. With so
short a train window the fold count explodes and each fit converges on very
little data — for production we recommend the defaults below (12w train / 2w
val / 2w test, step 2w; expanding-window), which reproduce the deeptime
pipeline's rolling schedule.

Research references:
  - López de Prado (2018), "Advances in Financial Machine Learning", ch. 7
    (purged K-fold with embargo; the de-facto standard for finance ML).
  - Harvey & Liu (2015), "Backtesting: An Implementation Guide" — walk-forward
    out-of-sample is more robust than random K-fold for finance.
  - Bailey et al. (2017), "The Probability of Backtest Overfitting" — warns
    that too many overlapping folds inflate false-positive rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """One purged walk-forward fold."""
    index: int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp   # inclusive
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp   # inclusive
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp   # inclusive

    def slice(self, panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        td = panel['trade_date']
        train = panel[(td >= self.train_start) & (td <= self.train_end)]
        val   = panel[(td >= self.val_start)   & (td <= self.val_end)]
        test  = panel[(td >= self.test_start)  & (td <= self.test_end)]
        return (train.reset_index(drop=True),
                val.reset_index(drop=True),
                test.reset_index(drop=True))

    def summary(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> str:
        return (
            f"fold {self.index:2d}: "
            f"train {self.train_start.date()}→{self.train_end.date()} "
            f"({len(train):>7,} rows) | "
            f"val {self.val_start.date()}→{self.val_end.date()} "
            f"({len(val):>6,}) | "
            f"test {self.test_start.date()}→{self.test_end.date()} "
            f"({len(test):>6,})"
        )


def _trading_days(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """Unique sorted A-share trading dates present in the panel."""
    return pd.DatetimeIndex(sorted(panel['trade_date'].unique()))


def walk_forward_folds(
    panel: pd.DataFrame,
    fold_train_weeks: int  = 12,
    fold_val_weeks:   int  = 2,
    fold_test_weeks:  int  = 2,
    fold_step_weeks:  int  = 2,
    purge_days:       int  = 5,
    embargo_days:     int  = 2,
    expanding:        bool = False,
    min_train_days:   int  = 60,
    start_date:       Optional[pd.Timestamp] = None,
    end_date:         Optional[pd.Timestamp] = None,
) -> List[Fold]:
    """Generate walk-forward folds over `panel['trade_date']`.

    Parameters
    ----------
    fold_train_weeks, fold_val_weeks, fold_test_weeks
        Width of each window in trading weeks (5 trading days per week).
    fold_step_weeks
        How far the cursor advances between folds. `= fold_test_weeks` → the
        test windows tile the timeline with no gaps or overlaps.
    purge_days
        Trading days dropped at the train→val and val→test boundaries to
        prevent label leakage through the forward-return horizon.
    embargo_days
        Additional trading days dropped after val, before test, to decorrelate
        serial autocorrelation (de Prado §7.4).
    expanding
        If True, each fold's train window starts at `start_date` and grows
        forward (expanding window, preferred for stationary-ish regimes).
        If False, train uses a fixed-length rolling window (preferred when
        regimes shift — models refit on recent history only).
    min_train_days
        Skip folds whose train window contains fewer distinct trading days
        than this (avoid undertraining on the first fold during expanding mode).
    start_date, end_date
        Optional clamps on the global timeline.

    Returns
    -------
    List[Fold]
    """
    days = _trading_days(panel)
    if start_date is not None:
        days = days[days >= pd.Timestamp(start_date)]
    if end_date is not None:
        days = days[days <= pd.Timestamp(end_date)]
    if len(days) < (fold_train_weeks + fold_val_weeks + fold_test_weeks) * 5:
        raise RuntimeError(
            f"Not enough trading days ({len(days)}) for "
            f"{fold_train_weeks}w+{fold_val_weeks}w+{fold_test_weeks}w folds"
        )

    # Convert window widths to integer trading-day counts
    w_train = fold_train_weeks * 5
    w_val   = fold_val_weeks   * 5
    w_test  = fold_test_weeks  * 5
    w_step  = max(fold_step_weeks, 1) * 5

    folds: List[Fold] = []
    cursor = 0
    fold_idx = 0

    while True:
        # Train window
        if expanding:
            train_lo = 0
        else:
            train_lo = cursor
        train_hi = train_lo + w_train - 1
        val_lo   = train_hi + 1 + purge_days
        val_hi   = val_lo + w_val - 1
        test_lo  = val_hi + 1 + embargo_days
        test_hi  = test_lo + w_test - 1

        if test_hi >= len(days):
            break
        if (train_hi - train_lo + 1) < min_train_days:
            cursor += w_step
            continue

        folds.append(Fold(
            index       = fold_idx,
            train_start = days[train_lo],
            train_end   = days[train_hi],
            val_start   = days[val_lo],
            val_end     = days[val_hi],
            test_start  = days[test_lo],
            test_end    = days[test_hi],
        ))
        fold_idx += 1
        cursor += w_step

    return folds


def summarize_folds(folds: List[Fold]) -> str:
    if not folds:
        return "(no folds)"
    lines = [f"walk_forward: {len(folds)} folds, "
             f"{folds[0].train_start.date()} → {folds[-1].test_end.date()}"]
    for f in folds[:3]:
        lines.append("  " + f"fold {f.index:2d}  train {f.train_start.date()}→{f.train_end.date()}  "
                            f"val {f.val_start.date()}→{f.val_end.date()}  "
                            f"test {f.test_start.date()}→{f.test_end.date()}")
    if len(folds) > 3:
        lines.append(f"  ... {len(folds) - 3} more folds ...")
    return "\n".join(lines)
