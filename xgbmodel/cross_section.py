"""
Cross-sectional feature engineering for the xgbmodel panel.

These features rank each stock *within its trading day* against all other
stocks. They are the critical complement to per-stock technical features
because macro / calendar features are constant across stocks on any given
day and therefore contribute nothing to cross-sectional IC.

Added columns (all float32, in [0, 1]):
  cs_rank_pct_chg              — today's raw return rank
  cs_rank_turnover_rate_f      — today's free-float turnover rank
  cs_rank_vol_ratio_20         — volume-vs-20d-avg rank
  cs_rank_amt_ratio_20         — amount-vs-20d-avg rank
  cs_rank_rsi_14               — RSI(14) rank
  cs_rank_momentum_20          — 20-day momentum rank
  cs_rank_vol_pct_20           — 20-day volatility rank
  cs_rank_dist_from_high_20    — distance from 20d high rank
  cs_rank_net_mf_amount_ratio  — net money flow rank
  cs_rank_up_limit_ratio       — how close to limit-up rank (inverse = hotness)

Plus de-meaned versions (per-day mean subtracted) for key features —
useful for XGBoost to learn non-linear interactions with the daily cross-section.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


CS_RANK_FEATURES = [
    'pct_chg',
    'turnover_rate_f',
    'vol_ratio_20',
    'amt_ratio_20',
    'rsi_14',
    'momentum_20',
    'vol_pct_20',
    'dist_from_high_20',
    'net_mf_amount_ratio',
    'up_limit_ratio',
    'overnight_gap',
    'log_ret',
]

CS_DEMEAN_FEATURES = [
    'pct_chg',
    'turnover_rate_f',
    'rsi_14',
    'momentum_20',
]


def add_cross_section_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional rank and de-meaned features to a (stock, date) panel.

    Called *once* at the end of build_panel so ranks are computed over the
    full cross-section of stocks available on each trading day.

    Performance: groupby('trade_date') on a 6M-row panel takes ~15s total
    for ~12 columns thanks to the C rank() path.
    """
    td = panel['trade_date']
    new_cols = {}

    for col in CS_RANK_FEATURES:
        if col not in panel.columns:
            continue
        # rank(pct=True) → uniform in [0, 1] by day; NaNs preserved, fill to 0.5 (median)
        ranked = panel.groupby(td)[col].rank(pct=True, method='average').astype('float32')
        new_cols[f'cs_rank_{col}'] = ranked.fillna(0.5)

    for col in CS_DEMEAN_FEATURES:
        if col not in panel.columns:
            continue
        daily_mean = panel.groupby(td)[col].transform('mean').astype('float32')
        new_cols[f'cs_demean_{col}'] = (panel[col] - daily_mean).astype('float32')

    # Daily breadth: fraction of stocks with positive return today. Same value
    # for every row on a given day but combined with a stock's own return it
    # tells the model "is this stock outperforming in a broad or narrow rally?".
    pos = (panel['pct_chg'] > 0).astype('float32')
    breadth = pos.groupby(td).transform('mean').astype('float32')
    new_cols['cs_market_breadth'] = breadth

    # Daily dispersion: std of pct_chg across stocks — high dispersion days
    # are where cross-sectional models tend to have more signal.
    dispersion = panel.groupby(td)['pct_chg'].transform('std').astype('float32')
    new_cols['cs_daily_dispersion'] = dispersion.fillna(0.0)

    return pd.concat([panel, pd.DataFrame(new_cols, index=panel.index)], axis=1, copy=False)


def cross_section_column_names() -> List[str]:
    """The authoritative list of cross-sectional feature names this module adds."""
    cols = [f'cs_rank_{c}' for c in CS_RANK_FEATURES]
    cols += [f'cs_demean_{c}' for c in CS_DEMEAN_FEATURES]
    cols += ['cs_market_breadth', 'cs_daily_dispersion']
    return cols
