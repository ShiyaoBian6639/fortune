"""
Feature categorisation for the Lim 2019 TFT.

The TFT splits inputs into three time-classes:
  STATIC          — per-entity, time-invariant (or slow-changing)
  KNOWN_FUTURE    — known in advance for the forecast horizon
  PAST_OBSERVED   — only known up to the current time

Categorisation is done by name match. Anything unrecognised falls into
PAST_OBSERVED (the safe default).

Usage:
    from model_compare.feature_categories import categorize
    static, past_known, past_obs = categorize(feat_cols)
"""
from typing import List, Tuple


# ─── Static features (per-stock, time-invariant or very slow-changing) ──────
STATIC_FEATURES = {
    # Hard-coded categoricals
    'sector_id', 'province_id', 'board_id', 'market_cap_bucket', 'industry_id',
    # Index-membership PIT (changes ~1× per year per stock — treat as static)
    'in_csi300', 'in_csi500', 'in_csi1000', 'in_sse50',
    # Corporate static features
    'log_reg_capital', 'years_listed', 'avg_education_rank',
    'chairman_tenure_days', 'holdernum',
    'top10_pct', 'top10_hhi', 'top10_pct_top1', 'n_funds_in_top10',
}


# ─── Known-future features (calendar; same value at past and future steps) ──
KNOWN_FUTURE_FEATURES = {
    'dow',            # day of week
    'dom',            # day of month
    'month',
    'quarter',
    'year_week',
    'is_quarter_end', 'is_year_end',
    'is_first_dow',   'is_last_dow',
    'days_to_holiday',
    # Limit prices: previous close is always known by t, so the limit prices
    # for t+1 are also known the moment we know close(t). Reasonable to treat
    # as "known future" for our 1-day-ahead pipeline.
    'up_limit_ratio', 'down_limit_ratio',
}


def categorize(feat_cols: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Split feat_cols into (static, past_known, past_observed) lists.

    Order within each category preserves the input order so feature indices
    are deterministic.
    """
    static, past_known, past_obs = [], [], []
    for f in feat_cols:
        if f in STATIC_FEATURES:
            static.append(f)
        elif f in KNOWN_FUTURE_FEATURES:
            past_known.append(f)
        else:
            past_obs.append(f)
    return static, past_known, past_obs


def report(feat_cols: List[str]) -> None:
    """Print a categorisation summary — useful when debugging panel shape."""
    s, k, o = categorize(feat_cols)
    print(f"[feat-cat] STATIC          : {len(s):>3}  {s[:8]}{'…' if len(s)>8 else ''}")
    print(f"[feat-cat] KNOWN_FUTURE    : {len(k):>3}  {k}")
    print(f"[feat-cat] PAST_OBSERVED   : {len(o):>3}  (first 5: {o[:5]})")
    return s, k, o
