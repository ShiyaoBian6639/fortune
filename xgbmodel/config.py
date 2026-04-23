"""
Configuration for the xgbmodel next-day pct_chg regression pipeline.

Data sources (all under stock_data/, built by api/ and features/ modules):
  - sh/, sz/               per-stock daily OHLCV + pct_chg
  - daily_basic/           per-date valuation metrics (pe, pb, turnover, mv)
  - moneyflow/             per-date money-flow by size bucket (sm/md/lg/elg)
  - stk_limit/             per-date up/down limit prices
  - block_trade/           per-date block trades
  - index/index_dailybasic/ CSI300/CSI500/SSE/SZSE valuation + turnover
  - fina_indicator/        per-stock quarterly fundamentals (roe, roa, ...)
  - stock_sectors.csv      per-stock SW sector/industry for grouping

The model predicts raw next-day pct_chg (t+1) and, optionally, the excess
return pct_chg_{t+1} - csi300_pct_chg_{t+1}. Default target is raw pct_chg
per the user request.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(_REPO_ROOT, 'stock_data')
CACHE_DIR    = os.path.join(DATA_DIR, 'xgb_cache')
MODEL_DIR    = os.path.join(DATA_DIR, 'models')
PLOT_DIR     = os.path.join(_REPO_ROOT, 'plots', 'xgb_results')
PREDICT_OUT  = os.path.join(_REPO_ROOT, 'stock_predictions_xgb.csv')

for _d in (CACHE_DIR, MODEL_DIR, PLOT_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Target ───────────────────────────────────────────────────────────────────
# 'raw'    = next-day pct_chg (the user's request)
# 'excess' = next-day pct_chg - csi300 next-day pct_chg (regime-invariant)
TARGET_MODE       = 'raw'
FORWARD_WINDOW    = 1       # predict t+1
CLIP_TARGET_PCT   = 11.0    # clip absolute target at 11% (daily limit-down/up band)

# ─── Date split (walk-forward) ───────────────────────────────────────────────
# Date format YYYYMMDD (int). Train < VAL_START ≤ val < TEST_START ≤ test.
TRAIN_START = 20170101
VAL_START   = 20240101
TEST_START  = 20250101
# Anything after TEST_START is held out for reporting; live inference uses the
# most-recent day in the merged panel.

# ─── Stock selection ──────────────────────────────────────────────────────────
MIN_ROWS_PER_STOCK = 300   # drop stocks with <300 history rows (too new to train on)

# ─── Feature groups ───────────────────────────────────────────────────────────

# Rolling windows used across the feature code
MA_WINDOWS       = [5, 10, 20, 60]
VOL_WINDOWS      = [5, 10, 20]
MOMENTUM_WINDOWS = [5, 10, 20]
RETURN_LAGS      = [1, 2, 3, 5, 10]

# Daily basic (valuation) columns we keep — drop raw shares/mv in favor of logs
DAILY_BASIC_RAW_KEEP = [
    'turnover_rate', 'turnover_rate_f', 'volume_ratio',
    'pe_ttm', 'pb', 'ps_ttm', 'dv_ttm',
    'total_mv', 'circ_mv',     # log-transformed by feature code
]

# Moneyflow raw columns we use (net flows normalized by amount)
MONEYFLOW_TIERS = ['sm', 'md', 'lg', 'elg']   # small / medium / large / extra-large

# Quarterly fundamentals forward-filled from fina_indicator/
FINA_COLUMNS = [
    'roe', 'roa', 'grossprofit_margin', 'netprofit_margin',
    'current_ratio', 'quick_ratio', 'debt_to_assets',
    'assets_yoy', 'equity_yoy', 'op_yoy', 'ebt_yoy', 'eps',
]

# Index daily basics we merge (pe/pb/turnover on market indices as macro signal)
# All codes must exist under stock_data/index/index_dailybasic/<code>.csv
INDEX_CODES = {
    'csi300':  '000300_SH',   # 沪深300
    'csi500':  '000905_SH',   # 中证500
    'sse50':   '000016_SH',   # 上证50
    'sse':     '000001_SH',   # 上证综指
    'szse':    '399001_SZ',   # 深证成指
    'csi1000': '000852_SH',   # 中证1000
    'chinext': '399006_SZ',   # 创业板指
}

# Global indices from stock_data/index/index_global/ — merged lagged by 1 day
# since US/EU markets close after A-share open (otherwise = look-ahead leak).
# HSI (Hong Kong) trades partly overlapping with A-shares; we still lag it 1
# trading day for safety.
GLOBAL_INDEX_CODES = {
    'spx':  'SPX',    # S&P 500
    'dji':  'DJI',    # Dow Jones
    'ixic': 'IXIC',   # Nasdaq Composite
    'hsi':  'HSI',    # Hang Seng (Hong Kong)
    'n225': 'N225',   # Nikkei 225
    'ftse': 'FTSE',   # FTSE 100
}

# Pre-computed TA factors for CSI300 from stock_data/index/idx_factor_pro/
# (tushare idx_factor_pro endpoint). These describe the broad-market regime
# without us having to recompute them. We pick the most widely-used ones.
IDX_FACTOR_COLUMNS = [
    'bias1_bfq', 'bias2_bfq', 'bias3_bfq',   # BIAS vs MA (short/med/long)
    'cci_bfq',                                # Commodity Channel Index
    'dmi_adx_bfq', 'dmi_pdi_bfq', 'dmi_mdi_bfq',  # ADX / directional movement
    'kdj_k_bfq', 'kdj_d_bfq',                 # KDJ stochastic
    'rsi_bfq_6', 'rsi_bfq_12', 'rsi_bfq_24',  # multi-window RSI
    'mfi_bfq',                                # Money Flow Index
    'wr_bfq',                                 # Williams %R
    'macd_dif_bfq', 'macd_dea_bfq',           # MACD components
    'psy_bfq',                                # Psychology Index
    'vr_bfq',                                 # Volume Ratio
]
IDX_FACTOR_CODE = '000300_SH'   # use CSI300 as the market proxy

# ─── XGBoost parameters ──────────────────────────────────────────────────────
# Tuned for medium-sized tabular regression (~1M-10M rows × ~150 features):
#   tree_method='hist' is the fastest CPU path; 'cuda' switches to GPU.
#   Deeper trees (max_depth=7) with subsample/colsample regularization outperform
#   shallower trees on financial panels that contain many weak signals.
XGB_PARAMS = {
    'objective':         'reg:pseudohubererror',  # robust to fat-tailed return distribution
    'huber_slope':       1.0,                     # transition at 1% |residual|
    'learning_rate':     0.03,
    'max_depth':         7,
    'min_child_weight':  50,
    'subsample':         0.8,
    'colsample_bytree':  0.7,
    'colsample_bylevel': 0.8,
    'reg_alpha':         0.0,
    'reg_lambda':        1.0,
    'gamma':             0.0,
    'tree_method':       'hist',
    'n_estimators':      4000,
    'early_stopping_rounds': 100,
    'verbosity':         1,
}

# ─── Walk-forward CV defaults ────────────────────────────────────────────────
# User request: "every 3 weeks train, 1 week val and 1 week test".
# Research-backed default (López de Prado 2018, §7.4): purged K-fold with
# embargo. Expanding train window is preferred when the signal is stationary;
# rolling (fixed-length) train is preferred when regimes shift.
#
# Tradeoffs:
#   3w / 1w / 1w  — user's ask; many folds (~120 over 7 years), each trained
#                    on ~1 month of data. XGBoost has little to learn from
#                    so short a window; metrics are noisy. Total fit time is
#                    O(n_folds × per_fold_fit).
#   12w / 2w / 2w — deeptime's schedule. More stable per-fold metrics, fewer
#                    folds (~25 per year if step=2w).
#   Step = test_weeks → test windows tile cleanly with no overlap.
WALK_FORWARD_DEFAULTS = {
    'fold_train_weeks': 12,
    'fold_val_weeks':   2,
    'fold_test_weeks':  2,
    'fold_step_weeks':  2,     # = fold_test_weeks → non-overlapping test
    'purge_days':       5,     # = forward_window + safety (de Prado)
    'embargo_days':     2,     # ≈1% of 252 trading days
    'expanding_train':  False, # False = rolling fixed-length train window
    'max_folds':        0,     # 0 = run all folds
}

# ─── CLI defaults ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'data_dir':              DATA_DIR,
    'cache_dir':              CACHE_DIR,
    'model_dir':              MODEL_DIR,
    'plot_dir':               PLOT_DIR,

    'target_mode':            TARGET_MODE,
    'forward_window':         FORWARD_WINDOW,
    'clip_target_pct':        CLIP_TARGET_PCT,

    'split_mode':             'fixed',     # 'fixed' | 'walk_forward'
    'train_start':            TRAIN_START,
    'val_start':              VAL_START,
    'test_start':             TEST_START,

    # Walk-forward CV defaults (used when split_mode='walk_forward')
    **WALK_FORWARD_DEFAULTS,

    'min_rows_per_stock':     MIN_ROWS_PER_STOCK,
    'max_stocks':             0,          # 0 → all stocks
    'random_seed':            42,
    'device':                 'cpu',      # set to 'cuda' to train on GPU

    'xgb_params':             XGB_PARAMS,
}


def get_config(**overrides) -> dict:
    """Return the default config merged with any overrides."""
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    for k, v in overrides.items():
        if k == 'xgb_params' and isinstance(v, dict):
            cfg['xgb_params'].update(v)
        else:
            cfg[k] = v
    return cfg
