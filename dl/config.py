"""
Configuration and constants for stock price prediction.
"""

import os
import torch

# Buckets for multi-class classification.
# Labels are CROSS-SECTIONAL RELATIVE returns: stock_return − CSI300_return (same horizon).
# Symmetric boundaries centred at 0: half the stocks always outperform, half underperform,
# so the model cannot gain by predicting "bear" for everything regardless of market regime.
# Boundaries are calibrated for 3–5 day relative returns in Chinese A-shares.
# Each tuple is (min_pct, max_pct, label_name)
CHANGE_BUCKETS = [
    (-float('inf'), -4.3, 'strong underperform'),  # bottom ~14%
    (         -4.3, -2.0, 'underperform'),          # ~14%
    (         -2.0, -0.5, 'mild underperform'),     # ~16%
    (         -0.5,  0.5, 'neutral'),               # ~16%
    (          0.5,  2.0, 'mild outperform'),       # ~16%
    (          2.0,  4.3, 'outperform'),            # ~14%
    (          4.3, float('inf'), 'strong outperform'),  # top ~14%
]

NUM_CLASSES = len(CHANGE_BUCKETS)   # 7

# Relative-return auxiliary head: stock return − CSI300 return (same horizon).
# Distribution is approximately symmetric and roughly equiprobable (no class weights needed).
# 5 symmetric classes based on σ ≈ 1–2% for 3–5 day relative returns.
RELATIVE_CHANGE_BUCKETS = [
    (-float('inf'), -3, 'strong underperform'),  # bottom quintile
    (          -3,  -1, 'underperform'),
    (          -1,   1, 'neutral'),              # near-zero alpha
    (           1,   3, 'outperform'),
    (           3,  float('inf'), 'strong outperform'),  # top quintile
]
NUM_RELATIVE_CLASSES = len(RELATIVE_CHANGE_BUCKETS)   # 5

# Train/val/test split mode.
# 'rolling_window':  alternating 3m-train / 1m-val blocks across all history; test = 2025-07+.
#                    Both train and val see all seasons → no seasonal distribution shift.
# 'temporal':        simple chronological 70/15/15 split with purge gaps.
# 'interleaved_val': Q1 (Jan–Mar) of 2018–2025 as val windows; test = 2025-07+.
#                    NOTE: causes seasonal distribution shift (Q2–Q4 train vs Q1 val).
# 'regime':          regime-aware split using CSI300 MA-250 signal.
# 'random':          legacy random-permutation split (susceptible to contamination).
SPLIT_MODE             = 'rolling_window'
REGIME_MIN_BLOCK_DAYS  = 40     # merge regime blocks shorter than this (trading days)
REGIME_PURGE_GAP_DAYS  = 35     # gap at each split boundary = seq_len(30) + max_fw(5) = 35
REGIME_VAL_DAYS        = 200    # recent trading days of the major bear block → val
REGIME_TEST_DAYS       = 190    # recent trading days of the final block → test

# Interleaved validation windows: Q1 (Jan 1 → Apr 1) of each year 2018–2025.
# Each tuple is (start_date_int, end_date_int) — end is exclusive.
# Training includes Q2–Q4 of every year so the model sees bull and bear each year.
# Test window: 2025-07-01 onwards — pure holdout, never touched during training.
INTERLEAVED_VAL_WINDOWS = [
    (20180101, 20180401),
    (20190101, 20190401),
    (20200101, 20200401),
    (20210101, 20210401),
    (20220101, 20220401),
    (20230101, 20230401),
    (20240101, 20240401),
    (20250101, 20250401),
]
INTERLEAVED_TEST_START = 20250701   # pure holdout; not used during training/val

# Rolling walk-forward split — strict temporal ordering, no leakage.
#
# Each fold: [TRAIN months] → purge → [VAL months] → purge → [TEST months]
# Cursor advances by ROLLING_STEP_MONTHS after each fold.
#
# Leakage guarantee:
#   Once a date is labeled val or test in any fold, it cannot be re-labeled train
#   in a subsequent fold (the "only overwrite 'train'" rule in the split code).
#   Safe when ROLLING_STEP_MONTHS >= ROLLING_VAL_MONTHS + ROLLING_TEST_MONTHS.
#   If step < val+test, overlapping folds are still correct but some past val/test
#   dates are withheld from later training windows — acceptable for expanding windows.
#
# All folds whose test_end <= INTERLEAVED_TEST_START contribute to the pooled
# train/val/test sets used for model fitting and held-out evaluation.
# Data from INTERLEAVED_TEST_START onwards is always reserved as a final holdout.
#
# Tuning guide:
#   Longer TRAIN → more stable gradient estimates, less folds.
#   Larger STEP  → less overlap between folds, fewer folds, faster cache build.
#   STEP = VAL + TEST → perfectly non-overlapping folds (recommended default).
ROLLING_TRAIN_MONTHS = 12   # months of training data per fold
ROLLING_VAL_MONTHS   = 2    # months of validation data per fold (immediately after train)
ROLLING_TEST_MONTHS  = 2    # months of test data per fold (immediately after val)
ROLLING_STEP_MONTHS  = 16   # how far cursor advances per fold
# IMPORTANT: set ROLLING_STEP_MONTHS = TRAIN+VAL+TEST for non-overlapping folds (recommended).
# A smaller step creates overlapping folds: each fold adds one new val+test window, but
# because training = everything NOT in any val/test window, a step << TRAIN means the
# accumulating val/test windows will eat into the effective training pool.
# Rule of thumb: step >= VAL + TEST to keep val/test non-overlapping across folds.

# Multi-horizon label settings.
# Each sequence gets one label per horizon (day 3, 4, 5 close-to-close return).
# The model has one independent classification head per horizon; they share the
# Transformer backbone but learn separate decision boundaries.
FORWARD_WINDOWS  = [3, 4, 5]           # horizons (trading days ahead)
NUM_HORIZONS     = len(FORWARD_WINDOWS) # 3
HORIZON_WEIGHTS  = [1.0, 1.0, 1.0]     # loss weight per horizon (equal initially)

# Default configuration
DEFAULT_CONFIG = {
    # Data paths
    'data_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data'),
    'sector_file': 'stock_sectors.csv',

    # Data processing
    'sequence_length': 30,
    'forward_window': 5,       # Max horizon (= max(FORWARD_WINDOWS)); controls look-ahead guard
    'min_data_points': 100,
    'max_stocks': 100,  # Set to None for full dataset
    'max_sequences_per_stock': 600,
    'num_workers': 0,  # val/test only; train uses ChunkedMemmapLoader (background thread, no spawn)
    'chunk_samples': 40_000,   # samples per chunk ~512 MB; peak RAM ≈ 3× (depth-2 prefetch) ≈ 1.5 GB

    # Model architecture
    # Reduced from d_model=256/4-layers to 128/2-layers: fewer params → less
    # overfitting on noisy financial data; ~4× faster per epoch.
    # d_model=192 / 6 heads → head_dim=32 (same as before); 3 layers for richer
    # temporal feature extraction from 30-step × 213-feature sequences.
    'd_model': 192,
    'nhead': 6,
    'num_layers': 3,
    'dim_feedforward': 768,    # 4 × d_model
    'dropout': 0.1,            # 0.3 caused underfitting (best val at epoch 3)

    # Model type selection
    'model_type': 'transformer',  # 'transformer' or 'tft'

    # TFT-specific architecture
    'tft_hidden':      160,   # d_model for TFT (must be divisible by tft_heads)
    'tft_heads':         4,   # InterpretableMultiHeadAttention heads
    'tft_lstm_layers':   2,   # stacked LSTM layers in encoder/decoder
    'tft_dropout':      0.1,

    # Training
    'batch_size': 512,
    'epochs': 50,
    'learning_rate': 5e-5,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'early_stopping_patience': 15,
    'warmup_epochs': 8,

    # Relative-return auxiliary head disabled: both main labels and relative_labels
    # now compute stock − CSI300 (same signal, different bucket counts).
    # The aux head provides no independent gradient; disabling it removes noise.
    'use_relative_head': False,
    'relative_head_weight': 0.3,

    # Loss function settings
    # Switched from focal+class_weights to ce+label_smoothing:
    #   focal (gamma=2) suppresses easy-example gradients while class_weights
    #   amplify rare-class gradients → extreme overconfidence → T=7-25.
    #   CE+label_smoothing(0.1) is calibrated out of the box; T should drop to ~1-3.
    'loss_type': 'ce',     # Options: 'ce' (CrossEntropy), 'focal', 'cb'
    # class_weights with CE at scale: even CE + class_weights causes T=127 (catastrophic
    # overconfidence) when the imbalance ratio is 7.6x and full dataset is used.
    # Mechanism: 7.6x gradient amplification on rare extreme classes → model learns to
    # predict rare classes with extreme confidence → wrong confident predictions → T explodes.
    # Use label_smoothing=0.1 (below) as the sole regularisation against overconfidence.
    # If bear-bias is a problem, address it via post-hoc threshold calibration, not weights.
    'use_class_weights': False,
    'use_weighted_sampling': False,  # Can cause extreme predictions
    'focal_gamma': 2.0,
    'focal_alpha': None,
    'label_smoothing': 0.1,  # Prevents overconfidence on noisy financial labels
    'cb_beta': 0.9999,

    # Mixed precision
    'use_amp': True,   # FP16 autocast + GradScaler; ~2x faster matmul on Ampere GPUs

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'random_seed': 42,
}

# Tushare API token
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'

# Chinese holidays data
CHINESE_HOLIDAYS_FIXED = {
    'new_year': [(1, 1, 'New Year', 1)],
    'labor_day': [(5, 1, 'Labor Day', 5)],
    'national_day': [(10, 1, 'National Day', 7)],
}

# Lunar calendar festival dates (year -> (month, day))
SPRING_FESTIVAL_DATES = {
    2017: (1, 27), 2018: (2, 15), 2019: (2, 4), 2020: (1, 24),
    2021: (2, 11), 2022: (1, 31), 2023: (1, 21), 2024: (2, 9),
    2025: (1, 28), 2026: (2, 16), 2027: (2, 5),
}

QINGMING_DATES = {
    2017: (4, 2), 2018: (4, 5), 2019: (4, 5), 2020: (4, 4),
    2021: (4, 3), 2022: (4, 3), 2023: (4, 5), 2024: (4, 4),
    2025: (4, 4), 2026: (4, 5), 2027: (4, 5),
}

DRAGON_BOAT_DATES = {
    2017: (5, 28), 2018: (6, 16), 2019: (6, 7), 2020: (6, 25),
    2021: (6, 12), 2022: (6, 3), 2023: (6, 22), 2024: (6, 8),
    2025: (5, 31), 2026: (6, 19), 2027: (6, 9),
}

MID_AUTUMN_DATES = {
    2017: (10, 4), 2018: (9, 22), 2019: (9, 13), 2020: (10, 1),
    2021: (9, 19), 2022: (9, 10), 2023: (9, 29), 2024: (9, 15),
    2025: (10, 6), 2026: (9, 25), 2027: (9, 15),
}

DOUBLE_NINTH_DATES = {
    2017: (10, 28), 2018: (10, 17), 2019: (10, 7), 2020: (10, 25),
    2021: (10, 14), 2022: (10, 4), 2023: (10, 23), 2024: (10, 11),
    2025: (10, 29), 2026: (10, 18), 2027: (10, 8),
}

WINTER_SOLSTICE_DATES = {
    2017: (12, 22), 2018: (12, 22), 2019: (12, 22), 2020: (12, 21),
    2021: (12, 21), 2022: (12, 22), 2023: (12, 22), 2024: (12, 21),
    2025: (12, 21), 2026: (12, 22), 2027: (12, 22),
}

QIXI_DATES = {
    2017: (8, 28), 2018: (8, 17), 2019: (8, 7), 2020: (8, 25),
    2021: (8, 14), 2022: (8, 4), 2023: (8, 22), 2024: (8, 10),
    2025: (8, 29), 2026: (8, 19), 2027: (8, 8),
}

LABA_DATES = {
    2017: (1, 5), 2018: (1, 24), 2019: (1, 13), 2020: (1, 2),
    2021: (1, 20), 2022: (1, 10), 2023: (12, 30), 2024: (1, 18),
    2025: (1, 7), 2026: (1, 26), 2027: (1, 15),
}

# Market-wide context features derived from major Chinese and global indices.
# These are date-level signals shared across all stocks on the same trading day.
MARKET_CONTEXT_FEATURES = [
    # CSI300 (沪深300) daily valuation — from index_dailybasic/000300_SH.csv
    'csi300_pe_ttm', 'csi300_pb', 'csi300_turnover',
    # CSI500 (中证500) daily valuation — from index_dailybasic/000905_SH.csv
    'csi500_pe_ttm', 'csi500_pb',
    # SSE Composite (上证综指) daily valuation — from index_dailybasic/000001_SH.csv
    'sse_pe_ttm',
    # CSI300 technical indicators — from idx_factor_pro/000300_SH.csv
    'csi300_rsi6', 'csi300_macd', 'csi300_cci', 'csi300_bias1', 'csi300_kdj_k',
    # Global index daily returns, lagged 1 trading day (no lookahead into same day)
    # Rationale: DJI/IXIC close after Chinese market; HSI/N225 close before.
    # Lag-1 is a conservative safe choice that avoids all lookahead.
    'dji_ret_lag1', 'hsi_ret_lag1', 'ixic_ret_lag1', 'n225_ret_lag1',
    # CSI300 additional technical indicators — from idx_factor_pro/000300_SH.csv
    'csi300_rsi12', 'csi300_rsi24',
    'csi300_kdj_d',
    'csi300_adx', 'csi300_pdi', 'csi300_mdi',
    # NOTE: csi300_obv intentionally excluded — raw cumulative OBV is non-stationary
    # (drifts 3–7σ over multi-year spans after StandardScaler; causes OOD inputs at test time)
    'csi300_mfi',
    'csi300_updays', 'csi300_downdays',
    'csi300_roc', 'csi300_mtm',
    # SSE50 (上证50) valuation — from index_dailybasic/000016_SH.csv
    'sse50_pe_ttm', 'sse50_pb', 'sse50_turnover',
    # ChiNext/GEM (创业板) valuation — from index_dailybasic/399006_SZ.csv
    'gem_pe_ttm', 'gem_pb', 'gem_turnover',
    # CSI1000 (中证1000) valuation — from index_dailybasic/000852_SH.csv
    'csi1000_pe_ttm', 'csi1000_pb',
    # S&P 500 lagged return — from index_global/SPX.csv
    'spx_ret_lag1',
    # CSI500 technical indicators — from idx_factor_pro/000905_SH.csv
    'csi500_rsi6', 'csi500_rsi12', 'csi500_rsi24',
    'csi500_macd', 'csi500_cci', 'csi500_bias1',
    'csi500_kdj_k', 'csi500_kdj_d',
    'csi500_adx', 'csi500_pdi', 'csi500_mdi',
    'csi500_mfi',   # csi500_obv excluded (non-stationary cumulative)
    'csi500_updays', 'csi500_downdays',
    'csi500_roc', 'csi500_mtm',
    # SSE50 technical indicators — from idx_factor_pro/000016_SH.csv
    'sse50_rsi6', 'sse50_rsi12', 'sse50_rsi24',
    'sse50_macd', 'sse50_cci', 'sse50_bias1',
    'sse50_kdj_k', 'sse50_kdj_d',
    'sse50_adx', 'sse50_pdi', 'sse50_mdi',
    'sse50_mfi',    # sse50_obv excluded (non-stationary cumulative)
    'sse50_updays', 'sse50_downdays',
    'sse50_roc', 'sse50_mtm',
    # ChiNext/GEM technical indicators — from idx_factor_pro/399006_SZ.csv
    'gem_rsi6', 'gem_rsi12', 'gem_rsi24',
    'gem_macd', 'gem_cci', 'gem_bias1',
    'gem_kdj_k', 'gem_kdj_d',
    'gem_adx', 'gem_pdi', 'gem_mdi',
    'gem_mfi',      # gem_obv excluded (non-stationary cumulative)
    'gem_updays', 'gem_downdays',
    'gem_roc', 'gem_mtm',
    # CSI1000 technical indicators — from idx_factor_pro/000852_SH.csv
    'csi1000_rsi6', 'csi1000_rsi12', 'csi1000_rsi24',
    'csi1000_macd', 'csi1000_cci', 'csi1000_bias1',
    'csi1000_kdj_k', 'csi1000_kdj_d',
    'csi1000_adx', 'csi1000_pdi', 'csi1000_mdi',
    'csi1000_mfi',  # csi1000_obv excluded (non-stationary cumulative)
    'csi1000_updays', 'csi1000_downdays',
    'csi1000_roc', 'csi1000_mtm',
]

# Per-stock index membership features from index_weight files.
# Static within a quarterly rebalancing window (approximated as latest snapshot).
INDEX_MEMBERSHIP_FEATURES = [
    'is_csi300',      # constituent of CSI300 (0 / 1)
    'csi300_weight',  # weight in CSI300 index (0.0 if not a constituent)
    'is_csi500',      # constituent of CSI500 (0 / 1)
    'is_sse50',       # constituent of SSE50  (0 / 1)
]

# Feature columns used in the model
FEATURE_COLUMNS = [
    # Basic price features
    'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
    # Moving average ratios
    'sma_5_ratio', 'sma_10_ratio', 'sma_20_ratio',
    'vol_sma_5_ratio', 'vol_sma_10_ratio', 'vol_sma_20_ratio',
    # Volatility
    'volatility_5', 'volatility_10', 'volatility_20',
    # Classic indicators
    'rsi', 'macd', 'macd_signal', 'macd_diff',
    'bb_position', 'volume_change', 'price_position',
    # ATR
    'atr_ratio',
    # OBV
    'obv_ratio',
    # Stochastic
    'stoch_k', 'stoch_d', 'stoch_diff',
    # Williams %R
    'williams_r',
    # CCI
    'cci',
    # Rate of Change
    'roc_5', 'roc_10', 'roc_20',
    # Momentum
    'momentum_5', 'momentum_10',
    # ADX and Directional Indicators
    'plus_di', 'minus_di', 'adx', 'di_diff',
    # Moving Average Crossovers
    'sma_5_10_cross', 'sma_10_20_cross', 'ema_12_26_cross',
    # Lag features
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
    # Gaps
    'gap', 'gap_abs',
    # Trend indicators
    'above_sma_5', 'above_sma_10', 'above_sma_20', 'trend_score',
    # Candle patterns
    'body_size', 'upper_shadow', 'lower_shadow', 'is_bullish_candle',
    # Consecutive days
    'consecutive_up', 'consecutive_down',
    # Distance from highs/lows
    'dist_from_high_20', 'dist_from_low_20',
    # Volume-price
    'price_vs_vwap',
    # W Bottom and M Top pattern features
    'w_bottom_short', 'w_bottom_long', 'm_top_short', 'm_top_long',
    'w_bottom_signal', 'm_top_signal', 'pattern_bias',
    # Date/Time Cyclical Features
    'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos',
    'month_sin', 'month_cos', 'woy_sin', 'woy_cos',
    'doy_sin', 'doy_cos', 'quarter_sin', 'quarter_cos',
    # Trading day indicators
    'is_monday', 'is_friday', 'is_month_start', 'is_month_end',
    'is_year_start', 'is_year_end',
    # Holiday Features
    'is_pre_holiday', 'is_post_holiday',
    'days_to_holiday_norm', 'days_from_holiday_norm', 'holiday_effect',
    # Special period indicators
    'is_january', 'is_december', 'is_earnings_season', 'is_weak_season',
    # Daily Basic Features (from Tushare daily_basic)
    'turnover_rate', 'turnover_rate_f', 'volume_ratio',
    'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
    'dv_ratio', 'dv_ttm',
    'total_mv_norm', 'circ_mv_norm',  # Normalized market cap
    'float_ratio', 'free_ratio',  # Float/free share ratios
    # Market-wide context features (index_dailybasic, idx_factor_pro, index_global)
    *MARKET_CONTEXT_FEATURES,
    # Per-stock index membership (index_weight)
    *INDEX_MEMBERSHIP_FEATURES,
    # Limit up/down features (stk_limit) — will be zeros if data not downloaded yet
    'is_limit_up', 'is_limit_down',
    # Money flow features (moneyflow) — will be zeros if data not downloaded yet
    'net_lg_flow_ratio', 'net_elg_flow_ratio',
]

# TFT feature categorization.
# Known-future features: deterministic calendar/holiday features computable for any future date.
# The split happens inside TemporalFusionTransformer.forward() via _FUTURE_FEAT_IDX so the
# existing (N, 30, 213) cache format is unchanged — only future_inputs is new storage.
KNOWN_FUTURE_FEATURE_COLUMNS = [
    # Cyclical date encodings (12)
    'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos',
    'month_sin', 'month_cos', 'woy_sin', 'woy_cos',
    'doy_sin', 'doy_cos', 'quarter_sin', 'quarter_cos',
    # Boolean calendar flags (6)
    'is_monday', 'is_friday', 'is_month_start', 'is_month_end',
    'is_year_start', 'is_year_end',
    # Holiday features (5)
    'is_pre_holiday', 'is_post_holiday',
    'days_to_holiday_norm', 'days_from_holiday_norm', 'holiday_effect',
    # Seasonal flags (4)
    'is_january', 'is_december', 'is_earnings_season', 'is_weak_season',
]
_FUTURE_FEAT_SET   = set(KNOWN_FUTURE_FEATURE_COLUMNS)
_FUTURE_FEAT_IDX   = [FEATURE_COLUMNS.index(c) for c in KNOWN_FUTURE_FEATURE_COLUMNS]
_OBS_PAST_FEAT_IDX = [i for i, c in enumerate(FEATURE_COLUMNS) if c not in _FUTURE_FEAT_SET]
OBSERVED_PAST_FEATURE_COLUMNS = [FEATURE_COLUMNS[i] for i in _OBS_PAST_FEAT_IDX]
NUM_KNOWN_FUTURE_FEATURES  = len(KNOWN_FUTURE_FEATURE_COLUMNS)   # 27
NUM_OBSERVED_PAST_FEATURES = len(OBSERVED_PAST_FEATURE_COLUMNS)  # 186

# Technical features to normalise cross-sectionally (per trading day, across all stocks).
# These are features where "how does this stock compare to peers today" is meaningful.
# Computable from raw price/volume data alone (used in the lightweight first-pass function).
# Excluded: bounded indicators (RSI 0-100, stoch, williams), boolean flags, cyclical
# date encodings, market-context features (already market-wide), and valuation ratios
# (handled separately by CS_NORMALIZE_FEATURES in data_processing.py).
CS_NORMALIZE_TECH_FEATURES = [
    'returns', 'log_returns', 'volume_change',
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
    'momentum_5', 'momentum_10',
    'roc_5', 'roc_10', 'roc_20',
    'volatility_5', 'volatility_10', 'volatility_20',
    'dist_from_high_20', 'dist_from_low_20',
    'obv_ratio',
    'net_lg_flow_ratio', 'net_elg_flow_ratio',
]

# Daily basic columns to load from CSV files
DAILY_BASIC_COLUMNS = [
    'ts_code', 'trade_date', 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
    'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
    'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv'
]


def get_config(**overrides):
    """
    Get configuration with optional overrides.

    Usage:
        # Default config (100 stocks for testing)
        config = get_config()

        # Full dataset
        config = get_config(max_stocks=None)

        # Custom settings
        config = get_config(max_stocks=500, epochs=100)
    """
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return config


def get_class_names():
    """Get list of class names for reporting."""
    return [name for _, _, name in CHANGE_BUCKETS]


def get_horizon_name(horizon_idx: int) -> str:
    """
    Return a short, column-safe name for horizon index h.

    horizon_idx 0 → 'day3', 1 → 'day4', 2 → 'day5'
    (maps via FORWARD_WINDOWS so the name reflects the actual forecast day)
    """
    fw = FORWARD_WINDOWS[horizon_idx] if horizon_idx < len(FORWARD_WINDOWS) else horizon_idx + 1
    return f"day{fw}"
