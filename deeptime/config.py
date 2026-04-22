"""
Configuration for the deeptime regression pipeline.

Key differences from dl/config.py:
  - FORWARD_WINDOWS = [1,2,3,4,5]  (vs [3,4,5] in dl/)
  - Regression targets (excess return float32, not 7-class int64)
  - seq_len = 60  (vs 30 in dl/)
  - Extended features: fina_indicator (12) + block_trade (4) + moneyflow SM/MD (2)
  - Known-future stream includes price limit ratios (29 features total)
"""

import os
import torch

# Reduce CUDA memory fragmentation — expandable_segments is Linux-only,
# so only set it there; on Windows this env var has no effect but avoids a warning.
if os.name != 'nt':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ─── Prediction horizons ──────────────────────────────────────────────────────
FORWARD_WINDOWS  = [1, 2, 3, 4, 5]   # trading days ahead
NUM_HORIZONS     = len(FORWARD_WINDOWS)
HORIZON_WEIGHTS  = [1.0, 1.0, 1.0, 1.0, 1.0]

# ─── Sequence length ──────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 30   # matches dl/; seq_len=60 exhausts VRAM on RTX 4070 Super
MAX_FORWARD_WINDOW = max(FORWARD_WINDOWS)   # = 5

# ─── Target configuration ─────────────────────────────────────────────────────
# 'excess': pct_chg - csi300_pct_chg (regime-invariant, symmetric around 0)
# 'raw':    raw pct_chg
TARGET_MODE = 'excess'

# ─── Loss function ────────────────────────────────────────────────────────────
HUBER_DELTA = 1.0    # Huber transition point (1% excess return)

# ─── Rolling walk-forward split (same as dl/) ─────────────────────────────────
ROLLING_TRAIN_MONTHS = 12
ROLLING_VAL_MONTHS   = 2
ROLLING_TEST_MONTHS  = 2
ROLLING_STEP_MONTHS  = 16
INTERLEAVED_TEST_START = 20250701   # global holdout
PURGE_GAP_DAYS = SEQUENCE_LENGTH + MAX_FORWARD_WINDOW   # = 35

# ─── New feature columns ──────────────────────────────────────────────────────

FINA_INDICATOR_COLUMNS = [
    'roe', 'roa', 'grossprofit_margin', 'netprofit_margin',
    'current_ratio', 'quick_ratio', 'debt_to_assets',
    'assets_yoy', 'equity_yoy', 'op_yoy', 'ebt_yoy', 'eps',
    # Binary flag: 1 = fina data available (ann_date ≤ trade_date), 0 = pre-announcement
    # Lets model distinguish "zero fundamentals because data missing" from genuine zeros.
    'has_fina_data',
]

BLOCK_TRADE_COLUMNS = [
    'block_vol_ratio', 'block_amt_ratio', 'block_count_log', 'block_buy_sell_ratio',
]

EXTRA_MONEYFLOW_COLUMNS = [
    'net_sm_flow_ratio', 'net_md_flow_ratio',
]

PRICE_LIMIT_RATIO_COLUMNS = [
    'up_limit_ratio', 'down_limit_ratio',
]

# ─── Extended features from new data sources ─────────────────────────────────

# From earnings forecast (业绩预告)
FORECAST_FEATURES = [
    'has_forecast',        # 是否有业绩预告 (0/1)
    'forecast_direction',  # 预告方向: -1=预减, 0=不确定, 1=预增
    'forecast_magnitude',  # 预告变化幅度 normalized to [-1, 1]
]

# From express earnings report (业绩快报)
EXPRESS_FEATURES = [
    'has_express',         # 是否有业绩快报 (0/1)
    'express_growth',      # 业绩快报营收同比增长 normalized
    'express_profit_yoy',  # 业绩快报利润同比增长 normalized
]

# From limit/dragon-tiger data (涨跌停/龙虎榜)
LIMIT_TS_FEATURES = [
    'limit_times',         # 连板次数 (capped at 10)
    'on_dragon_tiger',     # 是否上龙虎榜 (0/1)
    'dragon_tiger_net',    # 龙虎榜净买入比例 [-1, 1]
]

# From chip distribution (筹码分布)
CHIP_FEATURES = [
    'winner_rate',         # 获利比例 [0, 1]
    'cost_concentration',  # 成本集中度 [0, 2]
]

# Static features from financial statements (for GAT)
STATIC_FINANCIAL_FEATURES = [
    'debt_to_equity',        # 负债/权益比
    'current_ratio_static',  # 流动比率
    'asset_turnover',        # 资产周转率
    'gross_margin_static',   # 毛利率
    'operating_margin',      # 营业利润率
    'net_margin_static',     # 净利率
    'has_dividend',          # 是否有分红
    'avg_div_yield',         # 平均股息率
    'div_consistency',       # 分红一致性
]

# Static features from THS membership
STATIC_MEMBERSHIP_FEATURES = [
    'n_ths_concepts',        # 所属概念数量
    'n_ths_industries',      # 所属行业数量
    'is_hot_concept',        # 是否热门概念股
]

# All extended time series features
EXTENDED_TS_FEATURES = (
    FORECAST_FEATURES +
    EXPRESS_FEATURES +
    LIMIT_TS_FEATURES +
    CHIP_FEATURES
)

# All extended static features (for GAT node embeddings)
EXTENDED_STATIC_FEATURES = (
    STATIC_FINANCIAL_FEATURES +
    STATIC_MEMBERSHIP_FEATURES
)

# ─── All feature columns ──────────────────────────────────────────────────────
# Import the base features from dl/config and extend them.
# Importing at module level to keep DT_FEATURE_COLUMNS as a plain list.

def _build_dt_feature_columns():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dl.config import FEATURE_COLUMNS as DL_FEATURE_COLUMNS
    return (
        list(DL_FEATURE_COLUMNS)
        + list(PRICE_LIMIT_RATIO_COLUMNS)   # continuous limit ratios (better than binary)
        + list(FINA_INDICATOR_COLUMNS)
        + list(BLOCK_TRADE_COLUMNS)
        + list(EXTRA_MONEYFLOW_COLUMNS)
        + list(EXTENDED_TS_FEATURES)        # NEW: forecast, express, limit, chip features
    )


DT_FEATURE_COLUMNS = _build_dt_feature_columns()

# Known-future: 27 calendar features (from dl/) + 2 price limit ratio features
_DL_KNOWN_FUTURE = [
    'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos',
    'month_sin', 'month_cos', 'woy_sin', 'woy_cos',
    'doy_sin', 'doy_cos', 'quarter_sin', 'quarter_cos',
    'is_monday', 'is_friday', 'is_month_start', 'is_month_end',
    'is_year_start', 'is_year_end',
    'is_pre_holiday', 'is_post_holiday',
    'days_to_holiday_norm', 'days_from_holiday_norm', 'holiday_effect',
    'is_january', 'is_december', 'is_earnings_season', 'is_weak_season',
]

DT_KNOWN_FUTURE_COLUMNS = _DL_KNOWN_FUTURE + list(PRICE_LIMIT_RATIO_COLUMNS)   # 29 total

_DT_FUTURE_SET         = set(DT_KNOWN_FUTURE_COLUMNS)
_DT_FUTURE_FEAT_IDX    = [DT_FEATURE_COLUMNS.index(c) for c in DT_KNOWN_FUTURE_COLUMNS]
_DT_OBS_PAST_FEAT_IDX  = [i for i, c in enumerate(DT_FEATURE_COLUMNS) if c not in _DT_FUTURE_SET]
DT_OBSERVED_PAST_COLUMNS = [DT_FEATURE_COLUMNS[i] for i in _DT_OBS_PAST_FEAT_IDX]

NUM_DT_FEATURES          = len(DT_FEATURE_COLUMNS)         # 233
NUM_DT_KNOWN_FUTURE      = len(DT_KNOWN_FUTURE_COLUMNS)    # 29
NUM_DT_OBSERVED_PAST     = len(DT_OBSERVED_PAST_COLUMNS)   # 204

# ─── Cross-section normalization (extend dl/'s list with new features) ────────
# These features are normalized cross-sectionally (ranked within each day) to
# handle distribution shifts over time. The sanity check flagged ps, ps_ttm,
# roe, op_yoy as having >5× std shift between 2017-2019 and 2023-2025.
DT_CS_NORMALIZE_TECH_FEATURES = [
    'returns', 'log_returns', 'volume_change',
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
    'momentum_5', 'momentum_10',
    'roc_5', 'roc_10', 'roc_20',
    'volatility_5', 'volatility_10', 'volatility_20',
    'dist_from_high_20', 'dist_from_low_20',
    'obv_ratio',
    'net_lg_flow_ratio', 'net_elg_flow_ratio',
    'net_sm_flow_ratio', 'net_md_flow_ratio',
    'block_vol_ratio', 'block_amt_ratio',
    # Extended features that benefit from cross-sectional normalization
    'dragon_tiger_net',     # relative to market
    'cost_concentration',   # chip distribution metric
    # Features with detected distribution shift (sanity_checks.py flagged these)
    'ps', 'ps_ttm',         # P/S ratios: collapsed variance post-2020
    'roe',                  # earnings quality shift post-COVID
    'op_yoy',               # YoY operating profit: base effect
]

# ─── Sector / static embeddings ───────────────────────────────────────────────
# Shenwan (申万) classification from api/sector_info.py
NUM_SECTORS_EMBED     = 35    # SW L1: 31 sectors + padding
NUM_INDUSTRIES_EMBED  = 140   # SW L2: ~130 sub-industries + padding
NUM_SUB_IND_EMBED     = 10    # unused (kept for compat); SW L3 not downloaded
NUM_SIZE_DECILES      = 11    # market-cap decile 0-9 + unknown

# New static variates from stock_basic / sector_info.py
NUM_AREAS_EMBED       = 45    # provinces/regions (~35 unique + padding)
NUM_BOARD_TYPES       = 6     # 主板/中小板/创业板/科创板/北交所 + unknown
NUM_IPO_AGE_BUCKETS   = 7     # <1yr, 1-2yr, 2-3yr, 3-5yr, 5-10yr, >10yr + unknown

# THS concept membership (from ths_index)
NUM_THS_CONCEPTS      = 500   # THS has ~400+ concept indices + padding

SECTOR_EMB_DIM     = 64   # SW L1 sector (31 → 64-dim)
INDUSTRY_EMB_DIM   = 32   # SW L2 sub-industry (~130 → 32-dim)
SUB_IND_EMB_DIM    = 8    # placeholder (SW L3 not used)
SIZE_EMB_DIM       = 16   # market-cap decile
AREA_EMB_DIM       = 16   # province/region
BOARD_EMB_DIM      = 8    # exchange board type
IPO_AGE_EMB_DIM    = 8    # IPO age bucket
THS_CONCEPT_EMB_DIM = 16  # THS concept membership (multi-hot → 16-dim)

# Number of continuous static features (from financial statements)
NUM_STATIC_CONTINUOUS = len(STATIC_FINANCIAL_FEATURES)  # 9 features

# Total static_dim = 64+32+8+16+16+8+8+16 + 9 = 177

# ─── Model architecture ───────────────────────────────────────────────────────
# Default config tuned for RTX 4070 Super (12 GB VRAM).
# For RTX 5090 (32 GB VRAM), see RTX_5090_CONFIG preset below.
#
# VRAM budget formula:
#   VSN activation = B × T × num_vars × hidden × 2 (FP16)
#   Example: B=256, T=30, V=204, H=128 → 256×30×204×128×2 = 0.40 GB
#   Total with gradients + LSTM + attn + optimizer: ~8-10 GB
#
# TFT-GAT paper recommendations (https://www.mdpi.com/2673-9909/5/4/176):
#   hidden_size: 128-512 for large datasets
#   attention_heads: 4-8 (head_dim = hidden/heads should be 32-64)
#   dropout: 0.1-0.3 (higher for larger models)
#   lstm_layers: 2 (diminishing returns beyond)
TFT_HIDDEN      = 128
TFT_HEADS       = 4
TFT_LSTM_LAYERS = 2
# Dropout raised 0.1→0.15: reduces overfitting on subset runs (≤1000 stocks).
# The SectorGAT B×B attention matrix and EfficientVSN shared projection both
# benefit from stronger regularisation when train sequences are few.
TFT_DROPOUT     = 0.15

# Tushare API token (same as dl/)
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'

# ─── Default config ───────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data')

DEFAULT_CONFIG = {
    'data_dir':        _DATA_DIR,
    'cache_dir':       os.path.join(_DATA_DIR, 'deeptime_cache'),
    'sector_file':     'stock_sectors.csv',

    # Data
    'sequence_length':          SEQUENCE_LENGTH,
    'forward_window':           MAX_FORWARD_WINDOW,
    'min_data_points':          100,
    'max_stocks':               100,
    'max_sequences_per_stock':  None,
    'num_workers':              0,
    # chunk_samples: controls peak RAM during training and GPU utilization
    # Auto-scaled in loader: max(chunk_samples, batch_size * 50)
    # For batch=4096: auto-scales to 204K samples (~5 GB RAM per chunk)
    # For batch=192:  20K is fine (~0.5 GB per chunk)
    # Set higher for better GPU utilization on high-end GPUs (RTX 5090 etc)
    'chunk_samples':            100_000,   # raised from 20K for better GPU util
    'prefetch_factor':          2,         # number of chunks to prefetch (double-buffering)

    # Target
    'target_mode':   TARGET_MODE,
    'huber_delta':   HUBER_DELTA,
    'loss_type':     'huber',   # 'huber' or 'huber+ic'

    # Rolling split
    'split_mode':                  'rolling_window',
    'rolling_train_months':        ROLLING_TRAIN_MONTHS,
    'rolling_val_months':          ROLLING_VAL_MONTHS,
    'rolling_test_months':         ROLLING_TEST_MONTHS,
    'rolling_step_months':         ROLLING_STEP_MONTHS,
    'interleaved_test_start':      INTERLEAVED_TEST_START,
    'purge_gap_days':              PURGE_GAP_DAYS,

    # Model
    'tft_hidden':       TFT_HIDDEN,
    'tft_heads':        TFT_HEADS,
    'tft_lstm_layers':  TFT_LSTM_LAYERS,
    'tft_dropout':      TFT_DROPOUT,
    'num_sectors':      NUM_SECTORS_EMBED,
    'num_industries':   NUM_INDUSTRIES_EMBED,
    'num_sub_ind':      NUM_SUB_IND_EMBED,
    'num_size_deciles': NUM_SIZE_DECILES,
    'num_areas':        NUM_AREAS_EMBED,
    'num_board_types':  NUM_BOARD_TYPES,
    'num_ipo_age':      NUM_IPO_AGE_BUCKETS,

    # Training
    # batch=192: ~900 seqs/s, peak ~8 GB dedicated VRAM — safe after adding 3 new
    #            static arrays (areas/boards/ipo_ages) raised peak from 8.5→10.6 GB at 256
    # batch=256: was 8.5 GB but now 10.6 GB with new static arrays — too close to 11.6 GB limit
    # batch=128: 927 seqs/s, peak ~5.5 GB — safe fallback
    'batch_size':               192,
    'epochs':                   50,

    # LR: 2e-5 is safer default for subset runs (≤1000 stocks).
    # At 5e-5 gradient norms grow 0.5→20 during warmup causing val-loss divergence.
    # For the full 5190-stock run, try 3e-5 or 5e-5.
    'learning_rate':            2e-5,

    # Weight decay: 0.05→0.1 to fight overfitting (train IC 0.47 vs val IC 0.05
    # after 13 epochs indicates severe overfit; stronger L2 is the right lever).
    'weight_decay':             0.1,

    # Gradient clip lowered 1.0→0.5: pre-clip norms of 15-20 with max_norm=1.0
    # scales each step to 5-7% of raw gradient (direction-distorting).
    # 0.5 keeps the scaling ratio ≥2.5% so step directions are meaningful.
    'max_grad_norm':            0.5,

    'early_stopping_patience':  15,
    'base_batch_for_lr':        192,  # reference batch size for LR linear scaling
    'warmup_epochs':            2,
    'use_amp':                  True,
    'random_seed':              42,

    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
}


# ─── RTX 5090 Optimized Config (32 GB VRAM) ──────────────────────────────────
# Based on TFT-GAT paper (https://www.mdpi.com/2673-9909/5/4/176) and
# RTX 5090 specs (32GB GDDR7, 1.79 TB/s bandwidth, 680 5th-gen Tensor Cores).
#
# Usage: python -m deeptime.main --preset rtx5090 ...
#   Or manually: --batch_size 512 --hidden 256 --heads 8 --seq_len 60 \
#                --lr 5e-5 --warmup_epochs 5 --dropout 0.2
#
# VRAM estimate at these settings: ~8-10 GB (leaves 22GB headroom)
# Can push to batch=768, hidden=384 if needed.
RTX_5090_CONFIG = {
    # Model — larger capacity for 32GB VRAM
    'tft_hidden':       256,       # 2× default (richer representations)
    'tft_heads':        8,         # 2× default (multi-aspect attention)
    'tft_lstm_layers':  2,         # keep 2 (diminishing returns)
    'tft_dropout':      0.20,      # slightly higher for larger model
    'sequence_length':  60,        # 2× default (more temporal context)

    # Training — larger batches for better GPU utilization
    'batch_size':       512,       # 2.7× default (fits in 32GB)
    'learning_rate':    5e-5,      # sqrt-scaled: 2e-5 × √(512/192) ≈ 3.3e-5, rounded up
    'warmup_epochs':    5,         # longer warmup for larger batches
    'weight_decay':     0.1,       # keep same
    'max_grad_norm':    0.5,       # keep same

    # Data loading — optimized for RTX 5090 bandwidth
    'chunk_samples':    250_000,   # ~6 GB RAM per chunk, 50+ batches per chunk
    'prefetch_factor':  3,         # triple-buffering for high bandwidth

    # Early stopping
    'early_stopping_patience': 20, # more patience for larger model
    'base_batch_for_lr': 512,      # disable auto LR scaling (already tuned)
}

# Aggressive RTX 5090 config — use if IC plateaus with conservative settings
RTX_5090_AGGRESSIVE_CONFIG = {
    'tft_hidden':       384,
    'tft_heads':        8,         # head_dim = 384/8 = 48
    'tft_lstm_layers':  2,
    'tft_dropout':      0.25,
    'sequence_length':  60,

    'batch_size':       768,
    'learning_rate':    8e-5,
    'warmup_epochs':    8,
    'weight_decay':     0.15,
    'max_grad_norm':    0.5,

    # Data loading — max throughput
    'chunk_samples':    300_000,   # ~8 GB RAM per chunk
    'prefetch_factor':  3,

    'early_stopping_patience': 25,
    'base_batch_for_lr': 768,
}

PRESET_CONFIGS = {
    'rtx5090': RTX_5090_CONFIG,
    'rtx5090_aggressive': RTX_5090_AGGRESSIVE_CONFIG,
}


def get_config(preset: str = None, **overrides) -> dict:
    """
    Get configuration with optional preset and overrides.

    Args:
        preset: Optional preset name ('rtx5090', 'rtx5090_aggressive')
        **overrides: Override any config key

    Returns:
        Configuration dict
    """
    cfg = DEFAULT_CONFIG.copy()
    if preset and preset in PRESET_CONFIGS:
        cfg.update(PRESET_CONFIGS[preset])
    cfg.update(overrides)
    return cfg


def get_horizon_name(h: int) -> str:
    fw = FORWARD_WINDOWS[h] if h < len(FORWARD_WINDOWS) else h + 1
    return f"day{fw}"
