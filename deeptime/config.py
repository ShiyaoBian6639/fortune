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
DT_CS_NORMALIZE_TECH_FEATURES = [
    'returns', 'log_returns', 'volume_change',
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
    'momentum_5', 'momentum_10',
    'roc_5', 'roc_10', 'roc_20',
    'volatility_5', 'volatility_10', 'volatility_20',
    'dist_from_high_20', 'dist_from_low_20',
    'obv_ratio',
    'net_lg_flow_ratio', 'net_elg_flow_ratio',
    'net_sm_flow_ratio', 'net_md_flow_ratio',  # new
    'block_vol_ratio', 'block_amt_ratio',       # new
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

SECTOR_EMB_DIM     = 64   # SW L1 sector (31 → 64-dim)
INDUSTRY_EMB_DIM   = 32   # SW L2 sub-industry (~130 → 32-dim)
SUB_IND_EMB_DIM    = 8    # placeholder (SW L3 not used)
SIZE_EMB_DIM       = 16   # market-cap decile
AREA_EMB_DIM       = 16   # province/region
BOARD_EMB_DIM      = 8    # exchange board type
IPO_AGE_EMB_DIM    = 8    # IPO age bucket
# Total static_dim = 64+32+8+16+16+8+8 = 152

# ─── Model architecture ───────────────────────────────────────────────────────
# RTX 4070 Super (12 GB VRAM) budget:
#   VSN activation = B*T × num_vars × hidden × fp16
#   With B=256, T=30, num_vars=204, hidden=128: 256×30×204×128×2 = 0.40 GB
#   With gradients + LSTM + attn + optimizer states: ~5 GB total — safe.
TFT_HIDDEN      = 128   # was 160; reduces VSN by 20%, fits 12 GB VRAM
TFT_HEADS       = 4
TFT_LSTM_LAYERS = 2
TFT_DROPOUT     = 0.1

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
    # chunk_samples: controls peak RAM during training
    # 20K × 30 × 204 × float32 = 0.49 GB per chunk; 3× prefetch depth ≈ 1.5 GB RAM
    'chunk_samples':            20_000,

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
    'learning_rate':            5e-5,
    'early_stopping_patience':  15,
    'warmup_epochs':            8,
    'use_amp':                  True,
    'random_seed':              42,

    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
}


def get_config(**overrides) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(overrides)
    return cfg


def get_horizon_name(h: int) -> str:
    fw = FORWARD_WINDOWS[h] if h < len(FORWARD_WINDOWS) else h + 1
    return f"day{fw}"
