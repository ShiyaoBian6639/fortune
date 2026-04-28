"""
Configuration for the multimodal stock prediction package.

Extends dl/config.py with BERT and news-specific settings.
10-class prediction matching dl/CHANGE_BUCKETS (gain in buckets).
"""

import os
import torch

from dl.config import (
    DEFAULT_CONFIG,
    FEATURE_COLUMNS,
    DAILY_BASIC_COLUMNS,
    CHANGE_BUCKETS,
    get_config,
)

# ─── Text encoder settings ────────────────────────────────────────────────────

BERT_MODEL_NAME      = 'hfl/chinese-macbert-base'
BERT_HIDDEN_DIM      = 768
NEWS_WINDOW          = 3    # rolling trading-day window (DASF-Net empirical finding)
MAX_ARTICLES_PER_DAY = 16   # cap per day; select by content length desc
BERT_MAX_LENGTH      = 128  # avg Sina article ≈ 80-120 BERT tokens

# ─── 10-class label scheme ────────────────────────────────────────────────────
# Identical formula to dl/: label = (max(high[t:t+5]) - close[t-1]) / close[t-1] * 100
# Buckets reuse dl/CHANGE_BUCKETS directly — no duplication of thresholds.

MM_CLASS_BUCKETS = CHANGE_BUCKETS          # 10 buckets, same as dl/
MM_NUM_CLASSES   = len(MM_CLASS_BUCKETS)   # 10
MM_CLASS_NAMES   = [name for _, _, name in MM_CLASS_BUCKETS]

# ─── News coverage window ─────────────────────────────────────────────────────
# Sina news data available from ~Oct 2018.  Restrict dataset to this window
# so every sample has real (non-zero) news embeddings.
NEWS_START_DATE = '20181008'   # first trading day with news coverage
NEWS_END_DATE   = '20260410'   # last date covered by sina + wallstreetcn

# ─── Multimodal config ────────────────────────────────────────────────────────

MULTIMODAL_CONFIG = {
    **DEFAULT_CONFIG,

    # Use all available stocks (override dl/ default of 100)
    'max_stocks': None,

    # Data paths (resolved relative to this file's location at runtime)
    'news_dir': os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'stock_data', 'news'
    ),
    'news_cache_path': os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'stock_data', 'cache', 'news_embeddings.npz'
    ),
    'token_cache_path': os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'stock_data', 'cache', 'news_tokens.npz'
    ),
    'multimodal_cache_dir': os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'stock_data', 'cache', 'multimodal'
    ),
    'checkpoint_dir': os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'stock_data', 'checkpoints', 'multimodal'
    ),

    # BERT settings
    'bert_model_name':       BERT_MODEL_NAME,
    'bert_hidden_dim':       BERT_HIDDEN_DIM,
    'bert_max_length':       BERT_MAX_LENGTH,
    'max_articles_per_day':  MAX_ARTICLES_PER_DAY,

    # News window
    'news_window':     NEWS_WINDOW,
    'news_start_date': NEWS_START_DATE,
    'news_end_date':   NEWS_END_DATE,
    'news_only_mode':  True,   # restrict samples to news-covered window

    # Model architecture — same as dl/ except num_classes
    'num_classes':       MM_NUM_CLASSES,
    'news_num_layers':   2,
    'd_model':           256,
    'nhead':             8,
    'num_layers':        4,    # price encoder depth
    'dim_feedforward':   1024,
    'dropout':           0.1,

    # Phase 1 batch — frozen BERT, no BERT memory pressure.
    # Bumped from 1024 → 2048 to better feed the 5090.  At d_model=256, 4
    # transformer layers, 213 features, the model is tiny relative to the GPU
    # and the previous batch left it 30–40 % utilized.
    'batch_size':         2048,

    # MultimodalChunkedLoader chunk size for Phase 1 training.
    # 100K × (30 × 213 + 3 × 768) × 4 ≈ 3.4 GB per chunk; depth-2 prefetch
    # holds at most 3 chunks → ~10 GB peak from chunks, plus ~6 GB for the
    # in-RAM price_matrix below.  Comfortable on a 64 GB host.
    'chunk_samples':      100_000,

    # Hold the entire (T_total, F) price matrix in RAM instead of memmap.
    # ~6 GB for the full dataset.  Eliminates the OS-page-cache thrashing that
    # makes random fancy-indexing on a 6 GB memmap slow, and is the single
    # biggest GPU-utilization win for this pipeline.  Set False if the host
    # has < 32 GB RAM.
    'load_price_in_ram':  True,

    # Pin host memory of gathered chunks so .to(device, non_blocking=True) can
    # overlap H2D copy with GPU compute.  'auto' enables it whenever device
    # starts with 'cuda'; False/True override.
    'pin_memory':         'auto',

    # LoRA: fine-tune only adapter matrices (~295K params) instead of full BERT (110M).
    # Adam optimizer states: ~3 MB vs ~880 MB → frees ~877 MB for larger batches.
    # target_modules=['query','value'] injects rank-8 adapters into every layer's
    # Q and V projections (standard LoRA-BERT configuration).
    'use_lora':             True,
    'lora_r':               8,
    'lora_alpha':           16,    # effective scale = alpha/r = 2×
    'lora_target_modules':  ['query', 'value'],
    'lora_dropout':         0.1,

    # Phase 2: BERT is run inline — each step pushes B×W×A sequences through BERT.
    # With LoRA, optimizer memory drops from ~880MB to ~3MB, allowing larger batches.
    # B=32, W=3, A=16 → 1536 BERT sequences per micro-batch (was 768 with batch=16).
    # Gradient checkpointing still applied to keep activation memory bounded.
    'phase2_batch_size':   32,   # 32×3×16=1536 BERT seqs/step; LoRA frees optimizer RAM
    'phase2_accum_steps':   2,   # effective batch = 32×2 = 64 (was 32)
    # Limit training samples used in Phase 2 to keep epoch time practical.
    # Measured speed: ~7.5 sec/step (batch=32, gradient checkpointing).
    # 30k samples → 938 steps → ~2h/epoch (practical for 20-epoch training).
    # Increase once you know the full-dataset step time from the progress bar.
    'phase2_max_samples':       30_000,
    # Cap val and test for Phase 2 too — uncapped val (~650K samples) ran BERT
    # inline ~22× longer than training, silently consuming most of each epoch.
    # 30K samples × ~1 s/step at batch=32 ≈ 5 minutes per val pass.
    'phase2_val_max_samples':   30_000,
    'phase2_test_max_samples':  30_000,
    # CE + label_smoothing matches the dl/ pipeline's hard-won lesson.  Class
    # weights stay OFF: with a 10-class label distribution and ~30× imbalance,
    # inverse-frequency weighting collapses the model onto the rarest class
    # (val_acc → 1 %, T → 127 in dl/'s prior calibration).  Label smoothing
    # alone is the right amount of regularisation.
    'loss_type':         'ce',
    'label_smoothing':   0.1,
    'use_class_weights': False,
    'use_amp':           True,

    # Phase 1 uses MultimodalChunkedLoader (background thread, no workers).
    # Val/test use DataLoader with num_workers=0 (sequential, no multiprocessing).
    # Phase 2 uses num_workers=0 due to large in-memory token_cache.

    # Phase 1: frozen BERT
    'phase1_epochs':   10,
    'phase1_lr':       5e-4,
    'phase1_patience': 5,

    # Phase 2: full fine-tune (cache rebuild → train)
    'phase2_epochs':   20,
    'phase2_lr':       2e-5,
    'phase2_bert_lr':  2e-5,
    'phase2_patience': 10,

    # Split ratios — TIME-BASED (not random) to prevent leakage
    'train_ratio': 0.70,
    'val_ratio':   0.15,
    'test_ratio':  0.15,

    'random_seed': 42,
}


def get_multimodal_config(**overrides) -> dict:
    """Return multimodal config with optional key-value overrides."""
    cfg = MULTIMODAL_CONFIG.copy()
    cfg.update(overrides)
    return cfg
