"""
multimodal — Multimodal Transformer for Chinese stock movement prediction.

Fuses 30-day price/technical sequences with MacBERT-encoded Sina financial
news via gated cross-attention (MSGCA 2024), predicting 10-class gain buckets
(same as dl/CHANGE_BUCKETS) within a 5-day forward window.

Quick start:
    from multimodal.main import main
    main()   # or: python -m multimodal.main --help

Architecture references:
    - MSGCA (2024): https://arxiv.org/html/2406.06594v1
    - DASF-Net (2024): https://www.mdpi.com/1911-8074/18/8/417
    - MASTER (AAAI 2024): https://arxiv.org/abs/2312.15235
    - MacBERT: https://github.com/ymcui/Chinese-BERT-wwm
"""

from .config import get_multimodal_config, MM_NUM_CLASSES, MM_CLASS_NAMES
from .models import MultimodalStockTransformer, GatedCrossAttention, create_multimodal_model
from .text_encoder import (
    MacBERTEncoder,
    build_daily_news_cache, load_daily_news_cache,
    build_daily_token_cache, load_daily_token_cache,
)
from .dataset import (
    MultimodalStockDataset,
    MultimodalChunkedLoader,
    Phase2Dataset,
    create_val_test_dataloaders,
    create_phase2_dataloaders,
)
from .data_pipeline import build_predict_sequences
from .training import train_phase1, train_phase2, evaluate_multimodal, save_checkpoint, load_checkpoint

__all__ = [
    # Config
    'get_multimodal_config',
    'MM_NUM_CLASSES',
    'MM_CLASS_NAMES',
    # Models
    'MultimodalStockTransformer',
    'GatedCrossAttention',
    'create_multimodal_model',
    # Text encoder
    'MacBERTEncoder',
    'build_daily_news_cache', 'load_daily_news_cache',
    'build_daily_token_cache', 'load_daily_token_cache',
    # Data pipeline
    'build_predict_sequences',
    # Dataset
    'MultimodalStockDataset',
    'MultimodalChunkedLoader',
    'Phase2Dataset',
    'create_val_test_dataloaders',
    'create_phase2_dataloaders',
    # Training
    'train_phase1',
    'train_phase2',
    'evaluate_multimodal',
    'save_checkpoint',
    'load_checkpoint',
]
