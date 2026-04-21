"""
Stock Price Prediction using Transformer Neural Networks.

This package provides modular components for:
- Data processing with multiprocessing support
- Transformer-based classification models
- Various loss functions for handling class imbalance
- Training and evaluation utilities
- Visualization tools
- Prediction functions

Usage:
    # Run training with default settings (100 stocks for testing)
    python -m dl.main

    # Run on full dataset
    python -m dl.main --max_stocks 0

    # Custom configuration
    python -m dl.main --max_stocks 500 --epochs 100 --loss_type cb
"""

from .config import get_config, get_class_names, NUM_CLASSES, FEATURE_COLUMNS
from .data_processing import (
    load_sector_data,
    load_stock_data,
    prepare_dataset,
    prepare_dataset_to_disk,
    normalize_data,
    split_data,
    calculate_technical_features
)
from .models import TransformerClassifier, StockDataset, create_model
from .losses import FocalLoss, ClassBalancedLoss, create_loss_function
from .training import train_model, evaluate, set_seed, compute_metrics
from .plotting import plot_all_results
from .predict import predict_specific_stocks
from .memmap_dataset import MemmapDataset, MemmapDataWriter, load_memmap_datasets, cache_exists, get_cache_info

__all__ = [
    # Config
    'get_config',
    'get_class_names',
    'NUM_CLASSES',
    'FEATURE_COLUMNS',
    # Data
    'load_sector_data',
    'load_stock_data',
    'prepare_dataset',
    'prepare_dataset_to_disk',
    'normalize_data',
    'split_data',
    'calculate_technical_features',
    # Memory-efficient dataset
    'MemmapDataset',
    'MemmapDataWriter',
    'load_memmap_datasets',
    # Models
    'TransformerClassifier',
    'StockDataset',
    'create_model',
    # Losses
    'FocalLoss',
    'ClassBalancedLoss',
    'create_loss_function',
    # Training
    'train_model',
    'evaluate',
    'set_seed',
    'compute_metrics',
    # Plotting
    'plot_all_results',
    # Prediction
    'predict_specific_stocks',
]
