"""
Multi-Class Stock Price Change Classification using Transformer

This script uses a Transformer encoder to predict the percentage range of
stock price change for the next day based on historical data.

Input: Historical OHLCV data up to time t-1
Output: Class label representing percentage change range (e.g., -5% to -3%, 0% to 1%, etc.)
"""

import os
import sys
import math
import json
import gc
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import seaborn as sns

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Percentage change buckets for multi-class classification
# Each tuple is (min_pct, max_pct, label_name)
CHANGE_BUCKETS = [
    (-float('inf'), -10, '< -10%'),
    (-10, -5, '-10% to -5%'),
    (-5, -3, '-5% to -3%'),
    (-3, -2, '-3% to -2%'),
    (-2, -1, '-2% to -1%'),
    (-1, 0, '-1% to 0%'),
    (0, 1, '0% to 1%'),
    (1, 2, '1% to 2%'),
    (2, 3, '2% to 3%'),
    (3, 5, '3% to 5%'),
    (5, 10, '5% to 10%'),
    (10, float('inf'), '> 10%'),
]

NUM_CLASSES = len(CHANGE_BUCKETS)

# Configuration
CONFIG = {
    'data_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data'),
    'sector_file': 'stock_sectors.csv',
    'sequence_length': 30,  # Increased for more context
    'batch_size': 128,
    'epochs': 50,  # More epochs for multi-class
    'learning_rate': 5e-5,  # Lower learning rate for stability
    'd_model': 64,  # Larger model for more classes
    'nhead': 4,  # Number of attention heads
    'num_layers': 3,  # More layers for complexity
    'dim_feedforward': 128,  # Larger feedforward
    'dropout': 0.15,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'min_data_points': 100,  # Minimum number of data points per stock
    'max_stocks': 3000,  # More stocks for better generalization
    'max_sequences_per_stock': 600,  # More sequences per stock
    # Loss function settings for handling class imbalance
    'loss_type': 'ce',  # Options: 'ce' (CrossEntropy), 'focal' (Focal Loss), 'cb' (Class-Balanced)
    'use_class_weights': True,  # Apply class weights to loss
    'use_weighted_sampling': False,  # Disabled - can cause extreme predictions
    'focal_gamma': 2.0,  # Focal loss focusing parameter (higher = more focus on hard examples)
    'focal_alpha': None,  # Focal loss class weights (None = auto-compute from class distribution)
    'label_smoothing': 0.0,  # Label smoothing factor (0 = no smoothing)
    'cb_beta': 0.9999,  # Class-balanced loss beta parameter
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'random_seed': 42,
}

# Tushare API token
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'


# ============================================================================
# Loss Functions for Handling Class Imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-class classification.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The focal loss reduces the contribution of easy examples (high p_t) and
    focuses training on hard examples (low p_t). This is especially useful
    when dealing with highly imbalanced datasets.

    Args:
        alpha: Class weights. Can be:
            - None: No class weighting
            - float: Same weight for all classes
            - Tensor: Per-class weights
        gamma: Focusing parameter (default=2.0). Higher values focus more on hard examples.
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Label smoothing factor (default=0.0)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits) of shape (N, C) where C = number of classes
            targets: Ground truth labels of shape (N,)

        Returns:
            Focal loss value
        """
        num_classes = inputs.size(1)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Convert targets to one-hot
            targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
            # Apply smoothing
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        else:
            targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)

        # Compute softmax probabilities
        p = torch.softmax(inputs, dim=1)

        # Compute cross entropy component
        ce = -targets_one_hot * torch.log(p + 1e-8)

        # Compute focal weight: (1 - p_t)^gamma
        p_t = (p * targets_one_hot).sum(dim=1, keepdim=True)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce

        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.unsqueeze(0).expand_as(inputs)
            focal_loss = alpha_t * focal_loss

        # Sum over classes
        focal_loss = focal_loss.sum(dim=1)

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.

    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)
    https://arxiv.org/abs/1901.05555

    The effective number of samples is defined as:
    E_n = (1 - beta^n) / (1 - beta)

    where n is the number of samples and beta is a hyperparameter.
    As beta -> 1, E_n -> n (no re-weighting)
    As beta -> 0, E_n -> 1 (all classes weighted equally)

    Args:
        samples_per_class: Number of samples in each class
        beta: Hyperparameter for computing effective number (default=0.9999)
        gamma: Focal loss gamma parameter (default=0.0, no focal)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        samples_per_class: np.ndarray,
        beta: float = 0.9999,
        gamma: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

        # Compute effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        effective_num = np.where(effective_num == 0, 1e-8, effective_num)

        # Compute weights
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)  # Normalize

        self.weights = torch.FloatTensor(weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Class-balanced loss value
        """
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)

        # Compute softmax probabilities
        p = torch.softmax(inputs, dim=1)

        # Get weights for each sample based on its target class
        weights_for_samples = self.weights[targets]

        # Compute cross entropy
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        # Apply focal component if gamma > 0
        if self.gamma > 0:
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.gamma
            ce_loss = focal_weight * ce_loss

        # Apply class-balanced weights
        cb_loss = weights_for_samples * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return cb_loss.mean()
        elif self.reduction == 'sum':
            return cb_loss.sum()
        else:
            return cb_loss


def create_loss_function(
    loss_type: str,
    num_classes: int,
    class_counts: np.ndarray,
    device: str,
    gamma: float = 2.0,
    beta: float = 0.9999,
    label_smoothing: float = 0.0,
    use_class_weights: bool = True
) -> nn.Module:
    """
    Factory function to create the appropriate loss function.

    Args:
        loss_type: 'ce' (CrossEntropy), 'focal' (Focal Loss), or 'cb' (Class-Balanced)
        num_classes: Number of classes
        class_counts: Array of sample counts per class
        device: Device to use
        gamma: Focal loss gamma parameter
        beta: Class-balanced loss beta parameter
        label_smoothing: Label smoothing factor
        use_class_weights: Whether to use class weights

    Returns:
        Loss function module
    """
    # Compute class weights (inverse frequency)
    if use_class_weights:
        weights = 1.0 / (class_counts + 1)
        weights = weights / weights.sum() * num_classes
        class_weights = torch.FloatTensor(weights).to(device)
    else:
        class_weights = None

    if loss_type == 'focal':
        print(f"Using Focal Loss (gamma={gamma}, label_smoothing={label_smoothing})")
        return FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing
        )

    elif loss_type == 'cb':
        print(f"Using Class-Balanced Loss (beta={beta}, gamma={gamma})")
        return ClassBalancedLoss(
            samples_per_class=class_counts,
            beta=beta,
            gamma=gamma
        )

    else:  # Default: CrossEntropy
        print(f"Using CrossEntropy Loss (label_smoothing={label_smoothing})")
        if label_smoothing > 0:
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
        else:
            return nn.CrossEntropyLoss(weight=class_weights)


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for balanced batch sampling.

    This ensures that each batch has roughly equal representation from all classes,
    which helps the model learn minority classes better.

    Args:
        labels: Array of class labels

    Returns:
        WeightedRandomSampler instance
    """
    class_counts = np.bincount(labels)
    # Weight for each class = 1 / count
    class_weights = 1.0 / class_counts
    # Weight for each sample = weight of its class
    sample_weights = class_weights[labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True  # Allow oversampling of minority classes
    )

    return sampler


# Chinese holidays (approximate dates - some are lunar calendar based)
# Format: (month, day, name, duration_days)
CHINESE_HOLIDAYS = {
    # Fixed holidays
    'new_year': [(1, 1, 'New Year', 1)],
    'labor_day': [(5, 1, 'Labor Day', 5)],
    'national_day': [(10, 1, 'National Day', 7)],

    # Variable holidays (approximate - actual dates vary by year)
    # Spring Festival: Usually late Jan/early Feb, ~7 days
    'spring_festival': [(1, 25, 'Spring Festival', 7), (2, 10, 'Spring Festival', 7)],
    # Qingming: Usually Apr 4-6
    'qingming': [(4, 4, 'Qingming', 3)],
    # Dragon Boat: Usually June
    'dragon_boat': [(6, 12, 'Dragon Boat', 3)],
    # Mid-Autumn: Usually Sept/Oct
    'mid_autumn': [(9, 15, 'Mid-Autumn', 3)],
}


def get_chinese_holidays_for_year(year: int) -> List[Tuple[datetime, str, int]]:
    """
    Get Chinese holiday dates for a given year.
    Includes major traditional festivals and public holidays.

    Returns list of (date, holiday_name, duration) tuples.
    """
    holidays = []

    # ============ Fixed Public Holidays ============
    # 元旦 New Year's Day
    holidays.append((datetime(year, 1, 1), 'New Year', 1))
    # 劳动节 Labor Day
    holidays.append((datetime(year, 5, 1), 'Labor Day', 5))
    # 国庆节 National Day
    holidays.append((datetime(year, 10, 1), 'National Day', 7))

    # ============ Lunar Calendar Festivals (dates vary by year) ============

    # 春节 Spring Festival (Chinese New Year) - 7 days
    spring_festival_dates = {
        2017: (1, 27), 2018: (2, 15), 2019: (2, 4), 2020: (1, 24),
        2021: (2, 11), 2022: (1, 31), 2023: (1, 21), 2024: (2, 9),
        2025: (1, 28), 2026: (2, 16), 2027: (2, 5),
    }
    if year in spring_festival_dates:
        m, d = spring_festival_dates[year]
        holidays.append((datetime(year, m, d), 'Spring Festival', 7))

        # 元宵节 Lantern Festival - 15 days after Spring Festival
        lantern_date = datetime(year, m, d) + timedelta(days=15)
        holidays.append((lantern_date, 'Lantern Festival', 1))

    # 清明节 Qingming Festival (Tomb Sweeping Day) - 3 days
    qingming_dates = {
        2017: (4, 2), 2018: (4, 5), 2019: (4, 5), 2020: (4, 4),
        2021: (4, 3), 2022: (4, 3), 2023: (4, 5), 2024: (4, 4),
        2025: (4, 4), 2026: (4, 5), 2027: (4, 5),
    }
    if year in qingming_dates:
        m, d = qingming_dates[year]
        holidays.append((datetime(year, m, d), 'Qingming', 3))
    else:
        holidays.append((datetime(year, 4, 4), 'Qingming', 3))

    # 端午节 Dragon Boat Festival (Duanwu) - 3 days
    dragon_boat_dates = {
        2017: (5, 28), 2018: (6, 16), 2019: (6, 7), 2020: (6, 25),
        2021: (6, 12), 2022: (6, 3), 2023: (6, 22), 2024: (6, 8),
        2025: (5, 31), 2026: (6, 19), 2027: (6, 9),
    }
    if year in dragon_boat_dates:
        m, d = dragon_boat_dates[year]
        holidays.append((datetime(year, m, d), 'Dragon Boat', 3))

    # 中秋节 Mid-Autumn Festival (Moon Festival) - 3 days
    mid_autumn_dates = {
        2017: (10, 4), 2018: (9, 22), 2019: (9, 13), 2020: (10, 1),
        2021: (9, 19), 2022: (9, 10), 2023: (9, 29), 2024: (9, 15),
        2025: (10, 6), 2026: (9, 25), 2027: (9, 15),
    }
    if year in mid_autumn_dates:
        m, d = mid_autumn_dates[year]
        holidays.append((datetime(year, m, d), 'Mid-Autumn', 3))

    # 重阳节 Double Ninth Festival (Chongyang) - not a public holiday but culturally significant
    double_ninth_dates = {
        2017: (10, 28), 2018: (10, 17), 2019: (10, 7), 2020: (10, 25),
        2021: (10, 14), 2022: (10, 4), 2023: (10, 23), 2024: (10, 11),
        2025: (10, 29), 2026: (10, 18), 2027: (10, 8),
    }
    if year in double_ninth_dates:
        m, d = double_ninth_dates[year]
        holidays.append((datetime(year, m, d), 'Double Ninth', 1))

    # 冬至 Winter Solstice - significant for markets
    winter_solstice_dates = {
        2017: (12, 22), 2018: (12, 22), 2019: (12, 22), 2020: (12, 21),
        2021: (12, 21), 2022: (12, 22), 2023: (12, 22), 2024: (12, 21),
        2025: (12, 21), 2026: (12, 22), 2027: (12, 22),
    }
    if year in winter_solstice_dates:
        m, d = winter_solstice_dates[year]
        holidays.append((datetime(year, m, d), 'Winter Solstice', 1))

    # 七夕节 Qixi Festival (Chinese Valentine's Day)
    qixi_dates = {
        2017: (8, 28), 2018: (8, 17), 2019: (8, 7), 2020: (8, 25),
        2021: (8, 14), 2022: (8, 4), 2023: (8, 22), 2024: (8, 10),
        2025: (8, 29), 2026: (8, 19), 2027: (8, 8),
    }
    if year in qixi_dates:
        m, d = qixi_dates[year]
        holidays.append((datetime(year, m, d), 'Qixi', 1))

    # 腊八节 Laba Festival - 8th day of 12th lunar month
    laba_dates = {
        2017: (1, 5), 2018: (1, 24), 2019: (1, 13), 2020: (1, 2),
        2021: (1, 20), 2022: (1, 10), 2023: (12, 30), 2024: (1, 18),
        2025: (1, 7), 2026: (1, 26), 2027: (1, 15),
    }
    if year in laba_dates:
        m, d = laba_dates[year]
        try:
            holidays.append((datetime(year, m, d), 'Laba Festival', 1))
        except:
            pass

    return holidays


def calculate_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cyclical date/time features and holiday indicators.

    Args:
        df: DataFrame with 'trade_date' column

    Returns:
        DataFrame with additional date features
    """
    df = df.copy()

    # Ensure trade_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

    # Extract basic date components
    df['year'] = df['trade_date'].dt.year
    df['month'] = df['trade_date'].dt.month
    df['day'] = df['trade_date'].dt.day
    df['day_of_week'] = df['trade_date'].dt.dayofweek  # 0=Monday, 4=Friday
    df['day_of_year'] = df['trade_date'].dt.dayofyear
    df['week_of_year'] = df['trade_date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['trade_date'].dt.quarter

    # Cyclical encoding using sine/cosine
    # Day of week (5 trading days: 0-4)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

    # Day of month (1-31)
    df['dom_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['dom_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    # Month of year (1-12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Week of year (1-52)
    df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Day of year (1-365)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Quarter (1-4)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    # Trading day indicators
    df['is_monday'] = (df['day_of_week'] == 0).astype(float)
    df['is_friday'] = (df['day_of_week'] == 4).astype(float)

    # First/last trading day of month (approximate)
    df['is_month_start'] = (df['day'] <= 3).astype(float)
    df['is_month_end'] = (df['day'] >= 28).astype(float)

    # First/last trading day of year
    df['is_year_start'] = ((df['month'] == 1) & (df['day'] <= 5)).astype(float)
    df['is_year_end'] = ((df['month'] == 12) & (df['day'] >= 25)).astype(float)

    # Holiday features
    df['is_pre_holiday'] = 0.0
    df['is_post_holiday'] = 0.0
    df['days_to_holiday'] = 30.0  # Default: far from holiday
    df['days_from_holiday'] = 30.0
    df['holiday_effect'] = 0.0

    # Get unique years in data
    years = df['year'].unique()

    for year in years:
        holidays = get_chinese_holidays_for_year(year)

        for holiday_date, holiday_name, duration in holidays:
            # Create date range for holiday
            holiday_start = holiday_date
            holiday_end = holiday_date + timedelta(days=duration)

            # Pre-holiday effect (5 trading days before)
            pre_holiday_start = holiday_start - timedelta(days=7)
            mask_pre = (df['trade_date'] >= pre_holiday_start) & (df['trade_date'] < holiday_start)
            df.loc[mask_pre, 'is_pre_holiday'] = 1.0

            # Post-holiday effect (3 trading days after)
            post_holiday_end = holiday_end + timedelta(days=5)
            mask_post = (df['trade_date'] > holiday_end) & (df['trade_date'] <= post_holiday_end)
            df.loc[mask_post, 'is_post_holiday'] = 1.0

            # Days to/from holiday
            for idx in df.index:
                trade_date = df.loc[idx, 'trade_date']
                if isinstance(trade_date, pd.Timestamp):
                    trade_date = trade_date.to_pydatetime()

                days_to = (holiday_start - trade_date).days
                days_from = (trade_date - holiday_end).days

                if 0 < days_to < df.loc[idx, 'days_to_holiday']:
                    df.loc[idx, 'days_to_holiday'] = days_to
                if 0 < days_from < df.loc[idx, 'days_from_holiday']:
                    df.loc[idx, 'days_from_holiday'] = days_from

    # Normalize days to/from holiday (closer to holiday = higher value)
    df['days_to_holiday_norm'] = 1.0 / (df['days_to_holiday'] + 1)
    df['days_from_holiday_norm'] = 1.0 / (df['days_from_holiday'] + 1)

    # Combined holiday effect (pre-holiday often bullish, post-holiday mixed)
    df['holiday_effect'] = df['is_pre_holiday'] * 0.5 + df['days_to_holiday_norm'] * 0.3 - df['is_post_holiday'] * 0.2

    # Special period indicators
    # January effect (first month often has different patterns)
    df['is_january'] = (df['month'] == 1).astype(float)

    # Year-end effect (December)
    df['is_december'] = (df['month'] == 12).astype(float)

    # Earnings season (approx: Jan, Apr, Jul, Oct for quarterly reports)
    df['is_earnings_season'] = df['month'].isin([1, 4, 7, 10]).astype(float)

    # "Sell in May" effect indicator (May-October historically weaker)
    df['is_weak_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(float)

    # Drop intermediate columns
    df = df.drop(columns=['year', 'month', 'day', 'day_of_week', 'day_of_year',
                          'week_of_year', 'quarter', 'days_to_holiday', 'days_from_holiday'])

    return df


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_sector_data(save_path: str) -> pd.DataFrame:
    """
    Download stock sector/industry data from Tushare.

    Returns:
        DataFrame with columns: ts_code, industry, sector
    """
    try:
        import tushare as ts

        print("Downloading sector data from Tushare...")
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()

        # Get stock basic info which includes industry
        stock_basic = pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,market,list_date'
        )

        # Create sector mapping (simplified - group industries into broader sectors)
        sector_mapping = {
            # Technology
            '软件服务': 'Technology', '互联网': 'Technology', '计算机设备': 'Technology',
            '通信设备': 'Technology', '电子元件': 'Technology', '半导体': 'Technology',
            '消费电子': 'Technology', '光学光电': 'Technology', '通信服务': 'Technology',

            # Finance
            '银行': 'Finance', '证券': 'Finance', '保险': 'Finance', '多元金融': 'Finance',
            '信托': 'Finance',

            # Healthcare
            '医疗服务': 'Healthcare', '医药商业': 'Healthcare', '化学制药': 'Healthcare',
            '生物制品': 'Healthcare', '中药': 'Healthcare', '医疗器械': 'Healthcare',

            # Consumer
            '食品饮料': 'Consumer', '酿酒': 'Consumer', '家电': 'Consumer',
            '服装纺织': 'Consumer', '零售': 'Consumer', '商业百货': 'Consumer',
            '旅游酒店': 'Consumer', '餐饮': 'Consumer', '教育': 'Consumer',

            # Industrial
            '机械设备': 'Industrial', '电气设备': 'Industrial', '汽车': 'Industrial',
            '航空航天': 'Industrial', '船舶': 'Industrial', '铁路': 'Industrial',
            '工程机械': 'Industrial', '专用设备': 'Industrial', '仪器仪表': 'Industrial',

            # Materials
            '钢铁': 'Materials', '有色金属': 'Materials', '化工': 'Materials',
            '建材': 'Materials', '造纸': 'Materials', '塑料': 'Materials',
            '橡胶': 'Materials', '玻璃': 'Materials',

            # Energy
            '石油': 'Energy', '煤炭': 'Energy', '电力': 'Energy', '燃气': 'Energy',
            '新能源': 'Energy', '光伏': 'Energy', '风电': 'Energy',

            # Real Estate
            '房地产': 'Real Estate', '物业管理': 'Real Estate', '房产服务': 'Real Estate',

            # Utilities
            '水务': 'Utilities', '环保': 'Utilities', '公用事业': 'Utilities',

            # Agriculture
            '农业': 'Agriculture', '林业': 'Agriculture', '畜牧': 'Agriculture',
            '渔业': 'Agriculture', '农产品': 'Agriculture',
        }

        # Map industries to sectors
        stock_basic['sector'] = stock_basic['industry'].map(sector_mapping).fillna('Other')

        # Save to file
        sector_data = stock_basic[['ts_code', 'industry', 'sector']].copy()
        sector_data.to_csv(save_path, index=False)
        print(f"Sector data saved to {save_path}")
        print(f"Total stocks with sector info: {len(sector_data)}")
        print(f"Sector distribution:\n{sector_data['sector'].value_counts()}")

        return sector_data

    except Exception as e:
        print(f"Error downloading sector data: {e}")
        print("Creating empty sector DataFrame...")
        return pd.DataFrame(columns=['ts_code', 'industry', 'sector'])


def load_sector_data(data_dir: str) -> pd.DataFrame:
    """Load or download sector data."""
    sector_path = os.path.join(data_dir, CONFIG['sector_file'])

    if os.path.exists(sector_path):
        print(f"Loading existing sector data from {sector_path}")
        return pd.read_csv(sector_path)
    else:
        return download_sector_data(sector_path)


def detect_w_bottom(prices: np.ndarray, window: int = 20, tolerance: float = 0.03) -> np.ndarray:
    """
    Detect W bottom (double bottom) pattern.

    A W bottom consists of:
    1. First low (left bottom of W)
    2. A peak in the middle (neckline)
    3. Second low at similar level (right bottom of W)
    4. Price breaking above the neckline

    Args:
        prices: Array of closing prices
        window: Lookback window for pattern detection
        tolerance: Price tolerance for matching bottoms (as fraction)

    Returns:
        Array of pattern strength (0-1, higher = stronger pattern)
    """
    n = len(prices)
    pattern_strength = np.zeros(n)

    for i in range(window, n):
        segment = prices[i - window:i + 1]

        # Find local minima and maxima in the segment
        local_mins = []
        local_maxs = []

        for j in range(2, len(segment) - 2):
            # Local minimum
            if segment[j] < segment[j-1] and segment[j] < segment[j-2] and \
               segment[j] < segment[j+1] and segment[j] < segment[j+2]:
                local_mins.append((j, segment[j]))
            # Local maximum
            if segment[j] > segment[j-1] and segment[j] > segment[j-2] and \
               segment[j] > segment[j+1] and segment[j] > segment[j+2]:
                local_maxs.append((j, segment[j]))

        # Need at least 2 minima and 1 maximum for W pattern
        if len(local_mins) >= 2 and len(local_maxs) >= 1:
            # Get the two most recent minima
            sorted_mins = sorted(local_mins, key=lambda x: x[0])[-2:]
            first_min_idx, first_min_val = sorted_mins[0]
            second_min_idx, second_min_val = sorted_mins[1]

            # Find the maximum between them
            max_between = None
            for max_idx, max_val in local_maxs:
                if first_min_idx < max_idx < second_min_idx:
                    if max_between is None or max_val > max_between[1]:
                        max_between = (max_idx, max_val)

            if max_between is not None:
                # Check if bottoms are at similar levels
                bottom_diff = abs(first_min_val - second_min_val) / min(first_min_val, second_min_val)

                if bottom_diff <= tolerance:
                    # Calculate pattern strength
                    neckline = max_between[1]
                    avg_bottom = (first_min_val + second_min_val) / 2
                    current_price = segment[-1]

                    # Strength based on: similar bottoms, clear neckline, current price above neckline
                    depth = (neckline - avg_bottom) / avg_bottom  # Pattern depth
                    breakout = (current_price - neckline) / neckline if current_price > neckline else 0

                    strength = min(1.0, (1 - bottom_diff) * 0.3 + min(depth, 0.1) * 3 + min(breakout, 0.05) * 8)
                    pattern_strength[i] = max(0, strength)

    return pattern_strength


def detect_m_top(prices: np.ndarray, window: int = 20, tolerance: float = 0.03) -> np.ndarray:
    """
    Detect M top (double top) pattern.

    An M top consists of:
    1. First high (left peak of M)
    2. A trough in the middle (neckline)
    3. Second high at similar level (right peak of M)
    4. Price breaking below the neckline

    Args:
        prices: Array of closing prices
        window: Lookback window for pattern detection
        tolerance: Price tolerance for matching tops (as fraction)

    Returns:
        Array of pattern strength (0-1, higher = stronger pattern)
    """
    n = len(prices)
    pattern_strength = np.zeros(n)

    for i in range(window, n):
        segment = prices[i - window:i + 1]

        # Find local minima and maxima in the segment
        local_mins = []
        local_maxs = []

        for j in range(2, len(segment) - 2):
            # Local minimum
            if segment[j] < segment[j-1] and segment[j] < segment[j-2] and \
               segment[j] < segment[j+1] and segment[j] < segment[j+2]:
                local_mins.append((j, segment[j]))
            # Local maximum
            if segment[j] > segment[j-1] and segment[j] > segment[j-2] and \
               segment[j] > segment[j+1] and segment[j] > segment[j+2]:
                local_maxs.append((j, segment[j]))

        # Need at least 2 maxima and 1 minimum for M pattern
        if len(local_maxs) >= 2 and len(local_mins) >= 1:
            # Get the two most recent maxima
            sorted_maxs = sorted(local_maxs, key=lambda x: x[0])[-2:]
            first_max_idx, first_max_val = sorted_maxs[0]
            second_max_idx, second_max_val = sorted_maxs[1]

            # Find the minimum between them
            min_between = None
            for min_idx, min_val in local_mins:
                if first_max_idx < min_idx < second_max_idx:
                    if min_between is None or min_val < min_between[1]:
                        min_between = (min_idx, min_val)

            if min_between is not None:
                # Check if tops are at similar levels
                top_diff = abs(first_max_val - second_max_val) / min(first_max_val, second_max_val)

                if top_diff <= tolerance:
                    # Calculate pattern strength
                    neckline = min_between[1]
                    avg_top = (first_max_val + second_max_val) / 2
                    current_price = segment[-1]

                    # Strength based on: similar tops, clear neckline, current price below neckline
                    depth = (avg_top - neckline) / neckline  # Pattern depth
                    breakdown = (neckline - current_price) / neckline if current_price < neckline else 0

                    strength = min(1.0, (1 - top_diff) * 0.3 + min(depth, 0.1) * 3 + min(breakdown, 0.05) * 8)
                    pattern_strength[i] = max(0, strength)

    return pattern_strength


def detect_patterns_multi_window(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect W bottom and M top patterns across multiple time windows.

    Args:
        prices: Array of closing prices

    Returns:
        Tuple of (w_bottom_short, w_bottom_long, m_top_short, m_top_long)
    """
    # Short-term patterns (10-day window)
    w_bottom_short = detect_w_bottom(prices, window=10, tolerance=0.025)
    m_top_short = detect_m_top(prices, window=10, tolerance=0.025)

    # Long-term patterns (20-day window)
    w_bottom_long = detect_w_bottom(prices, window=20, tolerance=0.035)
    m_top_long = detect_m_top(prices, window=20, tolerance=0.035)

    return w_bottom_short, w_bottom_long, m_top_short, m_top_long


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators as features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional technical features
    """
    df = df.copy()

    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']

    # Moving averages
    for window in [5, 10, 20]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
        df[f'vol_sma_{window}'] = df['vol'].rolling(window=window).mean()
        df[f'vol_sma_{window}_ratio'] = df['vol'] / df[f'vol_sma_{window}']

    # Volatility
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume features
    df['volume_change'] = df['vol'].pct_change()
    df['amount_per_vol'] = df['amount'] / df['vol']

    # Price position in range
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    # ============ NEW FEATURES ============

    # ATR (Average True Range) - measures volatility
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    df['atr_ratio'] = true_range / df['atr_14']  # Current TR vs average

    # OBV (On Balance Volume) - volume flow indicator
    obv = np.where(df['close'] > df['close'].shift(1), df['vol'],
                   np.where(df['close'] < df['close'].shift(1), -df['vol'], 0))
    df['obv'] = np.cumsum(obv)
    df['obv_sma_10'] = df['obv'].rolling(window=10).mean()
    df['obv_ratio'] = df['obv'] / df['obv_sma_10']

    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

    # Williams %R
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

    # CCI (Commodity Channel Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_sma = typical_price.rolling(window=20).mean()
    tp_mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - tp_sma) / (0.015 * tp_mad)

    # Rate of Change (ROC) - momentum indicator
    df['roc_5'] = 100 * (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['roc_10'] = 100 * (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['roc_20'] = 100 * (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

    # Momentum
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)

    # ADX (Average Directional Index) - trend strength
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff().abs() * -1
    plus_dm = np.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm.abs() > plus_dm) & (minus_dm < 0), minus_dm.abs(), 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / df['atr_14']
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / df['atr_14']
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(window=14).mean()
    df['di_diff'] = plus_di - minus_di  # Directional indicator difference

    # Moving Average Crossovers
    df['sma_5_10_cross'] = (df['sma_5'] - df['sma_10']) / df['sma_10']  # Short vs medium
    df['sma_10_20_cross'] = (df['sma_10'] - df['sma_20']) / df['sma_20']  # Medium vs long
    df['ema_12_26_cross'] = (ema12 - ema26) / ema26  # MACD basis normalized

    # Lag features (previous returns)
    for lag in [1, 2, 3, 5]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)

    # Price gaps
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_abs'] = abs(df['gap'])

    # Trend strength indicators
    df['above_sma_5'] = (df['close'] > df['sma_5']).astype(float)
    df['above_sma_10'] = (df['close'] > df['sma_10']).astype(float)
    df['above_sma_20'] = (df['close'] > df['sma_20']).astype(float)
    df['trend_score'] = df['above_sma_5'] + df['above_sma_10'] + df['above_sma_20']

    # Candle patterns
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']
    df['is_bullish_candle'] = (df['close'] > df['open']).astype(float)

    # Consecutive up/down days
    df['up_day'] = (df['returns'] > 0).astype(int)
    df['consecutive_up'] = df['up_day'].groupby((df['up_day'] != df['up_day'].shift()).cumsum()).cumsum()
    df['consecutive_down'] = (1 - df['up_day']).groupby(((1 - df['up_day']) != (1 - df['up_day']).shift()).cumsum()).cumsum()

    # Distance from recent high/low
    df['dist_from_high_20'] = (df['close'] - df['high'].rolling(20).max()) / df['high'].rolling(20).max()
    df['dist_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['low'].rolling(20).min()

    # Volume-price relationship
    df['vwap'] = (df['amount']) / df['vol']  # Volume weighted average price proxy
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']

    # ============ W Bottom and M Top pattern detection ============
    prices = df['close'].values
    w_short, w_long, m_short, m_long = detect_patterns_multi_window(prices)

    df['w_bottom_short'] = w_short  # Short-term W bottom (10-day)
    df['w_bottom_long'] = w_long    # Long-term W bottom (20-day)
    df['m_top_short'] = m_short     # Short-term M top (10-day)
    df['m_top_long'] = m_long       # Long-term M top (20-day)

    # Combined pattern signals
    df['w_bottom_signal'] = (df['w_bottom_short'] + df['w_bottom_long']) / 2  # Bullish signal
    df['m_top_signal'] = (df['m_top_short'] + df['m_top_long']) / 2           # Bearish signal
    df['pattern_bias'] = df['w_bottom_signal'] - df['m_top_signal']           # Net bullish/bearish bias

    # ============ Date/Time and Holiday Features ============
    df = calculate_date_features(df)

    return df


def load_stock_data(data_dir: str, market: str = 'sh', max_stocks: int = None) -> Dict[str, pd.DataFrame]:
    """
    Load stock data from CSV files.

    Args:
        data_dir: Directory containing stock data
        market: 'sh' for Shanghai, 'sz' for Shenzhen
        max_stocks: Maximum number of stocks to load (None for all)

    Returns:
        Dictionary mapping stock codes to DataFrames
    """
    market_dir = os.path.join(data_dir, market)
    stocks = {}

    if not os.path.exists(market_dir):
        print(f"Market directory not found: {market_dir}")
        return stocks

    files = [f for f in os.listdir(market_dir) if f.endswith('.csv')]

    # Randomly sample files if max_stocks is specified
    if max_stocks is not None and len(files) > max_stocks:
        np.random.shuffle(files)
        files = files[:max_stocks]

    print(f"Loading up to {len(files)} stocks from {market.upper()} market...")

    for file in files:
        try:
            file_path = os.path.join(market_dir, file)
            df = pd.read_csv(file_path)

            if len(df) < CONFIG['min_data_points']:
                continue

            # Ensure data is sorted by date
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('trade_date').reset_index(drop=True)

            ts_code = file.replace('.csv', '')
            stocks[ts_code] = df

        except Exception as e:
            continue

    print(f"Loaded {len(stocks)} stocks with sufficient data")
    return stocks


def pct_change_to_class(pct_change: float) -> int:
    """
    Convert percentage change to class label based on CHANGE_BUCKETS.

    Args:
        pct_change: Percentage change (e.g., 2.5 for 2.5%)

    Returns:
        Class label (0 to NUM_CLASSES-1)
    """
    for i, (min_pct, max_pct, _) in enumerate(CHANGE_BUCKETS):
        if min_pct <= pct_change < max_pct:
            return i
    # Should not reach here, but return last class if it does
    return NUM_CLASSES - 1


def get_class_names() -> List[str]:
    """Get list of class names for reporting."""
    return [name for _, _, name in CHANGE_BUCKETS]


def prepare_dataset(
    stocks: Dict[str, pd.DataFrame],
    sector_data: pd.DataFrame,
    sequence_length: int = 30,
    max_sequences_per_stock: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare sequences and labels for all stocks.

    Args:
        stocks: Dictionary of stock DataFrames
        sector_data: DataFrame with sector information
        sequence_length: Number of historical days per sequence
        max_sequences_per_stock: Maximum sequences to sample per stock

    Returns:
        Tuple of (sequences, labels, sector_ids)
    """
    all_sequences = []
    all_labels = []
    all_sectors = []

    # Create sector encoding
    if len(sector_data) > 0:
        sector_to_id = {sector: i for i, sector in enumerate(sector_data['sector'].unique())}
    else:
        sector_to_id = {}
    sector_to_id['Unknown'] = len(sector_to_id)

    # Features to use (normalized versions)
    feature_cols = [
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
        # ============ Date/Time Cyclical Features ============
        # Day of week (sine/cosine)
        'dow_sin', 'dow_cos',
        # Day of month (sine/cosine)
        'dom_sin', 'dom_cos',
        # Month of year (sine/cosine)
        'month_sin', 'month_cos',
        # Week of year (sine/cosine)
        'woy_sin', 'woy_cos',
        # Day of year (sine/cosine)
        'doy_sin', 'doy_cos',
        # Quarter (sine/cosine)
        'quarter_sin', 'quarter_cos',
        # Trading day indicators
        'is_monday', 'is_friday',
        'is_month_start', 'is_month_end',
        'is_year_start', 'is_year_end',
        # ============ Holiday Features ============
        'is_pre_holiday', 'is_post_holiday',
        'days_to_holiday_norm', 'days_from_holiday_norm',
        'holiday_effect',
        # Special period indicators
        'is_january', 'is_december',
        'is_earnings_season', 'is_weak_season'
    ]

    print("Preparing sequences...")
    processed = 0

    for ts_code, df in stocks.items():
        # Calculate technical features
        df = calculate_technical_features(df)

        # Get sector for this stock
        sector_info = sector_data[sector_data['ts_code'] == ts_code] if len(sector_data) > 0 else pd.DataFrame()
        if len(sector_info) > 0:
            sector = sector_info['sector'].values[0]
        else:
            sector = 'Unknown'
        sector_id = sector_to_id.get(sector, sector_to_id['Unknown'])

        # Remove rows with NaN values
        df = df.dropna(subset=feature_cols)

        if len(df) < sequence_length + 1:
            continue

        # Extract features
        features = df[feature_cols].values
        closes = df['close'].values

        # Get sequence indices
        valid_indices = list(range(sequence_length, len(df)))

        # Sample if too many sequences
        if max_sequences_per_stock and len(valid_indices) > max_sequences_per_stock:
            valid_indices = list(np.random.choice(valid_indices, max_sequences_per_stock, replace=False))

        # Create sequences
        for i in valid_indices:
            # Input: features from t-sequence_length to t-1
            seq = features[i - sequence_length:i]

            # Calculate percentage change
            pct_change = 100 * (closes[i] - closes[i - 1]) / closes[i - 1]

            # Convert to class label based on buckets
            label = pct_change_to_class(pct_change)

            all_sequences.append(seq)
            all_labels.append(label)
            all_sectors.append(sector_id)

        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed}/{len(stocks)} stocks, {len(all_sequences)} sequences...")

    print(f"Total sequences created: {len(all_sequences)}")

    return np.array(all_sequences, dtype=np.float32), np.array(all_labels), np.array(all_sectors)


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray, sectors: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.sectors = torch.LongTensor(sectors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.sectors[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer-based multi-class classifier for stock price change prediction.

    The model uses:
    1. Input projection layer
    2. Positional encoding
    3. Transformer encoder layers
    4. Sector embedding (optional)
    5. Classification head for multiple percentage change buckets
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_sectors: int = 12,
        use_sector: bool = True
    ):
        super().__init__()

        self.use_sector = use_sector
        self.d_model = d_model
        self.num_classes = num_classes

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Sector embedding
        if use_sector:
            self.sector_embedding = nn.Embedding(num_sectors + 1, d_model // 4)
            classifier_input_dim = d_model + d_model // 4
        else:
            classifier_input_dim = d_model

        # Classification head - larger for multi-class
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)  # Multi-class classification
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, sectors: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequences [batch_size, seq_len, input_dim]
            sectors: Sector IDs [batch_size]

        Returns:
            Logits [batch_size, 2]
        """
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Use the last time step's output for classification
        x = x[:, -1, :]  # [batch_size, d_model]

        # Add sector information
        if self.use_sector and sectors is not None:
            sector_emb = self.sector_embedding(sectors)  # [batch_size, d_model // 4]
            x = torch.cat([x, sector_emb], dim=-1)

        # Classify
        logits = self.classifier(x)

        return logits


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels, sectors in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        sectors = sectors.to(device)

        optimizer.zero_grad()

        logits = model(sequences, sectors)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model.

    Returns:
        Tuple of (avg_loss, predictions, labels, probabilities)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels, sectors in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            sectors = sectors.to(device)

            logits = model(sequences, sectors)
            loss = criterion(logits, labels)

            # Get probabilities using softmax
            probs = torch.softmax(logits, dim=1)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())  # All class probabilities

    avg_loss = total_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)  # Shape: (num_samples, num_classes)

    return avg_loss, all_predictions, all_labels, all_probs


def normalize_data(
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    test_sequences: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Normalize sequences using StandardScaler fitted on training data."""
    # Reshape for scaler
    n_train, seq_len, n_features = train_sequences.shape

    # Fit scaler on training data
    scaler = StandardScaler()
    train_flat = train_sequences.reshape(-1, n_features)
    scaler.fit(train_flat)

    # Transform all sets
    train_normalized = scaler.transform(train_flat).reshape(n_train, seq_len, n_features)
    val_normalized = scaler.transform(val_sequences.reshape(-1, n_features)).reshape(val_sequences.shape)
    test_normalized = scaler.transform(test_sequences.reshape(-1, n_features)).reshape(test_sequences.shape)

    # Handle any remaining NaN or Inf values
    train_normalized = np.nan_to_num(train_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    val_normalized = np.nan_to_num(val_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    test_normalized = np.nan_to_num(test_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return train_normalized, val_normalized, test_normalized, scaler


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs)])

    # Plot Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='Random Baseline', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0.45, 0.65])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str, class_names: List[str] = None):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array
        save_path: Path to save the plot
        class_names: List of class names
    """
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = get_class_names()

    # Normalize confusion matrix for color intensity
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    # Adjust figure size based on number of classes
    fig_size = max(10, n_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    # Create heatmap using seaborn
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})

    # Add count annotations (only for cells with significant values)
    thresh = cm_normalized.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j] > 0:  # Only annotate non-zero cells
                ax.text(j + 0.5, i + 0.5, f'{cm[i, j]}',
                       ha="center", va="center", fontsize=7,
                       color="white" if cm_normalized[i, j] > thresh else "black")

    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title('Confusion Matrix - Price Change Classification', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: str):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities for positive class
        save_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Baseline')
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve plot saved to: {save_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: str):
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities for positive class
        save_path: Path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, color='green', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')
    baseline = np.mean(y_true)
    ax.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.3, color='green')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve plot saved to: {save_path}")


def plot_metrics_summary(metrics: Dict[str, float], save_path: str):
    """
    Plot bar chart of evaluation metrics.

    Args:
        metrics: Dictionary with metric names and values
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(metric_names)))

    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='navy', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=0.5, color='red', linestyle='--', label='Random Baseline (0.5)', alpha=0.7)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics summary plot saved to: {save_path}")


def plot_class_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """
    Plot class distribution comparison between actual and predicted.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    class_names = get_class_names()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual distribution
    ax1 = axes[0]
    unique, counts = np.unique(y_true, return_counts=True)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(class_names)))
    bars1 = ax1.bar(range(len(class_names)), [counts[list(unique).index(i)] if i in unique else 0 for i in range(len(class_names))],
                    color=colors, edgecolor='black')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Count')
    ax1.set_title('Actual Class Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Predicted distribution
    ax2 = axes[1]
    unique, counts = np.unique(y_pred, return_counts=True)
    bars2 = ax2.bar(range(len(class_names)), [counts[list(unique).index(i)] if i in unique else 0 for i in range(len(class_names))],
                    color=colors, edgecolor='black')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Count')
    ax2.set_title('Predicted Class Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to: {save_path}")


def plot_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """
    Plot per-class precision, recall, and F1 scores.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    class_names = get_class_names()

    # Calculate per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Pad if needed
    while len(precision) < len(class_names):
        precision = np.append(precision, 0)
        recall = np.append(recall, 0)
        f1 = np.append(f1, 0)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='darkorange')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='green')

    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to: {save_path}")


def plot_all_results(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_dir: str
):
    """
    Generate all visualization plots for multi-class classification.

    Args:
        history: Training history dictionary
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (num_samples x num_classes)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "-" * 40)
    print("Generating Visualizations")
    print("-" * 40)

    # 1. Training history
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))

    # 2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, os.path.join(save_dir, 'confusion_matrix.png'))

    # 3. Class distribution comparison
    plot_class_distribution(y_true, y_pred, os.path.join(save_dir, 'class_distribution.png'))

    # 4. Per-class metrics
    plot_per_class_metrics(y_true, y_pred, os.path.join(save_dir, 'per_class_metrics.png'))

    # 5. Overall metrics summary
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision\n(weighted)': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall\n(weighted)': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score\n(weighted)': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    plot_metrics_summary(metrics, os.path.join(save_dir, 'metrics_summary.png'))

    # 6. Combined summary figure
    plot_combined_summary_multiclass(history, y_true, y_pred, y_probs, os.path.join(save_dir, 'combined_summary.png'))


def plot_combined_summary_multiclass(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_path: str
):
    """
    Create a combined summary figure for multi-class classification.

    Args:
        history: Training history dictionary
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (num_samples x num_classes)
        save_path: Path to save the plot
    """
    class_names = get_class_names()
    fig = plt.figure(figsize=(18, 12))

    # 1. Training Loss (top-left)
    ax1 = fig.add_subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Training Accuracy (top-middle)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    random_baseline = 1.0 / NUM_CLASSES
    ax2.axhline(y=random_baseline, color='gray', linestyle='--', alpha=0.7, label=f'Random ({random_baseline:.2f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Class Distribution (top-right)
    ax3 = fig.add_subplot(2, 3, 3)
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)

    x = np.arange(len(class_names))
    width = 0.35

    true_counts = [counts_true[list(unique_true).index(i)] if i in unique_true else 0 for i in range(len(class_names))]
    pred_counts = [counts_pred[list(unique_pred).index(i)] if i in unique_pred else 0 for i in range(len(class_names))]

    ax3.bar(x - width/2, true_counts, width, label='Actual', color='steelblue', alpha=0.8)
    ax3.bar(x + width/2, pred_counts, width, label='Predicted', color='darkorange', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('Count')
    ax3.set_title('Class Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Confusion Matrix Heatmap (bottom-left, spanning more space)
    ax4 = fig.add_subplot(2, 3, 4)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=ax4,
                xticklabels=[c[:8] for c in class_names],
                yticklabels=[c[:8] for c in class_names])
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Confusion Matrix', fontweight='bold')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(ax4.get_yticklabels(), rotation=0, fontsize=7)

    # 5. Per-class F1 Scores (bottom-middle)
    ax5 = fig.add_subplot(2, 3, 5)
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    while len(f1_scores) < len(class_names):
        f1_scores = np.append(f1_scores, 0)
    colors = plt.cm.RdYlGn(f1_scores)
    bars = ax5.bar(range(len(class_names)), f1_scores, color=colors, edgecolor='black')
    ax5.set_xticks(range(len(class_names)))
    ax5.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax5.set_ylabel('F1 Score')
    ax5.set_title('Per-Class F1 Scores', fontweight='bold')
    ax5.set_ylim([0, 1.0])
    ax5.axhline(y=f1_score(y_true, y_pred, average='weighted', zero_division=0), color='red', linestyle='--', label='Weighted Avg')
    ax5.legend()
    ax5.grid(True, axis='y', alpha=0.3)

    # 6. Overall Metrics (bottom-right)
    ax6 = fig.add_subplot(2, 3, 6)
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(metrics)))
    bars = ax6.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='navy')
    ax6.axhline(y=random_baseline, color='red', linestyle='--', alpha=0.7, label=f'Random ({random_baseline:.2f})')
    for bar, val in zip(bars, metrics.values()):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax6.set_ylabel('Score')
    ax6.set_title('Overall Metrics (Weighted)', fontweight='bold')
    ax6.set_ylim([0, 1.0])
    ax6.legend()
    ax6.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f'Stock Price Change Prediction ({NUM_CLASSES} Classes) - Training & Evaluation Summary',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined summary plot saved to: {save_path}")


def predict_specific_stocks(
    stock_codes: List[str],
    model_path: str,
    data_dir: str,
    sector_data: pd.DataFrame,
    device: str = 'cpu'
) -> Dict[str, Dict]:
    """
    Make predictions for specific stocks using a trained model.

    Args:
        stock_codes: List of stock codes to predict (e.g., ['001270', '300788'])
        model_path: Path to the saved model checkpoint
        data_dir: Directory containing stock data
        sector_data: DataFrame with sector information
        device: Device to run predictions on

    Returns:
        Dictionary mapping stock codes to prediction results
    """
    print("\n" + "=" * 60)
    print("Predicting for Specific Stocks")
    print("=" * 60)

    # Load model checkpoint
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return {}

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_config = checkpoint['config']

    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']

    # Determine input dimensions
    input_dim = len(scaler.mean_)

    # Get number of sectors
    num_sectors = len(sector_data['sector'].unique()) if len(sector_data) > 0 else 0

    # Create sector encoding
    if len(sector_data) > 0:
        sector_to_id = {sector: i for i, sector in enumerate(sector_data['sector'].unique())}
    else:
        sector_to_id = {}
    sector_to_id['Unknown'] = len(sector_to_id)

    # Create model and load weights
    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        d_model=saved_config['d_model'],
        nhead=saved_config['nhead'],
        num_layers=saved_config['num_layers'],
        dim_feedforward=saved_config['dim_feedforward'],
        dropout=saved_config['dropout'],
        num_sectors=num_sectors,
        use_sector=(num_sectors > 0)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Feature columns (must match training)
    feature_cols = [
        'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
        'sma_5_ratio', 'sma_10_ratio', 'sma_20_ratio',
        'vol_sma_5_ratio', 'vol_sma_10_ratio', 'vol_sma_20_ratio',
        'volatility_5', 'volatility_10', 'volatility_20',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_position', 'volume_change', 'price_position',
        'atr_ratio', 'obv_ratio',
        'stoch_k', 'stoch_d', 'stoch_diff', 'williams_r', 'cci',
        'roc_5', 'roc_10', 'roc_20', 'momentum_5', 'momentum_10',
        'plus_di', 'minus_di', 'adx', 'di_diff',
        'sma_5_10_cross', 'sma_10_20_cross', 'ema_12_26_cross',
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
        'gap', 'gap_abs',
        'above_sma_5', 'above_sma_10', 'above_sma_20', 'trend_score',
        'body_size', 'upper_shadow', 'lower_shadow', 'is_bullish_candle',
        'consecutive_up', 'consecutive_down',
        'dist_from_high_20', 'dist_from_low_20', 'price_vs_vwap',
        'w_bottom_short', 'w_bottom_long', 'm_top_short', 'm_top_long',
        'w_bottom_signal', 'm_top_signal', 'pattern_bias',
        'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos',
        'month_sin', 'month_cos', 'woy_sin', 'woy_cos',
        'doy_sin', 'doy_cos', 'quarter_sin', 'quarter_cos',
        'is_monday', 'is_friday', 'is_month_start', 'is_month_end',
        'is_year_start', 'is_year_end',
        'is_pre_holiday', 'is_post_holiday',
        'days_to_holiday_norm', 'days_from_holiday_norm', 'holiday_effect',
        'is_january', 'is_december', 'is_earnings_season', 'is_weak_season'
    ]

    results = {}
    class_names = get_class_names()
    sequence_length = saved_config['sequence_length']

    for stock_code in stock_codes:
        print(f"\n{'-' * 50}")
        print(f"Stock: {stock_code}")
        print(f"{'-' * 50}")

        # Try to find stock file in both markets
        stock_path = None
        for market in ['sz', 'sh']:
            possible_path = os.path.join(data_dir, market, f"{stock_code}.csv")
            if os.path.exists(possible_path):
                stock_path = possible_path
                break

        if stock_path is None:
            print(f"  Stock data not found for {stock_code}")
            results[stock_code] = {'error': 'Stock data not found'}
            continue

        # Load stock data
        try:
            df = pd.read_csv(stock_path)
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('trade_date').reset_index(drop=True)
        except Exception as e:
            print(f"  Error loading stock data: {e}")
            results[stock_code] = {'error': str(e)}
            continue

        # Calculate technical features
        df = calculate_technical_features(df)

        # Get sector
        ts_code = f"{stock_code}.SZ" if 'sz' in stock_path else f"{stock_code}.SH"
        sector_info = sector_data[sector_data['ts_code'] == ts_code] if len(sector_data) > 0 else pd.DataFrame()
        if len(sector_info) > 0:
            sector = sector_info['sector'].values[0]
        else:
            sector = 'Unknown'
        sector_id = sector_to_id.get(sector, sector_to_id['Unknown'])

        # Remove rows with NaN
        df = df.dropna(subset=feature_cols)

        if len(df) < sequence_length + 1:
            print(f"  Insufficient data for prediction (need {sequence_length + 1} days, got {len(df)})")
            results[stock_code] = {'error': 'Insufficient data'}
            continue

        # Get the most recent data for prediction
        features = df[feature_cols].values
        latest_close = df['close'].iloc[-1]
        latest_date = df['trade_date'].iloc[-1]

        # Create sequence from the last sequence_length days
        seq = features[-sequence_length:]
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize using saved scaler
        seq_normalized = scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(1, sequence_length, -1)
        seq_normalized = np.nan_to_num(seq_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to tensor
        seq_tensor = torch.FloatTensor(seq_normalized).to(device)
        sector_tensor = torch.LongTensor([sector_id]).to(device)

        # Predict
        with torch.no_grad():
            logits = model(seq_tensor, sector_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probs)

        # Store results
        results[stock_code] = {
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'latest_close': latest_close,
            'predicted_class': int(predicted_class),
            'predicted_label': class_names[predicted_class],
            'confidence': float(probs[predicted_class]),
            'all_probabilities': {class_names[i]: float(probs[i]) for i in range(NUM_CLASSES)},
            'sector': sector,
        }

        # Print results
        print(f"  Latest Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"  Latest Close Price: {latest_close:.2f}")
        print(f"  Sector: {sector}")
        print(f"\n  Prediction for NEXT Trading Day:")
        print(f"    Predicted Change Range: {class_names[predicted_class]}")
        print(f"    Confidence: {probs[predicted_class]*100:.1f}%")

        # Show top 3 predictions
        top_indices = np.argsort(probs)[::-1][:3]
        print(f"\n  Top 3 Predictions:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. {class_names[idx]}: {probs[idx]*100:.1f}%")

        # Show probability distribution for extreme movements
        print(f"\n  Extreme Movement Probabilities:")
        print(f"    < -10%: {probs[0]*100:.1f}%")
        print(f"    < -5% (cumulative): {sum(probs[:2])*100:.1f}%")
        print(f"    > +5% (cumulative): {sum(probs[-2:])*100:.1f}%")
        print(f"    > +10%: {probs[-1]*100:.1f}%")

    return results


def main():
    """Main training and evaluation pipeline."""
    print("=" * 70)
    print("Multi-Class Stock Price Change Classification using Transformer")
    print(f"Predicting {NUM_CLASSES} percentage change buckets")
    print("=" * 70)

    # Set random seed
    set_seed(CONFIG['random_seed'])

    # Device
    device = CONFIG['device']
    print(f"\nUsing device: {device}")

    # Load sector data
    print("\n" + "-" * 40)
    print("Loading Sector Data")
    print("-" * 40)
    sector_data = load_sector_data(CONFIG['data_dir'])

    # Load stock data from both markets
    print("\n" + "-" * 40)
    print("Loading Stock Data")
    print("-" * 40)

    all_stocks = {}
    max_per_market = CONFIG['max_stocks'] // 2 if CONFIG['max_stocks'] else None
    for market in ['sh', 'sz']:
        market_stocks = load_stock_data(CONFIG['data_dir'], market, max_stocks=max_per_market)
        all_stocks.update(market_stocks)

    if len(all_stocks) == 0:
        print("No stock data found. Please ensure data exists in stock_data/sh and stock_data/sz directories.")
        return

    print(f"Total stocks loaded: {len(all_stocks)}")

    # Prepare dataset
    print("\n" + "-" * 40)
    print("Preparing Dataset")
    print("-" * 40)

    sequences, labels, sectors = prepare_dataset(
        all_stocks, sector_data, CONFIG['sequence_length'],
        max_sequences_per_stock=CONFIG['max_sequences_per_stock']
    )

    # Free memory from raw stock data
    del all_stocks
    gc.collect()

    print(f"Total samples: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Number of classes: {NUM_CLASSES}")

    # Print class distribution
    class_names = get_class_names()
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for i, name in enumerate(class_names):
        count = counts[list(unique).index(i)] if i in unique else 0
        pct = 100 * count / len(labels)
        print(f"  {name}: {count} ({pct:.1f}%)")

    # Save dimensions for model creation
    input_dim = sequences.shape[2]  # Number of features

    # Split data
    n_samples = len(sequences)
    indices = np.random.permutation(n_samples)

    train_end = int(n_samples * CONFIG['train_ratio'])
    val_end = train_end + int(n_samples * CONFIG['val_ratio'])

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Split sequences and labels
    train_sequences, train_labels, train_sectors = sequences[train_idx], labels[train_idx], sectors[train_idx]
    val_sequences, val_labels, val_sectors = sequences[val_idx], labels[val_idx], sectors[val_idx]
    test_sequences, test_labels, test_sectors = sequences[test_idx], labels[test_idx], sectors[test_idx]

    # Free original arrays
    del sequences, labels, sectors
    gc.collect()

    print(f"\nTrain: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")

    # Normalize data
    print("Normalizing data...")
    train_sequences, val_sequences, test_sequences, scaler = normalize_data(
        train_sequences, val_sequences, test_sequences
    )

    # Create datasets and dataloaders
    train_dataset = StockDataset(train_sequences, train_labels, train_sectors)
    val_dataset = StockDataset(val_sequences, val_labels, val_sectors)
    test_dataset = StockDataset(test_sequences, test_labels, test_sectors)

    # Use weighted sampling if enabled (helps with class imbalance)
    if CONFIG.get('use_weighted_sampling', False):
        print("Using weighted random sampling for balanced batches...")
        train_sampler = create_weighted_sampler(train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            sampler=train_sampler,  # Note: can't use shuffle=True with sampler
            drop_last=True  # Drop incomplete batches for stability
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Create model
    print("\n" + "-" * 40)
    print("Model Architecture")
    print("-" * 40)

    num_sectors = len(sector_data['sector'].unique()) if len(sector_data) > 0 else 0

    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        dropout=CONFIG['dropout'],
        num_sectors=num_sectors,
        use_sector=(num_sectors > 0)
    ).to(device)

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create loss function for handling class imbalance
    print("\n" + "-" * 40)
    print("Loss Function Configuration")
    print("-" * 40)
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)

    criterion = create_loss_function(
        loss_type=CONFIG.get('loss_type', 'focal'),
        num_classes=NUM_CLASSES,
        class_counts=class_counts,
        device=device,
        gamma=CONFIG.get('focal_gamma', 2.0),
        beta=CONFIG.get('cb_beta', 0.9999),
        label_smoothing=CONFIG.get('label_smoothing', 0.0),
        use_class_weights=CONFIG.get('use_class_weights', True)
    )

    # Print class weight summary
    print(f"Class counts: min={class_counts.min()}, max={class_counts.max()}, ratio={class_counts.max()/class_counts.min():.1f}x")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print("\n" + "-" * 40)
    print("Training")
    print("-" * 40)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 10

    # Track training history for visualization
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_predictions, val_labels_true, _ = evaluate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels_true, val_predictions)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    print("\n" + "-" * 40)
    print("Final Evaluation on Test Set")
    print("-" * 40)

    test_loss, test_predictions, test_labels_true, test_probs = evaluate(model, test_loader, criterion, device)

    # Calculate metrics
    accuracy = accuracy_score(test_labels_true, test_predictions)
    precision = precision_score(test_labels_true, test_predictions, average='weighted', zero_division=0)
    recall = recall_score(test_labels_true, test_predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_labels_true, test_predictions, average='weighted', zero_division=0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1 Score (weighted):  {f1:.4f}")

    # Classification report with all classes
    print("\nClassification Report:")
    class_names = get_class_names()
    # Only include classes that appear in the data
    labels_in_data = sorted(list(set(test_labels_true) | set(test_predictions)))
    target_names_filtered = [class_names[i] for i in labels_in_data]
    print(classification_report(test_labels_true, test_predictions,
                               labels=labels_in_data,
                               target_names=target_names_filtered,
                               zero_division=0))

    # Confusion matrix summary
    print("\nConfusion Matrix Summary:")
    cm = confusion_matrix(test_labels_true, test_predictions)
    print(f"Matrix shape: {cm.shape}")

    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        if i < cm.shape[0]:
            class_total = cm[i].sum()
            if class_total > 0:
                class_acc = cm[i, i] / class_total if i < cm.shape[1] else 0
                print(f"  {name}: {class_acc:.2%} ({cm[i, i]}/{class_total})")

    # Baseline comparison (random and majority)
    random_baseline = 1.0 / NUM_CLASSES
    unique, counts = np.unique(test_labels_true, return_counts=True)
    majority_class_pct = counts.max() / len(test_labels_true)
    print(f"\nRandom baseline ({NUM_CLASSES} classes): {random_baseline:.4f}")
    print(f"Majority class baseline: {majority_class_pct:.4f}")
    print(f"Model improvement over random: {(accuracy - random_baseline)*100:.2f}%")
    print(f"Model improvement over majority: {(accuracy - majority_class_pct)*100:.2f}%")

    # Generate visualizations
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots', 'dl_results')
    plot_all_results(
        history=history,
        y_true=test_labels_true,
        y_pred=test_predictions,
        y_probs=test_probs,
        save_dir=plots_dir
    )

    # Save model
    model_path = os.path.join(CONFIG['data_dir'], 'transformer_classifier.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'accuracy': accuracy,
        'history': history,
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Predict for specific stocks requested by user
    # 001270.SZ (*ST 铖昌) and 300788.SZ (中信出版)
    specific_stocks = ['001270', '300788']
    predictions = predict_specific_stocks(
        stock_codes=specific_stocks,
        model_path=model_path,
        data_dir=CONFIG['data_dir'],
        sector_data=sector_data,
        device=device
    )

    # Save predictions to JSON
    predictions_path = os.path.join(CONFIG['data_dir'], 'stock_predictions.json')
    # Convert datetime for JSON serialization
    predictions_json = {}
    for code, pred in predictions.items():
        predictions_json[code] = pred
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_json, f, ensure_ascii=False, indent=2)
    print(f"\nPredictions saved to: {predictions_path}")


if __name__ == '__main__':
    main()
