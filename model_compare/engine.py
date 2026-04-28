"""
Uniform engine interface for cross-model comparison.

Each engine wraps a fit/predict cycle with this signature:

    engine.fit_fold(train_df, val_df, test_df, feat_cols) -> (model, preds, metrics)

where preds is {'val': DataFrame, 'test': DataFrame} with columns
[ts_code, trade_date, pred, target] and metrics is the same shape as the
existing xgbmodel.train metrics.

The walk-forward orchestrator in `model_compare.walk` is engine-agnostic
and produces the canonical `xgb_preds/test.csv` output that the backtest
and dashboard already consume.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    model: Any
    preds: Dict[str, pd.DataFrame]   # {'val': ..., 'test': ...}
    metrics: Dict[str, Dict]         # {'val': {...}, 'test': {...}}
    best_iteration: Optional[int] = None
    extra: Dict = field(default_factory=dict)


class Engine(ABC):
    """Subclass this for each model type."""
    name: str = 'engine'

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def fit_fold(self,
                 train_df: pd.DataFrame,
                 val_df:   pd.DataFrame,
                 test_df:  pd.DataFrame,
                 feat_cols: List[str]) -> FitResult:
        ...

    def model_dir(self) -> Path:
        d = Path('stock_data') / f'models_{self.name}'
        d.mkdir(parents=True, exist_ok=True)
        (d / 'xgb_preds').mkdir(exist_ok=True)
        return d


# ─── Reusable score-frame helper (used by all gradient-boosting engines) ────
def score_frame(predict_fn, df_wide: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """Build a (ts_code, trade_date, pred, target) frame for an arbitrary
    predict_fn(X)→1d array. predict_fn is engine-specific."""
    pred = predict_fn(df_wide[feat_cols])
    return pd.DataFrame({
        'ts_code':    df_wide['ts_code'].values,
        'trade_date': df_wide['trade_date'].values,
        'pred':       np.asarray(pred, dtype='float32'),
        'target':     df_wide['target'].values,
    })
