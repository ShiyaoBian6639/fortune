"""
Sequence dataset for Transformer / TFT.

Slices the same panel used by gradient boosting into per-(stock, date) windows
of T=30 consecutive days × F=174 features. Aligns target with the panel's
existing 'target' column (next-day excess return, demeaned cross-sectionally).

The expensive part is grouping by ts_code, sorting by date, and indexing —
done once at panel-load time, then constant-time slicing per fold.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowedPanel:
    """Materialises (X[T,F], y, ts_code, trade_date) windows from a long-form
    panel. Holds float32 arrays per ts_code so __getitem__ is O(1).

    Z-scores features per-column using global mean/std computed once on the
    panel. XGBoost is scale-invariant, but Transformer/TFT need normalised
    inputs to avoid fp16 overflow during AMP training. The 1%/99% winsor
    pre-clip suppresses extreme outliers (raw `vol`, `amount`, etc.) that
    would otherwise dominate the std estimate.
    """

    def __init__(self, panel: pd.DataFrame, feat_cols: List[str], T: int = 30):
        self.T = T
        self.feat_cols = feat_cols
        self.F = len(feat_cols)

        # Sort once by (ts_code, trade_date)
        panel = panel.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

        # ── Compute per-feature normalization stats (winsorised z-score) ──
        # Using float32 for memory; sample 1M rows max to keep this fast.
        n = len(panel)
        sample = panel.sample(n=min(1_000_000, n), random_state=42)[feat_cols].to_numpy(dtype='float32')
        # 1%/99% winsor → robust mean/std
        lo = np.nanpercentile(sample, 1.0,  axis=0)
        hi = np.nanpercentile(sample, 99.0, axis=0)
        sample = np.clip(sample, lo, hi)
        mu = np.nanmean(sample, axis=0).astype('float32')
        sd = np.nanstd (sample, axis=0).astype('float32')
        sd = np.where(sd > 1e-6, sd, 1.0).astype('float32')
        self.feat_lo, self.feat_hi = lo.astype('float32'), hi.astype('float32')
        self.feat_mu, self.feat_sd = mu, sd
        print(f"[WindowedPanel] z-scoring {self.F} features "
              f"(mu range [{mu.min():.3f}, {mu.max():.3f}], "
              f"sd range [{sd.min():.3f}, {sd.max():.3f}])")

        def _norm(X):
            X = np.clip(X, self.feat_lo, self.feat_hi)
            return ((X - self.feat_mu) / self.feat_sd).astype('float32')

        # Group → numpy
        self._stock_ix: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for ts_code, sub in panel.groupby('ts_code', sort=False):
            X = sub[feat_cols].to_numpy(dtype='float32', copy=False)
            X = _norm(X)
            # Replace any remaining NaN with 0 (post-norm; shouldn't happen but defensive)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = sub['target'].to_numpy(dtype='float32', copy=False)
            d = sub['trade_date'].values
            self._stock_ix[ts_code] = (X, y, d)

        # Build (ts_code, end_idx) → flat index for the full universe.
        # A sample at end_idx=k means window covers rows [k-T+1, k] inclusive.
        self.samples: List[Tuple[str, int]] = []
        for ts_code, (X, y, _) in self._stock_ix.items():
            for k in range(T - 1, len(y)):
                if np.isfinite(y[k]):   # only training samples with a target
                    self.samples.append((ts_code, k))

    def __len__(self):
        return len(self.samples)

    def get_window(self, idx: int):
        ts_code, k = self.samples[idx]
        X, y, d = self._stock_ix[ts_code]
        return X[k - self.T + 1: k + 1], float(y[k]), ts_code, d[k]

    def filter_by_dates(self, lo: pd.Timestamp, hi: pd.Timestamp) -> List[int]:
        """Indices (into self.samples) whose target date is in [lo, hi]."""
        out = []
        for i, (ts_code, k) in enumerate(self.samples):
            d = self._stock_ix[ts_code][2][k]
            if lo <= d <= hi:
                out.append(i)
        return out


class WindowedDataset(Dataset):
    """Torch Dataset wrapping WindowedPanel and a list of sample indices."""
    def __init__(self, wp: WindowedPanel, indices: List[int]):
        self.wp      = wp
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        X, y, _, _ = self.wp.get_window(self.indices[i])
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)
