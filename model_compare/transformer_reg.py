"""
Pure encoder-only Transformer for next-day excess-return regression.

Architecture (intentionally minimal — no extras):
  Input        (B, T=30, F=174)
   ↓ Linear projection to d_model
   ↓ + sinusoidal positional encoding
   ↓ N × {MultiHead(causal) + FFN + LayerNorm + residual}
   ↓ mean pooling over T
   ↓ Linear → 1   (regression head)
  Loss: pseudo-Huber (matches XGBoost's huber objective)

Reference: Vaswani et al. 2017 "Attention is All You Need" — encoder block.
NOT reusing dl/models.py (which is a 7-class classifier head specialised for
relative-return classification).
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from xgbmodel.train import compute_metrics
from model_compare.engine import Engine, FitResult
from model_compare.seq_data import WindowedPanel, WindowedDataset


# ─── Sinusoidal positional encoding ─────────────────────────────────────────
def sinusoidal_pe(T: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(T, d_model)
    pos = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe   # (T, d_model)


class TransformerRegressor(nn.Module):
    def __init__(self, F_in: int, T: int = 30,
                 d_model: int = 192, n_heads: int = 6,
                 n_layers: int = 3, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.proj = nn.Linear(F_in, d_model)
        self.register_buffer('pe', sinusoidal_pe(T, d_model).unsqueeze(0))   # (1,T,d)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

        # Causal mask: position t can only attend to ≤ t
        mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        # x: (B, T, F)
        h = self.proj(x) + self.pe                   # (B, T, d)
        h = self.encoder(h, mask=self.causal_mask)   # (B, T, d)
        h = h.mean(dim=1)                            # (B, d)  mean-pool
        return self.head(h).squeeze(-1)              # (B,)


def pseudo_huber_loss(pred, target, slope: float = 1.0):
    z = (pred - target) / slope
    return (slope ** 2 * (torch.sqrt(1 + z * z) - 1)).mean()


# ─── Engine wrapper ─────────────────────────────────────────────────────────
class TransformerEngine(Engine):
    name = 'transformer_reg'

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.T          = cfg.get('seq_T',     30)
        self.d_model    = cfg.get('seq_d_model', 192)
        self.n_layers   = cfg.get('seq_n_layers', 3)
        self.n_heads    = cfg.get('seq_n_heads',  6)
        self.d_ff       = cfg.get('seq_d_ff',    512)
        self.dropout    = cfg.get('seq_dropout', 0.1)
        self.epochs     = cfg.get('seq_epochs',  30)
        self.batch_size = cfg.get('seq_batch_size', 1024)
        self.lr         = cfg.get('seq_lr', 3e-4)
        self.early_stop_patience = cfg.get('seq_early_stop_patience', 4)
        self._wp_cache  = None    # WindowedPanel built once from full panel

    def set_full_panel(self, panel_df: pd.DataFrame, feat_cols: List[str]):
        """Called once by the orchestrator before iterating folds. Builds the
        WindowedPanel from the complete dataset so per-fold date filters can
        recover full historical context for any window."""
        if self._wp_cache is None:
            print(f"[{self.name}] materialising windowed panel from full data "
                  f"(T={self.T}, n_rows={len(panel_df):,}) ...")
            self._wp_cache = WindowedPanel(panel_df, feat_cols, T=self.T)
            print(f"[{self.name}]   {len(self._wp_cache):,} windows total")

    def fit_fold(self, train_df, val_df, test_df, feat_cols):
        wp = self._wp_cache
        if wp is None:
            # Fallback if orchestrator didn't call set_full_panel — use the
            # union of fold slices (less ideal, retained for robustness)
            full = pd.concat([train_df, val_df, test_df], ignore_index=True)
            wp = WindowedPanel(full, feat_cols, T=self.T)
            self._wp_cache = wp

        train_lo, train_hi = train_df['trade_date'].min(), train_df['trade_date'].max()
        val_lo,   val_hi   = val_df  ['trade_date'].min(), val_df  ['trade_date'].max()
        test_lo,  test_hi  = test_df ['trade_date'].min(), test_df ['trade_date'].max()

        idx_train = wp.filter_by_dates(train_lo, train_hi)
        idx_val   = wp.filter_by_dates(val_lo,   val_hi)
        idx_test  = wp.filter_by_dates(test_lo,  test_hi)

        device = torch.device('cuda' if torch.cuda.is_available() and self.cfg.get('device') == 'cuda' else 'cpu')

        ds_tr = WindowedDataset(wp, idx_train)
        ds_va = WindowedDataset(wp, idx_val)
        ds_te = WindowedDataset(wp, idx_test)

        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True,  num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False, num_workers=0)
        dl_te = DataLoader(ds_te, batch_size=self.batch_size, shuffle=False, num_workers=0)

        model = TransformerRegressor(
            F_in=len(feat_cols), T=self.T,
            d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, d_ff=self.d_ff,
            dropout=self.dropout,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        best_val = float('inf')
        best_state = None
        patience = 0
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        for epoch in range(self.epochs):
            model.train()
            for X, y in dl_tr:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    pred = model(X)
                    loss = pseudo_huber_loss(pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            sched.step()

            # Val
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for X, y in dl_va:
                    X = X.to(device); y = y.to(device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        p = model(X)
                    v_preds.append(p.float().cpu().numpy())
                    v_targets.append(y.float().cpu().numpy())
            v_pred = np.concatenate(v_preds); v_targ = np.concatenate(v_targets)
            v_loss = float(np.mean((v_pred - v_targ) ** 2))
            print(f"  ep {epoch+1:>2}/{self.epochs} val_mse={v_loss:.4f}")

            if v_loss < best_val - 1e-5:
                best_val = v_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    print(f"  early stop at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Score val + test, build pred frames
        def _score(idx_list):
            ds = WindowedDataset(wp, idx_list)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for X, _ in loader:
                    X = X.to(device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        p = model(X)
                    preds.append(p.float().cpu().numpy())
            preds = np.concatenate(preds) if preds else np.array([])
            rows = []
            for j, ix in enumerate(idx_list):
                ts, k = wp.samples[ix]
                rows.append({
                    'ts_code':    ts,
                    'trade_date': wp._stock_ix[ts][2][k],
                    'pred':       float(preds[j]),
                    'target':     float(wp._stock_ix[ts][1][k]),
                })
            if not rows:
                return pd.DataFrame(columns=['ts_code','trade_date','pred','target'])
            return pd.DataFrame(rows)

        val_pred = _score(idx_val)
        test_pred = _score(idx_test)
        preds = {'val': val_pred, 'test': test_pred}
        metrics = {
            'val':  compute_metrics(preds['val'],  'val' ).as_dict(),
            'test': compute_metrics(preds['test'], 'test').as_dict(),
        }
        return FitResult(model=model, preds=preds, metrics=metrics, best_iteration=None)
