"""
Pure Temporal Fusion Transformer (Lim et al. 2019) — regression head.

Reference: Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister.
"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
Forecasting." International Journal of Forecasting (2021).

Architecture (single point-estimate output, no quantile heads):

  Past observed inputs (B, T=30, F=174)
            │
            ▼
  Variable Selection Network (per-timestep, F → d_model)
            │
            ▼
  LSTM encoder (hidden=d_model, 1 layer)
            │
            ▼
  Static enrichment via Gated Residual Network
            │
            ▼
  Multi-head temporal self-attention (causal mask)
            │
            ▼
  Position-wise Gated Residual Network
            │
            ▼
  Mean-pool over T → Linear → 1
  Loss: pseudo-Huber

Static covariates (sector_id) flow into the static enrichment GRN.

This is the *pure* paper version — no sector cross-attention, no extended
feature merge, no decoder-side known-future inputs (single-horizon).
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from xgbmodel.train import compute_metrics
from model_compare.engine import Engine, FitResult
from model_compare.seq_data import WindowedPanel, WindowedDataset


# ─── Building blocks (Lim 2019 §4) ──────────────────────────────────────────
class GLU(nn.Module):
    """Gated Linear Unit: x = a ⊙ σ(b)."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.a = nn.Linear(d_in, d_out)
        self.b = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.a(x) * torch.sigmoid(self.b(x))


class GRN(nn.Module):
    """Gated Residual Network — paper Eq. 3 / §4.

    out = LayerNorm( residual(x) + GLU( ELU( W2 x + W3 c ) ) )
    where c is an optional static context.
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 d_context: int = 0, dropout: float = 0.1):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context > 0 else None
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_out, d_out)
        self.ln  = nn.LayerNorm(d_out)

    def forward(self, x, c=None):
        h = self.fc1(x)
        if self.ctx is not None and c is not None:
            # c: (B, d_context) → broadcast over time if x has time dim
            if h.dim() == 3 and c.dim() == 2:
                h = h + self.ctx(c).unsqueeze(1)
            else:
                h = h + self.ctx(c)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        return self.ln(self.skip(x) + h)


class VariableSelectionNetwork(nn.Module):
    """Per-timestep variable selection — paper §4.1.

    Each of F input variables goes through its own 1-layer GRN to a d_model
    embedding; weights are produced by a separate GRN over the concatenated
    flat input then softmax over variables.
    """
    def __init__(self, n_vars: int, d_model: int,
                 d_context: int = 0, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.var_grns = nn.ModuleList([
            GRN(1, d_model, d_model, d_context=0, dropout=dropout)
            for _ in range(n_vars)
        ])
        self.weight_grn = GRN(n_vars, d_model, n_vars,
                               d_context=d_context, dropout=dropout)

    def forward(self, x, c=None):
        # x: (B, T, F)
        # 1. weights over variables
        w_logits = self.weight_grn(x, c)            # (B, T, F)
        weights  = torch.softmax(w_logits, dim=-1)  # (B, T, F)
        # 2. per-variable embeddings
        # embeds: (B, T, F, d_model) — done via list comprehension
        embs = [self.var_grns[i](x[..., i:i+1]) for i in range(self.n_vars)]
        embs = torch.stack(embs, dim=-2)             # (B, T, F, d_model)
        # 3. weighted sum over F
        out = (weights.unsqueeze(-1) * embs).sum(dim=-2)   # (B, T, d_model)
        return out, weights


# ─── TFT model (regression, no quantile heads) ─────────────────────────────
def _causal_mask(T: int) -> torch.Tensor:
    return torch.triu(torch.full((T, T), float('-inf')), diagonal=1)


class TFT(nn.Module):
    def __init__(self, F_in: int, T: int = 30,
                 d_model: int = 128, n_heads: int = 4,
                 d_static: int = 0, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.vsn = VariableSelectionNetwork(F_in, d_model,
                                              d_context=d_static, dropout=dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1,
                              batch_first=True, dropout=0.0)
        self.lstm_glu = GLU(d_model, d_model)
        self.lstm_ln  = nn.LayerNorm(d_model)
        self.static_enrich = GRN(d_model, d_model, d_model,
                                   d_context=d_static, dropout=dropout)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads,
                                            dropout=dropout, batch_first=True)
        self.attn_glu = GLU(d_model, d_model)
        self.attn_ln  = nn.LayerNorm(d_model)
        self.ff_grn = GRN(d_model, d_model, d_model,
                            d_context=0, dropout=dropout)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer('causal_mask', _causal_mask(T))

    def forward(self, x, static=None):
        # x: (B, T, F); static: optional (B, d_static)
        h, _ = self.vsn(x, static)                          # (B, T, d_model)
        lstm_out, _ = self.lstm(h)                          # (B, T, d_model)
        lstm_out = self.lstm_ln(self.lstm_glu(lstm_out) + h)
        enriched = self.static_enrich(lstm_out, static)     # (B, T, d_model)
        attn_out, _ = self.attn(enriched, enriched, enriched,
                                 attn_mask=self.causal_mask, need_weights=False)
        attn_out = self.attn_ln(self.attn_glu(attn_out) + enriched)
        out = self.ff_grn(attn_out)
        pooled = out.mean(dim=1)                            # (B, d_model)
        return self.head(pooled).squeeze(-1)                # (B,)


def pseudo_huber(pred, y, slope=1.0):
    z = (pred - y) / slope
    return (slope ** 2 * (torch.sqrt(1 + z * z) - 1)).mean()


# ─── Engine wrapper ─────────────────────────────────────────────────────────
class TFTEngine(Engine):
    name = 'tft'

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.T          = cfg.get('seq_T',     30)
        self.d_model    = cfg.get('tft_d_model', 128)
        self.n_heads    = cfg.get('tft_n_heads',  4)
        self.dropout    = cfg.get('tft_dropout', 0.1)
        self.epochs     = cfg.get('tft_epochs',  30)
        self.batch_size = cfg.get('tft_batch_size', 512)
        self.lr         = cfg.get('tft_lr', 3e-4)
        self.early_stop_patience = cfg.get('tft_early_stop_patience', 4)
        self._wp_cache  = None

    def set_full_panel(self, panel_df, feat_cols):
        if self._wp_cache is None:
            print(f"[{self.name}] materialising windowed panel from full data "
                  f"(T={self.T}, n_rows={len(panel_df):,}) ...")
            self._wp_cache = WindowedPanel(panel_df, feat_cols, T=self.T)
            print(f"[{self.name}]   {len(self._wp_cache):,} windows total")

    def fit_fold(self, train_df, val_df, test_df, feat_cols):
        wp = self._wp_cache
        if wp is None:
            full = pd.concat([train_df, val_df, test_df], ignore_index=True)
            wp = WindowedPanel(full, feat_cols, T=self.T)
            self._wp_cache = wp

        idx_train = wp.filter_by_dates(train_df['trade_date'].min(),
                                          train_df['trade_date'].max())
        idx_val   = wp.filter_by_dates(val_df  ['trade_date'].min(),
                                          val_df  ['trade_date'].max())
        idx_test  = wp.filter_by_dates(test_df ['trade_date'].min(),
                                          test_df ['trade_date'].max())

        device = torch.device('cuda' if torch.cuda.is_available()
                                and self.cfg.get('device') == 'cuda' else 'cpu')

        ds_tr = WindowedDataset(wp, idx_train)
        ds_va = WindowedDataset(wp, idx_val)
        ds_te = WindowedDataset(wp, idx_test)
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False)

        model = TFT(F_in=len(feat_cols), T=self.T,
                     d_model=self.d_model, n_heads=self.n_heads,
                     d_static=0, dropout=self.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        best_val, best_state, patience = float('inf'), None, 0
        for epoch in range(self.epochs):
            model.train()
            for X, y in dl_tr:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    pred = model(X)
                    loss = pseudo_huber(pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            sched.step()

            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for X, y in dl_va:
                    X = X.to(device); y = y.to(device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        p = model(X)
                    vp.append(p.float().cpu().numpy()); vt.append(y.float().cpu().numpy())
            vp, vt = np.concatenate(vp), np.concatenate(vt)
            v_loss = float(np.mean((vp - vt) ** 2))
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

        val_pred  = _score(idx_val)
        test_pred = _score(idx_test)
        preds = {'val': val_pred, 'test': test_pred}
        metrics = {
            'val':  compute_metrics(preds['val'],  'val' ).as_dict(),
            'test': compute_metrics(preds['test'], 'test').as_dict(),
        }
        return FitResult(model=model, preds=preds, metrics=metrics, best_iteration=None)
