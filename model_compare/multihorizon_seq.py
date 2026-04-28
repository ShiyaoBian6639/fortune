"""
Multi-horizon sequence trainer (transformer + TFT) — single model with 5 heads.

Training one model with 5 output heads is ~5× faster than training 5 separate
models per horizon, and avoids per-horizon hyperparameter drift. The loss is
the sum of per-horizon pseudo-Huber losses (optionally horizon-weighted to
emphasise the d1 forecast which gets used most directly by the strategy).

Architecture:
  Input        (B, T=30, F=~190)
   ↓ Linear projection to d_model=128
   ↓ + sinusoidal positional encoding
   ↓ N × {MultiHead (Flash Attention, fp16/bf16) + FFN + LayerNorm + residual}
   ↓ mean pooling over T
   ↓ Linear → 5  (one prediction per horizon: d1..d5)

Time split (matches GBM multi-horizon trainer):
  train  < 2024-01-01
  val    2024-01-01 to 2025-06-30
  test   ≥ 2025-07-01

Usage:
    ./venv/Scripts/python -m model_compare.multihorizon_seq \\
        --engine transformer --device cuda
    ./venv/Scripts/python -m model_compare.multihorizon_seq \\
        --engine tft --device cuda --subset_only --epochs 30
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'

from xgbmodel.config      import get_config
from xgbmodel.data_loader import build_panel, list_feature_columns


VAL_START   = pd.Timestamp('2024-01-01')
TEST_START  = pd.Timestamp('2025-07-01')
HORIZONS    = (1, 2, 3, 4, 5)


# ─── Sinusoidal PE ───────────────────────────────────────────────────────────
def sinusoidal_pe(T: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(T, d_model)
    pos = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def pseudo_huber(pred, target, slope=1.0):
    z = (pred - target) / slope
    return (slope ** 2 * (torch.sqrt(1 + z * z) - 1)).mean()


# ─── Multi-horizon Transformer ──────────────────────────────────────────────
class MultiHorizonTransformer(nn.Module):
    def __init__(self, F_in: int, T: int = 30,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 384,
                 dropout: float = 0.1, n_horizons: int = 5):
        super().__init__()
        self.T = T
        self.proj = nn.Linear(F_in, d_model)
        self.register_buffer('pe', sinusoidal_pe(T, d_model).unsqueeze(0))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        # 5-head output: one Linear for each horizon, sharing the encoder
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_horizons)])
        # Causal mask (each position attends only to ≤ its own time)
        mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):                                       # (B,T,F)
        h = self.proj(x) + self.pe                              # (B,T,d)
        h = self.encoder(h, mask=self.causal_mask)              # (B,T,d)
        h = h.mean(dim=1)                                        # (B,d)
        out = torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1)
        return out                                               # (B, n_horizons)


# ─── Multi-horizon TFT (lightweight Lim-2019-style) ─────────────────────────
class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(self.skip(x) + g * h)


class MultiHorizonTFT(nn.Module):
    """Lightweight TFT-style model: VSN → LSTM encoder → multi-head attention →
    GRN → 5-horizon heads. Avoids the full Lim 2019 complexity for runtime."""
    def __init__(self, F_in: int, T: int = 30,
                 d_model: int = 128, n_heads: int = 4,
                 dropout: float = 0.1, n_horizons: int = 5):
        super().__init__()
        self.T = T
        # Variable Selection Network (lightweight): per-feature gate
        self.vsn_grn = GatedResidualNetwork(F_in, F_in * 2, d_model, dropout)
        # LSTM encoder
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1,
                             batch_first=True)
        # Multi-head attention with built-in flash backend
        self.mha  = nn.MultiheadAttention(d_model, num_heads=n_heads,
                                            dropout=dropout, batch_first=True)
        # Post-attention GRN
        self.post_grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)
        # Per-horizon heads
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_horizons)])

        # Causal mask for the attention
        mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        self.register_buffer('attn_mask', mask)

    def forward(self, x):                                        # (B,T,F)
        h = self.vsn_grn(x)                                       # (B,T,d)
        lstm_out, _ = self.lstm(h)                                # (B,T,d)
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out,
                                attn_mask=self.attn_mask)
        h = self.post_grn(attn_out + lstm_out)                    # (B,T,d)
        h = h.mean(dim=1)                                          # (B,d)
        out = torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1)
        return out                                                 # (B, n_horizons)


# ─── Multi-horizon windowed dataset ──────────────────────────────────────────
class MultiHorizonWindowedPanel:
    """Builds (X[T,F], y[H], ts_code, trade_date) windows from a long-form panel
    with 5 target columns target_d1..target_d5. Z-scores features (1%/99%
    winsorised) for fp16-safe training."""

    def __init__(self, panel: pd.DataFrame, feat_cols: List[str], T: int = 30):
        self.T = T
        self.feat_cols = feat_cols
        self.F = len(feat_cols)

        target_cols = [f'target_d{h}' for h in HORIZONS]
        keep_cols = ['ts_code', 'trade_date'] + feat_cols + target_cols
        panel = (panel[keep_cols]
                 .sort_values(['ts_code', 'trade_date'])
                 .reset_index(drop=True))

        # Per-feature winsorised z-score stats (sample 1M rows for speed)
        n = len(panel)
        sample = panel.sample(n=min(1_000_000, n), random_state=42)[feat_cols]\
                  .to_numpy(dtype='float32')
        lo = np.nanpercentile(sample, 1.0,  axis=0)
        hi = np.nanpercentile(sample, 99.0, axis=0)
        sample = np.clip(sample, lo, hi)
        mu = np.nanmean(sample, axis=0).astype('float32')
        sd = np.nanstd (sample, axis=0).astype('float32')
        sd = np.where(sd > 1e-6, sd, 1.0).astype('float32')
        self.feat_lo, self.feat_hi = lo.astype('float32'), hi.astype('float32')
        self.feat_mu, self.feat_sd = mu, sd
        print(f"[mh-seq] z-score stats: F={self.F}  "
              f"sd range [{sd.min():.3f}, {sd.max():.3f}]", flush=True)

        def _norm(X):
            X = np.clip(X, self.feat_lo, self.feat_hi)
            return ((X - self.feat_mu) / self.feat_sd).astype('float32')

        self._stock_ix: Dict[str, tuple] = {}
        for ts_code, sub in panel.groupby('ts_code', sort=False):
            X = sub[feat_cols].to_numpy(dtype='float32', copy=False)
            X = _norm(X)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            Y = sub[target_cols].to_numpy(dtype='float32', copy=False)   # (N, H)
            d = sub['trade_date'].values
            self._stock_ix[ts_code] = (X, Y, d)

        # samples = (ts_code, end_idx) where end_idx >= T-1 and
        # all H targets at end_idx are non-NaN
        self.samples: List[tuple] = []
        for ts_code, (X, Y, _) in self._stock_ix.items():
            valid = np.all(np.isfinite(Y), axis=1)
            for k in range(T - 1, len(Y)):
                if valid[k]:
                    self.samples.append((ts_code, k))

    def __len__(self):
        return len(self.samples)

    def get(self, idx):
        ts_code, k = self.samples[idx]
        X, Y, d = self._stock_ix[ts_code]
        return X[k - self.T + 1: k + 1], Y[k], ts_code, d[k]

    def filter_by_dates(self, lo, hi):
        out = []
        for i, (ts_code, k) in enumerate(self.samples):
            d = self._stock_ix[ts_code][2][k]
            if lo <= d <= hi:
                out.append(i)
        return out


class MHDataset(Dataset):
    def __init__(self, wp, indices):
        self.wp = wp; self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        X, Y, _, _ = self.wp.get(self.indices[i])
        return torch.from_numpy(X), torch.from_numpy(Y)


# ─── Training driver ─────────────────────────────────────────────────────────
def _rank_ic(y_true, y_pred):
    if len(y_true) < 50:
        return float('nan')
    s = pd.DataFrame({'y': y_true, 'p': y_pred})
    return float(s['y'].corr(s['p'], method='spearman'))


def train_model(model, wp, idx_train, idx_val, idx_test,
                horizon_weights: torch.Tensor,
                epochs=30, batch_size=512, lr=3e-4, device='cuda',
                early_stop_patience=4, log=print):
    ds_tr = MHDataset(wp, idx_train)
    ds_va = MHDataset(wp, idx_val)
    ds_te = MHDataset(wp, idx_test)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    h_w = horizon_weights.to(device)
    best_val = float('inf'); best_state = None; patience = 0
    log(f"[mh-seq] train {len(idx_train):,}  val {len(idx_val):,}  test {len(idx_test):,}")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0; n_batch = 0
        for X, Y in dl_tr:
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                pred = model(X)                                  # (B,H)
                # Per-horizon weighted pseudo-Huber
                losses = [pseudo_huber(pred[:, h], Y[:, h]) for h in range(pred.size(1))]
                loss = sum(w * l for w, l in zip(h_w, losses)) / h_w.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ep_loss += float(loss.item()); n_batch += 1
        sched.step()

        # Val
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for X, Y in dl_va:
                X = X.to(device); Y = Y.to(device)
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    p = model(X)
                v_preds  .append(p.float().cpu().numpy())
                v_targets.append(Y.float().cpu().numpy())
        v_pred = np.concatenate(v_preds); v_targ = np.concatenate(v_targets)
        v_loss = float(np.mean((v_pred - v_targ) ** 2))
        d1_ic = _rank_ic(v_targ[:, 0], v_pred[:, 0])
        d3_ic = _rank_ic(v_targ[:, 2], v_pred[:, 2])
        d5_ic = _rank_ic(v_targ[:, 4], v_pred[:, 4])
        log(f"  ep {epoch+1:>2}/{epochs}  train_loss={ep_loss/max(n_batch,1):.3f}  "
            f"val_mse={v_loss:.3f}  val_IC d1={d1_ic:+.4f} d3={d3_ic:+.4f} d5={d5_ic:+.4f}")

        if v_loss < best_val - 1e-5:
            best_val = v_loss; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                log(f"  early stop at epoch {epoch+1}")
                break
    log(f"[mh-seq] training time: {(time.time()-t0)/60:.1f} min")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dl_te


def score_test(model, dl_te, device='cuda'):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for X, Y in dl_te:
            X = X.to(device); Y = Y.to(device)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                p = model(X)
            preds.append(p.float().cpu().numpy())
            targs.append(Y.float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(targs)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', choices=['transformer', 'tft'], default='transformer')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_stocks', type=int, default=0)
    p.add_argument('--subset_only', action='store_true',
                   help='use only stocks in stock_data/stock_subset.csv (in_subset=True)')
    p.add_argument('--target_basis', choices=['cc', 'oc'], default='oc')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--T',       type=int, default=30)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--horizon_weights', default='1.5,1.0,0.8,0.6,0.5',
                   help='per-horizon loss weights, comma-separated. d1 weighted '
                        'highest by default (most important to strategy).')
    args = p.parse_args()

    cfg = get_config(max_stocks=args.max_stocks, device=args.device, for_inference=False)
    cfg['target_basis'] = args.target_basis

    print(f"[mh-seq] building panel  target_basis={args.target_basis} ...", flush=True)
    panel = build_panel(cfg)

    # Subset filter
    if args.subset_only:
        sub_p = DATA / 'stock_subset.csv'
        if not sub_p.exists():
            raise SystemExit(f"--subset_only requires {sub_p}; run backtest.select_subset first.")
        sub = pd.read_csv(sub_p, encoding='utf-8-sig')
        keep = set(sub[sub['in_subset'] == True]['ts_code'])
        before = panel['ts_code'].nunique()
        panel = panel[panel['ts_code'].isin(keep)].reset_index(drop=True)
        print(f"[mh-seq] subset filter: {before} → {panel['ts_code'].nunique()} stocks", flush=True)

    feat_cols = list_feature_columns(panel)
    print(f"[mh-seq] panel: {panel.shape}  features={len(feat_cols)}", flush=True)

    # Build windowed panel (slow — done once)
    wp = MultiHorizonWindowedPanel(panel, feat_cols, T=args.T)
    print(f"[mh-seq] {len(wp):,} windows total", flush=True)

    # Time-split indices
    idx_train = wp.filter_by_dates(panel['trade_date'].min(), VAL_START - pd.Timedelta(days=1))
    idx_val   = wp.filter_by_dates(VAL_START, TEST_START - pd.Timedelta(days=1))
    idx_test  = wp.filter_by_dates(TEST_START, panel['trade_date'].max())
    print(f"[mh-seq] split: train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}",
          flush=True)

    # Build model
    Cls = MultiHorizonTransformer if args.engine == 'transformer' else MultiHorizonTFT
    model = Cls(F_in=len(feat_cols), T=args.T,
                d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers if args.engine == 'transformer' else 1,
                dropout=args.dropout, n_horizons=len(HORIZONS)).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mh-seq] {args.engine}: {n_params/1e6:.2f}M params", flush=True)

    h_w = torch.tensor([float(x) for x in args.horizon_weights.split(',')], dtype=torch.float32)
    print(f"[mh-seq] horizon weights: {h_w.tolist()}", flush=True)

    model, dl_te = train_model(
        model, wp, idx_train, idx_val, idx_test,
        horizon_weights=h_w,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device,
    )

    # Score test
    print(f"[mh-seq] scoring test ...", flush=True)
    test_pred, test_targ = score_test(model, dl_te, device=args.device)
    print()
    print("=" * 60)
    print(f"PER-HORIZON TEST IC ({args.engine}, target_basis={args.target_basis})")
    print("=" * 60)
    print(f"{'horizon':>10}  {'rank_IC':>9}  {'RMSE':>9}")
    for h_i, h in enumerate(HORIZONS):
        ic   = _rank_ic(test_targ[:, h_i], test_pred[:, h_i])
        rmse = float(np.sqrt(np.mean((test_pred[:, h_i] - test_targ[:, h_i]) ** 2)))
        print(f"{'d'+str(h):>10}  {ic:>+9.4f}  {rmse:>9.4f}")

    # Save model + per-stock predictions on test
    suffix = '_oc' if args.target_basis == 'oc' else '_cc'
    md = DATA / f'models_{args.engine}_mh{suffix}'
    md.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), md / 'model.pt')

    # Build per-(stock, date) test prediction DataFrame
    rows = []
    for j, ix in enumerate(idx_test):
        ts, k = wp.samples[ix]
        d = wp._stock_ix[ts][2][k]
        row = {'ts_code': ts, 'trade_date': pd.Timestamp(d)}
        for h_i, h in enumerate(HORIZONS):
            row[f'pred_d{h}']   = float(test_pred[j, h_i])
            row[f'target_d{h}'] = float(test_targ[j, h_i])
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_p  = md / 'xgb_preds' / 'test.csv'
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False)
    print(f"[mh-seq] saved {len(out_df):,} test rows → {out_p}", flush=True)

    with open(md / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump({
            'engine':         args.engine,
            'mode':           'multi_horizon_seq',
            'target_basis':   args.target_basis,
            'subset_only':    args.subset_only,
            'epochs':         args.epochs,
            'd_model':        args.d_model,
            'T':              args.T,
            'n_features':     len(feat_cols),
            'n_train':        len(idx_train),
            'n_val':          len(idx_val),
            'n_test':         len(idx_test),
            'horizon_weights': h_w.tolist(),
        }, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
