"""
Faithful Lim 2019 TFT, multi-horizon (t+1..t+5), categorised inputs.

Splits the 199 panel features into three TFT input classes:
  STATIC         — per-stock time-invariant: sector_id, in_csi300, years_listed, ...
  KNOWN_FUTURE   — calendar features (dow, dom, …) known at past AND future steps
  PAST_OBSERVED  — everything else: prices, volumes, technicals, macros, fundamentals

Then trains a TemporalFusionTransformer (model_compare/tft_paper.py) end-to-end
with the same time-split as the GBM multi-horizon trainer:
  train  < 2024-01-01
  val    2024-01-01 .. 2025-06-30
  test   ≥ 2025-07-01

Run:
    ./venv/Scripts/python -m model_compare.multihorizon_tft \\
        --device cuda --subset_only --epochs 30 --target_basis oc
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'

from xgbmodel.config      import get_config
from xgbmodel.data_loader import build_panel, list_feature_columns
from model_compare.feature_categories import categorize, report
from model_compare.tft_paper import TemporalFusionTransformer, pseudo_huber


VAL_START   = pd.Timestamp('2024-01-01')
TEST_START  = pd.Timestamp('2025-07-01')
HORIZONS    = (1, 2, 3, 4, 5)
H           = len(HORIZONS)


# ─── Windowed panel for TFT ──────────────────────────────────────────────────
class TFTWindowedPanel:
    """Builds samples (static, past_obs, past_kn, future_kn, y_h) from a long panel.

    For each (stock, end_idx) sample at trade_date d_t with window [t-T+1, t]:
      • static       = panel[t][static_cols].numpy()                # (S,)
      • past_obs     = panel[t-T+1..t][past_obs_cols].numpy()       # (T, P)
      • past_known   = panel[t-T+1..t][past_known_cols].numpy()     # (T, K)
      • future_known = panel[t+1..t+H][past_known_cols].numpy()     # (H, K)
      • y_h          = panel[t][target_d1..target_d5].numpy()       # (H,)

    Static features are taken from the LATEST timestamp (t) — this catches the
    PIT membership flag value at sample time. They're then constant for the
    sample (the model treats them as time-invariant).
    """
    def __init__(self, panel: pd.DataFrame,
                 static_cols: List[str],
                 past_obs_cols: List[str],
                 past_known_cols: List[str],
                 T: int = 30):
        self.T = T
        self.static_cols = static_cols
        self.past_obs_cols = past_obs_cols
        self.past_known_cols = past_known_cols
        self.S = len(static_cols)
        self.P = len(past_obs_cols)
        self.K = len(past_known_cols)

        target_cols = [f'target_d{h}' for h in HORIZONS]
        keep = ['ts_code', 'trade_date'] + static_cols + past_obs_cols + past_known_cols + target_cols
        panel = (panel[keep]
                 .sort_values(['ts_code', 'trade_date'])
                 .reset_index(drop=True))

        # Per-feature winsorised z-score stats over the time-varying features
        # (static features pass through unscaled — they're typically already
        # bounded categoricals/binaries).
        tv_cols = past_obs_cols + past_known_cols
        n = len(panel)
        sample = panel.sample(n=min(1_000_000, n), random_state=42)[tv_cols]\
                  .to_numpy(dtype='float32')
        lo = np.nanpercentile(sample, 1.0,  axis=0)
        hi = np.nanpercentile(sample, 99.0, axis=0)
        sample = np.clip(sample, lo, hi)
        mu = np.nanmean(sample, axis=0).astype('float32')
        sd = np.nanstd (sample, axis=0).astype('float32')
        sd = np.where(sd > 1e-6, sd, 1.0).astype('float32')
        self.tv_lo, self.tv_hi = lo.astype('float32'), hi.astype('float32')
        self.tv_mu, self.tv_sd = mu, sd
        print(f"[mh-tft] z-score stats: tv_features={len(tv_cols)} "
              f"sd range [{sd.min():.3f}, {sd.max():.3f}]", flush=True)

        def _norm_tv(X):
            X = np.clip(X, self.tv_lo, self.tv_hi)
            return ((X - self.tv_mu) / self.tv_sd).astype('float32')

        # Per-stock arrays
        self._stock_ix: Dict[str, tuple] = {}
        for ts_code, sub in panel.groupby('ts_code', sort=False):
            S_arr  = sub[static_cols].to_numpy(dtype='float32', copy=False) if self.S else None
            tv_arr = _norm_tv(sub[tv_cols].to_numpy(dtype='float32', copy=False))
            tv_arr = np.nan_to_num(tv_arr, nan=0.0, posinf=0.0, neginf=0.0)
            P_arr  = tv_arr[:, :self.P]
            K_arr  = tv_arr[:, self.P:]
            Y_arr  = sub[target_cols].to_numpy(dtype='float32', copy=False)
            d_arr  = sub['trade_date'].values
            self._stock_ix[ts_code] = (S_arr, P_arr, K_arr, Y_arr, d_arr)

        # samples: (ts_code, end_idx) where end_idx ≥ T-1 (full past window)
        # AND end_idx + H < N (full future-known available — KNOWN features
        # exist at every future trading day so this is safe).
        # AND all H targets at end_idx are non-NaN (training-time only).
        self.samples: List[tuple] = []
        for ts_code, (_, _, _, Y, _) in self._stock_ix.items():
            n_rows = len(Y)
            valid_y = np.all(np.isfinite(Y), axis=1)
            for k in range(T - 1, n_rows - H):
                if valid_y[k]:
                    self.samples.append((ts_code, k))

    def __len__(self):
        return len(self.samples)

    def get(self, idx):
        ts_code, k = self.samples[idx]
        S_arr, P_arr, K_arr, Y_arr, d_arr = self._stock_ix[ts_code]
        # static = sample at t (the prediction time)
        static = S_arr[k] if S_arr is not None else np.zeros(0, dtype='float32')
        past_obs   = P_arr[k - self.T + 1: k + 1]            # (T, P)
        past_known = K_arr[k - self.T + 1: k + 1]            # (T, K)
        # future-known: t+1..t+H (assumes trade_date for those days exists)
        future_known = K_arr[k + 1: k + 1 + H]               # (H, K)
        y = Y_arr[k]                                          # (H,)
        return static, past_obs, past_known, future_known, y, ts_code, d_arr[k]

    def filter_by_dates(self, lo, hi):
        out = []
        for i, (ts_code, k) in enumerate(self.samples):
            d = self._stock_ix[ts_code][4][k]
            if lo <= d <= hi:
                out.append(i)
        return out


class TFTDataset(Dataset):
    def __init__(self, wp: TFTWindowedPanel, indices: List[int]):
        self.wp = wp
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s, p, k, f, y, _, _ = self.wp.get(self.indices[i])
        return (torch.from_numpy(s),
                torch.from_numpy(p),
                torch.from_numpy(k),
                torch.from_numpy(f),
                torch.from_numpy(y))


# ─── Training driver ─────────────────────────────────────────────────────────
def _rank_ic(y_true, y_pred):
    if len(y_true) < 50:
        return float('nan')
    s = pd.DataFrame({'y': y_true, 'p': y_pred})
    return float(s['y'].corr(s['p'], method='spearman'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_stocks', type=int, default=0)
    p.add_argument('--subset_only', action='store_true')
    p.add_argument('--target_basis', choices=['cc', 'oc'], default='oc')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--T',       type=int, default=30)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--horizon_weights', default='1.5,1.0,0.8,0.6,0.5',
                   help='per-horizon loss weights')
    p.add_argument('--early_stop', type=int, default=4)
    p.add_argument('--past_obs_top_k', type=int, default=0,
                   help='If >0, keep only the top-K past-observed features by '
                        'GBM importance (loaded from xgb_pct_chg.meta.json). '
                        'The VSN scales linearly with n_past_obs, so K=50 '
                        'gives ~5× speedup vs the full 178 features.')
    p.add_argument('--xgb_meta', default='stock_data/models/xgb_pct_chg.meta.json',
                   help='source for feature_importance_top50 ranking')
    args = p.parse_args()

    cfg = get_config(max_stocks=args.max_stocks, device=args.device, for_inference=False)
    cfg['target_basis'] = args.target_basis

    print(f"[mh-tft] building panel  target_basis={args.target_basis} ...", flush=True)
    panel = build_panel(cfg)

    if args.subset_only:
        sub_p = DATA / 'stock_subset.csv'
        if not sub_p.exists():
            raise SystemExit(f"--subset_only requires {sub_p}; run backtest.select_subset")
        sub = pd.read_csv(sub_p, encoding='utf-8-sig')
        keep = set(sub[sub['in_subset'] == True]['ts_code'])
        before = panel['ts_code'].nunique()
        panel = panel[panel['ts_code'].isin(keep)].reset_index(drop=True)
        print(f"[mh-tft] subset: {before} → {panel['ts_code'].nunique()} stocks", flush=True)

    feat_cols = list_feature_columns(panel)
    static_cols, past_kn_cols, past_obs_cols = report(feat_cols)

    # Optional: trim past_obs to top-K features by GBM importance.
    if args.past_obs_top_k > 0:
        try:
            with open(args.xgb_meta, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            top_feats = [r['feature'] for r in meta.get('feature_importance_top50', [])]
            keep = [f for f in top_feats if f in past_obs_cols][: args.past_obs_top_k]
            print(f"[mh-tft] past_obs feature pruning: {len(past_obs_cols)} → {len(keep)} "
                  f"(top-{args.past_obs_top_k} by GBM importance)", flush=True)
            past_obs_cols = keep
        except Exception as e:
            print(f"[mh-tft] WARNING: feature pruning failed ({e}); using all features")

    print(f"[mh-tft] panel: {panel.shape}  features={len(static_cols)+len(past_kn_cols)+len(past_obs_cols)}",
          flush=True)

    # Build windowed panel
    wp = TFTWindowedPanel(panel, static_cols, past_obs_cols, past_kn_cols, T=args.T)
    print(f"[mh-tft] {len(wp):,} windows total", flush=True)

    idx_train = wp.filter_by_dates(panel['trade_date'].min(),
                                     VAL_START - pd.Timedelta(days=1))
    idx_val   = wp.filter_by_dates(VAL_START, TEST_START - pd.Timedelta(days=1))
    idx_test  = wp.filter_by_dates(TEST_START, panel['trade_date'].max())
    print(f"[mh-tft] split: train={len(idx_train):,} val={len(idx_val):,} test={len(idx_test):,}",
          flush=True)

    # Model
    model = TemporalFusionTransformer(
        n_static=len(static_cols),
        n_past_obs=len(past_obs_cols),
        n_past_known=len(past_kn_cols),
        n_future_known=len(past_kn_cols),    # same set as past_known
        T=args.T, H=H,
        d_model=args.d_model, n_heads=args.n_heads, dropout=args.dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mh-tft] TemporalFusionTransformer: {n_params/1e6:.2f}M params  "
          f"S={len(static_cols)} P={len(past_obs_cols)} K={len(past_kn_cols)}", flush=True)

    h_w = torch.tensor([float(x) for x in args.horizon_weights.split(',')],
                        dtype=torch.float32, device=args.device)
    print(f"[mh-tft] horizon weights: {h_w.tolist()}", flush=True)

    ds_tr = TFTDataset(wp, idx_train)
    ds_va = TFTDataset(wp, idx_val)
    ds_te = TFTDataset(wp, idx_test)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.device == 'cuda'))

    best_val = float('inf'); best_state = None; patience = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; n_batch = 0
        for s_t, p_t, k_t, f_t, y_t in dl_tr:
            s_t = s_t.to(args.device, non_blocking=True)
            p_t = p_t.to(args.device, non_blocking=True)
            k_t = k_t.to(args.device, non_blocking=True)
            f_t = f_t.to(args.device, non_blocking=True)
            y_t = y_t.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(args.device == 'cuda')):
                pred = model(s_t if len(static_cols) > 0 else None,
                              p_t if len(past_obs_cols) > 0 else None,
                              k_t if len(past_kn_cols)  > 0 else None,
                              f_t if len(past_kn_cols)  > 0 else None)        # (B, H)
                losses = [pseudo_huber(pred[:, h_i], y_t[:, h_i]) for h_i in range(H)]
                loss = sum(w * l for w, l in zip(h_w, losses)) / h_w.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ep_loss += float(loss.item()); n_batch += 1
        sched.step()

        # Val
        model.eval()
        v_p, v_t = [], []
        with torch.no_grad():
            for s_t, p_t, k_t, f_t, y_t in dl_va:
                s_t = s_t.to(args.device); p_t = p_t.to(args.device)
                k_t = k_t.to(args.device); f_t = f_t.to(args.device)
                y_t = y_t.to(args.device)
                with torch.amp.autocast('cuda', enabled=(args.device == 'cuda')):
                    pr = model(s_t if len(static_cols) > 0 else None,
                                p_t if len(past_obs_cols) > 0 else None,
                                k_t if len(past_kn_cols)  > 0 else None,
                                f_t if len(past_kn_cols)  > 0 else None)
                v_p.append(pr.float().cpu().numpy())
                v_t.append(y_t.float().cpu().numpy())
        v_pred = np.concatenate(v_p); v_targ = np.concatenate(v_t)
        v_loss = float(np.mean((v_pred - v_targ) ** 2))
        d1_ic = _rank_ic(v_targ[:, 0], v_pred[:, 0])
        d3_ic = _rank_ic(v_targ[:, 2], v_pred[:, 2])
        d5_ic = _rank_ic(v_targ[:, 4], v_pred[:, 4])
        print(f"  ep {epoch+1:>2}/{args.epochs}  train_loss={ep_loss/max(n_batch,1):.3f}  "
              f"val_mse={v_loss:.3f}  val_IC d1={d1_ic:+.4f} d3={d3_ic:+.4f} d5={d5_ic:+.4f}",
              flush=True)
        if v_loss < best_val - 1e-5:
            best_val = v_loss; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"  early stop at epoch {epoch+1}"); break
    print(f"[mh-tft] training time: {(time.time()-t0)/60:.1f} min", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Score test
    print(f"[mh-tft] scoring test ...", flush=True)
    model.eval()
    t_p, t_t, t_codes, t_dates = [], [], [], []
    with torch.no_grad():
        for batch_i in range(0, len(idx_test), args.batch_size):
            batch_idx = idx_test[batch_i:batch_i + args.batch_size]
            s_l, p_l, k_l, f_l, y_l, c_l, d_l = [], [], [], [], [], [], []
            for ix in batch_idx:
                s, p, k, f, y, ts, dd = wp.get(ix)
                s_l.append(s); p_l.append(p); k_l.append(k); f_l.append(f); y_l.append(y)
                c_l.append(ts); d_l.append(dd)
            s_t = torch.from_numpy(np.stack(s_l)).to(args.device) if len(static_cols) > 0 else None
            p_t = torch.from_numpy(np.stack(p_l)).to(args.device) if len(past_obs_cols) > 0 else None
            k_t = torch.from_numpy(np.stack(k_l)).to(args.device) if len(past_kn_cols)  > 0 else None
            f_t = torch.from_numpy(np.stack(f_l)).to(args.device) if len(past_kn_cols)  > 0 else None
            with torch.amp.autocast('cuda', enabled=(args.device == 'cuda')):
                pr = model(s_t, p_t, k_t, f_t)
            t_p.append(pr.float().cpu().numpy())
            t_t.append(np.stack(y_l).astype('float32'))
            t_codes.extend(c_l); t_dates.extend(d_l)
    test_pred = np.concatenate(t_p); test_targ = np.concatenate(t_t)

    print()
    print("=" * 60)
    print(f"PER-HORIZON TEST IC (faithful TFT, target_basis={args.target_basis})")
    print("=" * 60)
    print(f"{'horizon':>10}  {'rank_IC':>9}  {'RMSE':>9}")
    for h_i, h_v in enumerate(HORIZONS):
        ic   = _rank_ic(test_targ[:, h_i], test_pred[:, h_i])
        rmse = float(np.sqrt(np.mean((test_pred[:, h_i] - test_targ[:, h_i]) ** 2)))
        print(f"{'d'+str(h_v):>10}  {ic:>+9.4f}  {rmse:>9.4f}")

    suffix = '_oc' if args.target_basis == 'oc' else '_cc'
    md = DATA / f'models_tft_paper_mh{suffix}'
    md.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), md / 'model.pt')

    rows = []
    for j in range(len(test_pred)):
        row = {'ts_code': t_codes[j], 'trade_date': pd.Timestamp(t_dates[j])}
        for h_i, h_v in enumerate(HORIZONS):
            row[f'pred_d{h_v}']   = float(test_pred[j, h_i])
            row[f'target_d{h_v}'] = float(test_targ[j, h_i])
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_p  = md / 'xgb_preds' / 'test.csv'
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p, index=False)
    print(f"[mh-tft] saved {len(out_df):,} test rows → {out_p}", flush=True)

    with open(md / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump({
            'engine':         'tft_paper',
            'mode':           'multi_horizon_seq',
            'target_basis':   args.target_basis,
            'subset_only':    args.subset_only,
            'epochs':         args.epochs,
            'd_model':        args.d_model,
            'T':              args.T, 'H': H,
            'n_static':       len(static_cols),
            'n_past_obs':     len(past_obs_cols),
            'n_past_known':   len(past_kn_cols),
            'static_cols':    static_cols,
            'past_known_cols': past_kn_cols,
            'horizon_weights': h_w.tolist(),
        }, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
