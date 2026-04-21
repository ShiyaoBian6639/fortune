"""
Visualization for the deeptime pipeline.

8 plots saved to plots/deeptime_results/:
  1. training_history.png
  2. pred_vs_actual_horizons.png
  3. rolling_ic_heatmap.png
  4. vsn_feature_importance.png
  5. sector_ic_analysis.png
  6. temporal_attention_heatmap.png
  7. error_distribution_regime.png
  8. sector_cross_attention.png
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from .config import FORWARD_WINDOWS, NUM_HORIZONS, get_horizon_name, DT_OBSERVED_PAST_COLUMNS, DT_KNOWN_FUTURE_COLUMNS

_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'plots', 'deeptime_results',
)


def _save(fig, name: str):
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 1: Training history ─────────────────────────────────────────────────

def plot_training_history(history: dict, best_epoch: int = None):
    train_loss = history.get('train_loss', [])
    if not train_loss:
        print("  [skip] training history: no data (run --predict_only without saved history)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('deeptime — Training Dynamics', fontsize=14, fontweight='bold')
    epochs = range(1, len(train_loss) + 1)

    def _safe(lst):
        return [v if v == v else None for v in lst]  # replace nan with None

    ax = axes[0, 0]
    ax.plot(epochs, _safe(history.get('train_loss', [])), label='Train Loss', color='tab:blue')
    ax.plot(epochs, _safe(history.get('val_loss',   [])), label='Val Loss',   color='tab:orange')
    if best_epoch:
        ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best ep {best_epoch}')
    ax.set(title='Huber Loss', xlabel='Epoch', ylabel='Loss')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, _safe(history.get('train_ic', [])), label='Train IC', color='tab:blue')
    ax.plot(epochs, _safe(history.get('val_ic',   [])), label='Val IC',   color='tab:orange')
    if best_epoch:
        ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7)
    ax.axhline(0, color='grey', linestyle=':')
    ax.set(title='Mean IC (Spearman)', xlabel='Epoch', ylabel='IC')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, _safe(history.get('lr', [])), color='tab:purple')
    ax.set(title='Learning Rate Schedule', xlabel='Epoch', ylabel='LR')
    ax.set_yscale('log'); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    gn = history.get('grad_norm', [])
    finite_gn = [v for v in gn if v == v and np.isfinite(v)]
    ax.plot(epochs, _safe(gn), color='tab:red', alpha=0.7, label='Grad norm')
    if finite_gn:
        ax.axhline(np.mean(finite_gn), color='darkred', linestyle='--', alpha=0.6,
                   label=f'Mean={np.mean(finite_gn):.2f}')
    bad = sum(1 for v in gn if not (v == v and np.isfinite(v)))
    ax.set(title=f'Gradient Norm  ({bad}/{len(gn)} inf/nan epochs skipped)', xlabel='Epoch', ylabel='||g||')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, 'training_history.png')


# ─── Plot 2: Pred vs actual per horizon ───────────────────────────────────────

def plot_pred_vs_actual(
    all_preds:   np.ndarray,   # (N, 5)
    all_targets: np.ndarray,   # (N, 5)
    sector_ids:  np.ndarray = None,
    metrics:     dict = None,
):
    H = min(NUM_HORIZONS, all_preds.shape[1])
    fig, axes = plt.subplots(1, H, figsize=(4 * H, 4))
    fig.suptitle('deeptime — Predicted vs Actual Excess Return (Test Set)', fontsize=13, fontweight='bold')

    cmap = plt.cm.get_cmap('tab20', 32)

    for h in range(H):
        ax   = axes[h] if H > 1 else axes
        p    = all_preds[:, h]
        t    = all_targets[:, h]
        valid = np.isfinite(p) & np.isfinite(t)
        p, t  = p[valid], t[valid]

        # Sample for speed
        if len(p) > 5000:
            idx = np.random.choice(len(p), 5000, replace=False)
            p, t = p[idx], t[idx]
            s_ids = sector_ids[valid][idx] if sector_ids is not None else None
        else:
            s_ids = sector_ids[valid] if sector_ids is not None else None

        c = cmap(s_ids % 32) if s_ids is not None else 'steelblue'
        ax.scatter(t, p, c=c, alpha=0.3, s=4, rasterized=True)

        # OLS regression line
        z = np.polyfit(t, p, 1)
        xline = np.linspace(t.min(), t.max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), 'r-', linewidth=1.5)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)

        hn   = get_horizon_name(h)
        ic   = metrics.get(f'ic_{hn}',   0) if metrics else 0
        mae  = metrics.get(f'mae_{hn}',  0) if metrics else 0
        hr   = metrics.get(f'hr_{hn}',   0) if metrics else 0
        ax.set(title=f'Day +{FORWARD_WINDOWS[h]}', xlabel='Actual (%)', ylabel='Predicted (%)')
        ax.text(0.05, 0.95, f'IC={ic:.3f}\nMAE={mae:.2f}%\nHR={hr:.2%}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.grid(alpha=0.2)

    fig.tight_layout()
    _save(fig, 'pred_vs_actual_horizons.png')


# ─── Plot 3: Rolling IC heatmap ────────────────────────────────────────────────

def plot_rolling_ic_heatmap(
    monthly_ic:  Dict[str, Dict[str, float]],  # {YYYYMM: {horizon_name: ic}}
    regimes:     Optional[Dict[str, str]] = None,  # {YYYYMM: 'bull'|'bear'}
):
    if not monthly_ic:
        print("  [skip] rolling IC heatmap: no data")
        return

    months   = sorted(monthly_ic.keys())
    horizons = [get_horizon_name(h) for h in range(NUM_HORIZONS)]
    data     = np.array([[monthly_ic[m].get(hn, np.nan) for hn in horizons] for m in months])

    fig, ax = plt.subplots(figsize=(max(12, len(months) * 0.4), 4))
    fig.suptitle('deeptime — Monthly Rolling IC by Horizon', fontsize=13, fontweight='bold')

    im = ax.imshow(
        data.T, aspect='auto', cmap='RdYlGn', vmin=-0.1, vmax=0.1,
        interpolation='nearest',
    )
    ax.set_yticks(range(len(horizons)))
    ax.set_yticklabels(horizons)
    step = max(1, len(months) // 24)
    ax.set_xticks(range(0, len(months), step))
    ax.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45, ha='right', fontsize=7)
    plt.colorbar(im, ax=ax, label='IC')

    # Regime background
    if regimes:
        for xi, m in enumerate(months):
            r = regimes.get(m, 'bull')
            color = 'blue' if r == 'bull' else 'red'
            ax.axvline(xi, color=color, alpha=0.05, linewidth=4)

    fig.tight_layout()
    _save(fig, 'rolling_ic_heatmap.png')


# ─── Plot 4: VSN feature importance ───────────────────────────────────────────

def plot_vsn_feature_importance(
    enc_vsn_weights: np.ndarray,   # (N, seq_len, n_past)
    dec_vsn_weights: np.ndarray,   # (N, max_fw,  n_future)
    top_n: int = 30,
):
    # Average over batch and time
    enc_mean = enc_vsn_weights.mean(axis=(0, 1))   # (n_past,)
    dec_mean = dec_vsn_weights.mean(axis=(0, 1))   # (n_future,)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('deeptime — VSN Feature Importance', fontsize=13, fontweight='bold')

    # Color by feature category
    _CATEGORIES = {
        'OHLCV/Price': ['returns', 'log_returns', 'high_low', 'close_open', 'gap', 'body_size', 'shadow'],
        'Technical':   ['rsi', 'macd', 'stoch', 'williams', 'cci', 'adx', 'plus_di', 'minus_di', 'atr', 'obv', 'bb_', 'roc', 'momentum', 'sma', 'ema', 'vol_sma', 'volatil', 'dist_from', 'cross', 'above_sma', 'trend', 'consecutive', 'price_vs', 'w_bottom', 'm_top', 'pattern'],
        'Valuation':   ['pe', 'pb', 'ps', 'dv_', 'total_mv', 'circ_mv'],
        'Turnover':    ['turnover', 'volume_ratio', 'float_ratio', 'free_ratio'],
        'Market ctx':  ['csi300', 'csi500', 'sse50', 'gem_', 'csi1000', 'sse_', 'dji', 'hsi', 'ixic', 'n225', 'spx'],
        'Index memb':  ['is_csi', 'is_sse', 'csi300_weight'],
        'Fundamentals':FINA_COLS,
        'Block trade': ['block_vol', 'block_amt', 'block_count', 'block_buy'],
        'Money flow':  ['net_lg', 'net_elg', 'net_sm', 'net_md'],
        'Limits':      ['is_limit', 'up_limit', 'down_limit'],
        'Calendar':    ['dow_', 'dom_', 'month_', 'woy_', 'doy_', 'quarter_', 'is_monday', 'is_friday', 'is_month', 'is_year', 'is_pre_holiday', 'is_post_holiday', 'holiday', 'is_january', 'is_december', 'is_earnings', 'is_weak'],
    }
    CAT_COLORS = plt.cm.tab20.colors

    def _get_color(name, categories):
        for ci, (cat, kws) in enumerate(categories.items()):
            for kw in kws:
                if kw in name:
                    return CAT_COLORS[ci % len(CAT_COLORS)], cat
        return 'grey', 'Other'

    # Encoder
    feat_names = DT_OBSERVED_PAST_COLUMNS[:len(enc_mean)]
    top_idx    = np.argsort(enc_mean)[-top_n:][::-1]
    colors     = [_get_color(feat_names[i], _CATEGORIES)[0] for i in top_idx]
    ax1.barh([feat_names[i] for i in top_idx], enc_mean[top_idx], color=colors)
    ax1.set(title=f'Top {top_n} Encoder (Past-Observed) Features', xlabel='Mean VSN Weight')
    ax1.invert_yaxis(); ax1.grid(alpha=0.3, axis='x')

    # Decoder
    dec_names = DT_KNOWN_FUTURE_COLUMNS[:len(dec_mean)]
    top_dec   = np.argsort(dec_mean)[-min(20, len(dec_mean)):][::-1]
    colors2   = [_get_color(dec_names[i], _CATEGORIES)[0] for i in top_dec]
    ax2.barh([dec_names[i] for i in top_dec], dec_mean[top_dec], color=colors2)
    ax2.set(title='Top Known-Future Decoder Features', xlabel='Mean VSN Weight')
    ax2.invert_yaxis(); ax2.grid(alpha=0.3, axis='x')

    fig.tight_layout()
    _save(fig, 'vsn_feature_importance.png')


# ─── Plot 5: Sector IC analysis ───────────────────────────────────────────────

def plot_sector_ic_analysis(
    sector_metrics: Dict[str, Dict],  # {sector_name: {'ic': float, 'hr': float, 'n': int}}
):
    if not sector_metrics:
        print("  [skip] sector IC: no data")
        return

    sectors = list(sector_metrics.keys())
    ics  = [sector_metrics[s].get('ic',   0) for s in sectors]
    hrs  = [sector_metrics[s].get('hr', 0.5) for s in sectors]
    ns   = [sector_metrics[s].get('n',     0) for s in sectors]

    order = np.argsort(ics)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(sectors)*0.3)))
    fig.suptitle('deeptime — IC Analysis by Sector', fontsize=13, fontweight='bold')

    colors = ['green' if ic >= 0 else 'red' for ic in [ics[i] for i in order]]
    ax1.barh([sectors[i] for i in order], [ics[i] for i in order], color=colors, alpha=0.7)
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set(title='IC by Sector', xlabel='IC (mean)')
    ax1.grid(alpha=0.3, axis='x')

    colors2 = ['steelblue'] * len(sectors)
    ax2.barh([sectors[i] for i in order], [hrs[i] for i in order], color='steelblue', alpha=0.7)
    ax2.axvline(0.5, color='red', linewidth=1, linestyle='--', label='50% (random)')
    ax2.set(title='Hit Rate by Sector', xlabel='Directional Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3, axis='x')

    fig.tight_layout()
    _save(fig, 'sector_ic_analysis.png')


# ─── Plot 6: Temporal attention heatmap ───────────────────────────────────────

def plot_temporal_attention_heatmap(
    attn_weights: np.ndarray,  # (N, T_total, T_total)
    seq_len: int = 30,
):
    avg_attn = attn_weights.mean(axis=0)   # (T, T)
    T = avg_attn.shape[0]
    max_fw = T - seq_len  # 5 decoder steps

    attn_std = float(avg_attn.std())
    # Ideal: std >> 1/T (uniform); near-uniform means model ignores time structure
    uniform_std = 1.0 / T * np.sqrt(T - 1)  # std of uniform distribution over T
    selectivity = attn_std / (uniform_std + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'deeptime — Temporal Attention  (selectivity={selectivity:.2f}, 1.0=uniform)',
                 fontsize=13, fontweight='bold')

    # Left: full heatmap
    ax = axes[0]
    im = ax.imshow(avg_attn, cmap='Blues', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax)
    ax.axhline(seq_len - 0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.axvline(seq_len - 0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.8)
    # Annotate regions
    ax.text(seq_len/2, seq_len/2, 'Encoder\nself-attn', ha='center', va='center',
            fontsize=9, color='red', alpha=0.7)
    ax.text(seq_len/2, seq_len + max_fw/2, 'Decoder\n→ Encoder', ha='center', va='center',
            fontsize=9, color='red', alpha=0.7)
    ax.set(title='Full attention matrix', xlabel='Key (past ← → future)',
           ylabel='Query')
    # Tick labels: -seq_len to +max_fw
    step = max(1, T // 10)
    ticks = list(range(0, T, step))
    labels = [f't{i-seq_len+1:+d}' if i < seq_len else f't+{i-seq_len+1}' for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)

    # Right: decoder→encoder attention (how each future step uses past)
    ax = axes[1]
    dec_enc = avg_attn[seq_len:, :seq_len]   # (max_fw, seq_len)
    im2 = ax.imshow(dec_enc, cmap='Blues', aspect='auto', vmin=0)
    plt.colorbar(im2, ax=ax)
    ax.set_yticks(range(max_fw))
    ax.set_yticklabels([f't+{i+1}' for i in range(max_fw)], fontsize=9)
    enc_ticks = list(range(0, seq_len, max(1, seq_len // 8)))
    ax.set_xticks(enc_ticks)
    ax.set_xticklabels([f't{i-seq_len+1:+d}' for i in enc_ticks], fontsize=8)
    ax.set(title='Decoder → Encoder attention\n(which past steps each future horizon uses)',
           xlabel='Past timestep (key)', ylabel='Prediction horizon (query)')

    # Note if attention is near-uniform
    if selectivity < 2.0:
        fig.text(0.5, 0.01,
                 f'NOTE: Attention is near-uniform (selectivity={selectivity:.2f}). '
                 'Model is not strongly leveraging temporal order in the past window.',
                 ha='center', fontsize=9, color='darkred',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, 'temporal_attention_heatmap.png')


# ─── Plot 7: Error distribution by regime ─────────────────────────────────────

def plot_error_distribution(
    all_preds:   np.ndarray,   # (N, 5)
    all_targets: np.ndarray,   # (N, 5)
    anchor_dates: np.ndarray,  # (N,)
    sector_ids:  np.ndarray,   # (N,)
    is_csi300:   np.ndarray = None,   # (N,) bool
):
    errors = all_preds[:, 0] - all_targets[:, 0]   # day+1 errors

    # Regime: dates >= 20220101 and < 20230601 = bear; else = bull (approximate)
    is_bear = (anchor_dates >= 20220101) & (anchor_dates < 20230601)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('deeptime — Error Distribution Analysis (Day+1)', fontsize=13, fontweight='bold')

    def _kde(ax, data, label, color):
        from scipy.stats import gaussian_kde
        valid = data[np.isfinite(data)]
        if len(valid) < 10:
            return
        kde  = gaussian_kde(valid, bw_method=0.3)
        xr   = np.linspace(np.percentile(valid, 1), np.percentile(valid, 99), 200)
        ax.plot(xr, kde(xr), label=label, color=color)

    ax = axes[0, 0]
    _kde(ax, errors[is_bear],  'Bear regime', 'red')
    _kde(ax, errors[~is_bear], 'Bull regime', 'green')
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set(title='Error by Market Regime', xlabel='Prediction Error (%)', ylabel='Density')
    ax.legend(); ax.grid(alpha=0.3)

    if is_csi300 is not None:
        ax = axes[0, 1]
        _kde(ax, errors[is_csi300.astype(bool)],  'CSI300 member',    'blue')
        _kde(ax, errors[~is_csi300.astype(bool)], 'Non-CSI300',       'orange')
        ax.axvline(0, color='k', linewidth=0.8)
        ax.set(title='Error by Index Membership', xlabel='Prediction Error (%)')
        ax.legend(); ax.grid(alpha=0.3)

    # Error by size decile
    ax = axes[1, 0]
    decile_mae = []
    for d in range(10):
        mask = sector_ids == d
        if mask.sum() > 10:
            decile_mae.append(float(np.nanmean(np.abs(errors[mask]))))
        else:
            decile_mae.append(np.nan)
    ax.bar(range(10), decile_mae, color='steelblue', alpha=0.7)
    ax.set(title='MAE by Size Decile (0=smallest)', xlabel='Size Decile', ylabel='MAE (%)')
    ax.grid(alpha=0.3, axis='y')

    # Histogram of errors
    ax = axes[1, 1]
    valid_err = errors[np.isfinite(errors)]
    ax.hist(valid_err, bins=100, color='steelblue', alpha=0.7, density=True, label='Errors')
    xr = np.linspace(np.percentile(valid_err, 0.5), np.percentile(valid_err, 99.5), 200)
    mu, sig = np.nanmean(valid_err), np.nanstd(valid_err)
    from scipy.stats import norm
    ax.plot(xr, norm.pdf(xr, mu, sig), 'r-', label=f'N({mu:.2f},{sig:.2f})', linewidth=2)
    ax.set(title='Error Histogram', xlabel='Error (%)', ylabel='Density')
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, 'error_distribution_regime.png')


# ─── Plot 8: Sector cross-attention pattern ────────────────────────────────────

def plot_sector_cross_attention(
    sector_attn_weights: np.ndarray,   # (B, B) stock×stock GAT weights
    sector_ids: np.ndarray,            # (B,) sector index per stock in batch
    sector_names: List[str] = None,
):
    """
    SectorGAT returns (B, B) stock-to-stock attention.
    Aggregate by sector to produce a sector×sector heatmap.
    """
    if sector_attn_weights.ndim != 2:
        return
    B = sector_attn_weights.shape[0]
    # Handle both (B,B) and old (B,K) formats
    if sector_attn_weights.shape[1] != B:
        # Old format (B, K) — treat as before using sector_ids as row index
        K = sector_attn_weights.shape[1]
        active_secs = sorted(set(int(s) % K for s in sector_ids))
        n = len(active_secs)
        agg = np.zeros((n, n))
        cnts = np.zeros((n, n))
        s2r = {s: i for i, s in enumerate(active_secs)}
        for b in range(len(sector_attn_weights)):
            r = s2r.get(int(sector_ids[b]) % K, -1)
            if r < 0: continue
            for j, s in enumerate(active_secs):
                if s < K:
                    agg[r, j] += sector_attn_weights[b, s]
                    cnts[r, j] += 1
        cnts = cnts.clip(min=1); agg /= cnts
        names = [sector_names[s] if sector_names and s < len(sector_names) else f'S{s}' for s in active_secs]
    else:
        # New (B, B) format from SectorGAT
        unique_secs = sorted(set(int(s) for s in sector_ids))
        n = len(unique_secs)
        if n < 2:
            return
        s2r = {s: i for i, s in enumerate(unique_secs)}
        agg = np.zeros((n, n)); cnts = np.zeros((n, n))
        for bi in range(B):
            for bj in range(B):
                ri = s2r.get(int(sector_ids[bi]), -1)
                rj = s2r.get(int(sector_ids[bj]), -1)
                if ri >= 0 and rj >= 0:
                    agg[ri, rj] += sector_attn_weights[bi, bj]
                    cnts[ri, rj] += 1
        cnts = cnts.clip(min=1); agg /= cnts
        names = [sector_names[s] if sector_names and s < len(sector_names) else f'S{s}' for s in unique_secs]

    row_std = agg.std(axis=1)
    is_diag = np.trace(agg) / (agg.sum() + 1e-8) > 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n * 0.5 + 2)))
    fig.suptitle(
        'deeptime SectorGAT — Stock attention aggregated by Sector\n'
        + ('Intra-sector dominant (diagonal)' if is_diag else 'Cross-sector attention active'),
        fontsize=11, fontweight='bold')

    ax = axes[0]
    im = ax.imshow(agg, cmap='Blues', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax, label='Mean Attention Weight')
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    ax.set(title='Sector × Sector aggregated GAT', xlabel='Attended', ylabel='Source')

    ax = axes[1]
    ax.barh(names, row_std, color=['steelblue' if v > row_std.mean() else 'lightsteelblue' for v in row_std])
    ax.set(title='Row selectivity (higher = more selective)', xlabel='Std')
    ax.grid(alpha=0.3, axis='x')

    fig.tight_layout()
    _save(fig, 'sector_cross_attention.png')


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def plot_all(
    history:      dict,
    test_preds:   np.ndarray,
    test_targets: np.ndarray,
    test_dates:   np.ndarray,
    test_sectors: np.ndarray,
    test_metrics: dict,
    model=None,
    sector_names: List[str] = None,
):
    """Generate all 8 plots."""
    print("\nGenerating visualizations...")

    # 1. Training history
    plot_training_history(history, best_epoch=history.get('best_epoch'))

    # 2. Pred vs actual
    plot_pred_vs_actual(test_preds, test_targets, test_sectors, test_metrics)

    # 3. Rolling IC heatmap
    try:
        monthly_ic = _compute_monthly_ic(test_preds, test_targets, test_dates)
        plot_rolling_ic_heatmap(monthly_ic)
    except Exception as e:
        print(f"  [skip] rolling IC: {e}")

    # 4. VSN feature importance
    if model is not None and model._enc_vsn_weights is not None:
        try:
            enc_w = model._enc_vsn_weights.cpu().numpy()
            dec_w = model._dec_vsn_weights.cpu().numpy()
            plot_vsn_feature_importance(enc_w, dec_w)
        except Exception as e:
            print(f"  [skip] VSN importance: {e}")

    # 5. Sector IC
    try:
        sec_metrics = _compute_sector_metrics(test_preds, test_targets, test_sectors, sector_names)
        plot_sector_ic_analysis(sec_metrics)
    except Exception as e:
        print(f"  [skip] sector IC: {e}")

    # 6. Temporal attention
    if model is not None and model._attn_weights is not None:
        try:
            attn_w = model._attn_weights.cpu().numpy()
            plot_temporal_attention_heatmap(attn_w)
        except Exception as e:
            print(f"  [skip] temporal attention: {e}")

    # 7. Error distribution
    try:
        plot_error_distribution(test_preds, test_targets, test_dates, test_sectors)
    except Exception as e:
        print(f"  [skip] error distribution: {e}")

    # 8. Sector cross-attention
    if model is not None and model._sector_attn is not None:
        try:
            sec_attn = model._sector_attn.cpu().numpy()
            plot_sector_cross_attention(sec_attn, test_sectors, sector_names)
        except Exception as e:
            print(f"  [skip] sector cross-attention: {e}")

    print(f"  All plots saved to {_OUT_DIR}")


def _compute_monthly_ic(preds, targets, dates):
    """Aggregate IC by calendar month."""
    months = {}
    for i in range(len(preds)):
        d = int(dates[i])
        m = str(d)[:6]   # YYYYMM
        if m not in months:
            months[m] = {'preds': [], 'targets': []}
        months[m]['preds'].append(preds[i])
        months[m]['targets'].append(targets[i])

    from .training import compute_ic
    monthly_ic = {}
    for m, dd in months.items():
        p = np.array(dd['preds'])   # (N, H)
        t = np.array(dd['targets'])
        monthly_ic[m] = {get_horizon_name(h): compute_ic(p[:, h], t[:, h]) for h in range(NUM_HORIZONS)}
    return monthly_ic


def _compute_sector_metrics(preds, targets, sector_ids, sector_names):
    from .training import compute_ic, compute_hit_rate
    metrics = {}
    unique_secs = np.unique(sector_ids)
    for s in unique_secs:
        mask = sector_ids == s
        if mask.sum() < 10:
            continue
        p = preds[mask, 0]
        t = targets[mask, 0]
        valid = np.isfinite(p) & np.isfinite(t)
        if valid.sum() < 5:
            continue
        name = sector_names[s] if sector_names and s < len(sector_names) else f'Sector {s}'
        metrics[name] = {
            'ic': compute_ic(p[valid], t[valid]),
            'hr': compute_hit_rate(p[valid], t[valid]),
            'n':  int(mask.sum()),
        }
    return metrics


# FINA_COLS needed for color coding
FINA_COLS = ['roe', 'roa', 'grossprofit', 'netprofit', 'current_ratio', 'quick_ratio',
             'debt_to', 'assets_yoy', 'equity_yoy', 'op_yoy', 'ebt_yoy', 'eps']
