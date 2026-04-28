"""
Ensemble + cross-model comparison.

Loads OOF predictions from multiple stock_data/models_<name>/xgb_preds/test.csv,
joins on (ts_code, trade_date), and produces:
  1. simple-average ensemble    → models_ensemble_mean/xgb_preds/test.csv
  2. rank-average ensemble      → models_ensemble_rankavg/xgb_preds/test.csv
  3. per-model metric table     → printed + JSON
  4. residual correlation       → matrix between model predictions

Run:
    ./venv/Scripts/python -m model_compare.ensemble \
        --models xgb_default lightgbm catboost transformer_reg tft

Outputs feed directly into the dashboard's "Model selection" section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'


def load_test(model_name: str) -> pd.DataFrame:
    """Load OOF test predictions for a single model."""
    p = DATA / f'models_{model_name}' / 'xgb_preds' / 'test.csv'
    if not p.exists():
        # Fall back to canonical xgb_default at stock_data/models/
        if model_name == 'xgb_default':
            p = DATA / 'models' / 'xgb_preds' / 'test.csv'
    if not p.exists():
        raise FileNotFoundError(f"missing predictions: {p}")
    df = pd.read_csv(p, parse_dates=['trade_date'])
    df = df[['ts_code', 'trade_date', 'pred', 'target']].rename(
        columns={'pred': f'pred_{model_name}'})
    return df


def join_models(model_names: List[str]) -> pd.DataFrame:
    """Inner-join all models on (ts_code, trade_date). Loses rows that
    aren't predicted by every model (typically only at fold edges)."""
    out = None
    for n in model_names:
        df = load_test(n)
        out = df if out is None else out.merge(
            df.drop(columns=['target']), on=['ts_code', 'trade_date'], how='inner')
    return out


def compute_per_model_metrics(joined: pd.DataFrame, model_names: List[str]) -> dict:
    """Per-model rank IC (daily Spearman), Pearson, hit rate, RMSE."""
    from scipy.stats import spearmanr
    out = {}
    for n in model_names:
        col = f'pred_{n}'
        # Pearson + RMSE
        p = joined[col].values; y = joined['target'].values
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        pe   = float(np.corrcoef(p, y)[0, 1]) if len(p) > 1 else float('nan')
        # Hit rate (sign agreement)
        hr   = float(((p > 0) == (y > 0)).mean())
        # Daily rank IC
        def _ic(g):
            if len(g) < 10: return np.nan
            return spearmanr(g[col], g['target']).correlation
        daily_ic = joined.groupby('trade_date').apply(_ic).dropna()
        ic_mean = float(daily_ic.mean())
        ic_std  = float(daily_ic.std())
        icir    = ic_mean / ic_std if ic_std > 0 else float('nan')
        ic_pos  = float((daily_ic > 0).mean())
        out[n] = {
            'rmse':           rmse,
            'pearson_ic':     pe,
            'hit_rate':       hr,
            'rank_ic_mean':   ic_mean,
            'rank_ic_std':    ic_std,
            'icir':           icir,
            'ic_pos_ratio':   ic_pos,
            'n_obs':          int(len(p)),
        }
    return out


def build_mean_ensemble(joined: pd.DataFrame, model_names: List[str]) -> pd.DataFrame:
    """Simple per-row average of the model predictions."""
    cols = [f'pred_{n}' for n in model_names]
    out = joined[['ts_code', 'trade_date', 'target']].copy()
    out['pred'] = joined[cols].mean(axis=1).astype('float32')
    return out[['ts_code', 'trade_date', 'pred', 'target']]


def build_rankavg_ensemble(joined: pd.DataFrame, model_names: List[str]) -> pd.DataFrame:
    """Rank-average ensemble: convert each model's predictions to
    cross-sectional rank (within each trade_date), average ranks, then
    rank again as the final 'pred'."""
    df = joined.copy()
    cols = [f'pred_{n}' for n in model_names]
    rank_cols = []
    for c in cols:
        rank_cols.append(c + '_rank')
        df[rank_cols[-1]] = (df.groupby('trade_date')[c]
                                  .rank(pct=True, method='average').astype('float32'))
    df['rank_avg'] = df[rank_cols].mean(axis=1).astype('float32')
    df['pred']     = (df.groupby('trade_date')['rank_avg']
                            .rank(pct=True, method='average').astype('float32'))
    return df[['ts_code', 'trade_date', 'pred', 'target']]


def save_ensemble(df: pd.DataFrame, name: str) -> Path:
    out_dir = DATA / f'models_{name}' / 'xgb_preds'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'test.csv'
    df.to_csv(out_path, index=False)
    return out_path


def correlation_matrix(joined: pd.DataFrame, model_names: List[str]) -> pd.DataFrame:
    cols = [f'pred_{n}' for n in model_names]
    return joined[cols].corr().rename(
        index=lambda c: c.replace('pred_', ''),
        columns=lambda c: c.replace('pred_', ''),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True,
                   help='engine names to combine (e.g. xgb_default lightgbm tft)')
    args = p.parse_args()

    print(f"[ensemble] joining {len(args.models)} model predictions ...")
    joined = join_models(args.models)
    print(f"[ensemble] joined {len(joined):,} rows × {len(args.models)} models")
    if joined.empty:
        raise SystemExit("[ensemble] no overlapping rows — check that each "
                          "model has run walk-forward on the same panel.")

    metrics = compute_per_model_metrics(joined, args.models)
    print("\n=== Per-model metrics on joined OOF set ===")
    print(f"{'model':22} {'IC mean':>10} {'ICIR':>8} {'IC+%':>7} {'HR':>7} {'RMSE':>8}")
    for n in args.models:
        m = metrics[n]
        print(f"{n:22} {m['rank_ic_mean']:>+10.4f} {m['icir']:>8.2f} "
              f"{m['ic_pos_ratio']*100:>6.1f}% {m['hit_rate']*100:>6.1f}% "
              f"{m['rmse']:>8.4f}")

    # Build ensembles
    mean_df = build_mean_ensemble(joined, args.models)
    save_ensemble(mean_df, 'ensemble_mean')
    rank_df = build_rankavg_ensemble(joined, args.models)
    save_ensemble(rank_df, 'ensemble_rankavg')

    # Add ensemble metrics
    ens_metrics = compute_per_model_metrics(
        rank_df.assign(pred_ensemble_rankavg=rank_df['pred'])
                .merge(mean_df.assign(pred_ensemble_mean=mean_df['pred'])
                              [['ts_code','trade_date','pred_ensemble_mean']],
                       on=['ts_code','trade_date'], how='inner'),
        ['ensemble_mean', 'ensemble_rankavg'])

    print("\n=== Ensemble metrics ===")
    for n, m in ens_metrics.items():
        print(f"{n:22} {m['rank_ic_mean']:>+10.4f} {m['icir']:>8.2f} "
              f"{m['ic_pos_ratio']*100:>6.1f}% {m['hit_rate']*100:>6.1f}% "
              f"{m['rmse']:>8.4f}")

    # Correlation matrix
    corr = correlation_matrix(joined, args.models)
    print("\n=== Pred-vs-pred correlation ===")
    print(corr.round(3).to_string())

    # Persist comparison artefacts
    out_path = DATA / 'models_ensemble_comparison.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'models':   args.models,
            'metrics':  metrics,
            'ensembles': ens_metrics,
            'correlation': corr.values.tolist(),
            'corr_index':  list(corr.index),
            'n_obs':    int(len(joined)),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[ensemble] wrote {out_path}")


if __name__ == '__main__':
    main()
