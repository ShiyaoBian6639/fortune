"""
Build the predictions-focused dashboard.

Scope: everything about the model's signal — live next-day forecast, the
suggested portfolio, the top-30 best/worst predictions with 30-day feature
trends to visually validate, top-50 feature importance, and OOS accuracy
diagnostics.

Reads:
  - dashboard/data.json          (built by dashboard.build, has OOS metrics + live signal)
  - dashboard/combined_data.json (built by dashboard.build_combined, has live + feature catalog)
  - stock_data/models/xgb_pct_chg.meta.json
  - stock_data/sh, sz/*.csv      (for 30-day feature trends)
  - stock_data/daily_basic       (for turnover_rate_f trend)

Writes:
  - dashboard/predictions_data.json
  - dashboard/index_predictions.html  (single-file, JSON inlined)

Run:
  ./venv/Scripts/python -m dashboard.build_predictions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent

from dashboard.feature_catalog  import CATALOG, GROUPS, lookup
from dashboard.feature_trends   import (build_trends_for, build_slim_trends_all,
                                          feature_descriptions, SLIM_FEATURE_LABELS)
from dashboard.live_prediction  import build_live_payload


def _log(msg: str) -> None:
    print(f"[predictions] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live_preds_csv', default=None)
    ap.add_argument('--n_trends', type=int, default=30,
                    help='How many top-prediction stocks to compute 30-day feature trends for')
    args = ap.parse_args()

    # 1. OOS prediction-side metrics (uses dashboard.build's data.json output)
    data_p = OUT / 'data.json'
    if not data_p.exists():
        raise SystemExit(f"Missing {data_p} — run `dashboard.build` first.")
    with open(data_p, 'r', encoding='utf-8') as f:
        D = json.load(f)

    # 2. Full meta for top-50 importance
    meta_p = ROOT / 'stock_data' / 'models' / 'xgb_pct_chg.meta.json'
    with open(meta_p, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # 3. Live next-day forecast — three views matching combined dashboard
    preds_csv = Path(args.live_preds_csv) if args.live_preds_csv \
                else (ROOT / 'stock_predictions_xgb.csv')
    live = live_with_st = live_expanded = None
    if preds_csv.exists():
        try:
            live = build_live_payload(preds_csv, top_k=10,
                                       candidate_pool=40, max_st_per_day=4)
        except Exception as e:
            _log(f"WARNING live (backtest-faithful): {e}")
        try:
            live_with_st = build_live_payload(preds_csv, top_k=10,
                                                candidate_pool=40, max_st_per_day=-1)
        except Exception as e:
            _log(f"WARNING live (uncapped): {e}")
        try:
            live_expanded = build_live_payload(preds_csv, top_k=10,
                                                 candidate_pool=200, max_st_per_day=4)
        except Exception as e:
            _log(f"WARNING live_expanded: {e}")
    else:
        _log(f"WARNING {preds_csv} missing — skipping live")

    # 4. 30-day feature trends for the top-N predictions and bottom-N
    feature_date = D.get('feature_date')
    top_codes = [r['ts_code'] for r in D['live_topn']['top'][:args.n_trends]]
    bot_codes = [r['ts_code'] for r in D['live_topn']['bottom'][:args.n_trends]]
    _log(f"computing 30-day trends for {len(top_codes)} top + {len(bot_codes)} bottom")
    trends_top    = build_trends_for(top_codes, feature_date, window_days=30)
    trends_bottom = build_trends_for(bot_codes, feature_date, window_days=30)

    # Slim 30-day feature trends for ALL stocks in live_table — powers the
    # "全部预测" dropdown so any stock's recent feature path can be charted.
    all_codes = [r['ts_code'] for r in D.get('live_table', [])]
    _log(f"computing slim trends for all {len(all_codes)} stocks...")
    slim_trends = build_slim_trends_all(all_codes, feature_date, window_days=30)
    _log(f"slim trends built for {len(slim_trends)} stocks")

    # 4b. Model-comparison artefacts (if model_compare.ensemble has been run)
    cmp_p = ROOT / 'stock_data' / 'models_ensemble_comparison.json'
    model_comparison = None
    if cmp_p.exists():
        with open(cmp_p, 'r', encoding='utf-8') as f:
            model_comparison = json.load(f)
        _log(f"loaded model comparison: {len(model_comparison.get('models', []))} models")

    # 4c. Per-model backtest comparison (if model_compare.run_backtests has been run)
    bt_p = ROOT / 'stock_data' / 'models_backtest_comparison.json'
    backtest_comparison = None
    if bt_p.exists():
        with open(bt_p, 'r', encoding='utf-8') as f:
            backtest_comparison = json.load(f)
        n_b = len(backtest_comparison.get('baselines', []))
        n_t = len(backtest_comparison.get('tuned', []))
        _log(f"loaded backtest comparison: {n_b} baselines, {n_t} tuned")

    # 4d-pre. Per-engine SINGLE-HORIZON predictions (6 engines, t+1 only).
    # Used by the new "per-engine top picks" section + the engine selector
    # dropdown on the 全部预测 table. Also computes a rank-average ensemble
    # across all 6 engines as the default view.
    sh_engines = [
        ('xgb',            'xgb (canonical)', 'stock_predictions_xgb.csv'),
        ('xgb_default',    'xgb_default',     'stock_predictions_xgb_default.csv'),
        ('xgb_shallow',    'xgb_shallow',     'stock_predictions_xgb_shallow.csv'),
        ('xgb_strong_reg', 'xgb_strong_reg',  'stock_predictions_xgb_strong_reg.csv'),
        ('xgb_deep',       'xgb_deep',        'stock_predictions_xgb_deep.csv'),
        ('lightgbm',       'lightgbm',        'stock_predictions_lightgbm.csv'),
        ('catboost',       'catboost',        'stock_predictions_catboost.csv'),
    ]
    name_map = {r['ts_code']: r.get('name', '') for r in D.get('live_table', [])}
    per_engine = {}
    pred_frame = None
    for key, label, fname in sh_engines:
        fp = ROOT / fname
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        df['name']       = df['ts_code'].map(name_map).fillna('')
        df = df.sort_values('pred_pct_chg_next', ascending=False).reset_index(drop=True)
        df['pred_pct_chg_next'] = df['pred_pct_chg_next'].astype('float32').round(4)
        per_engine[key] = {
            'engine':       key,
            'engine_label': label,
            'feature_date': df.iloc[0]['trade_date'] if len(df) else '',
            'n_stocks':     len(df),
            'top30': df.head(30)[['ts_code','name','pred_pct_chg_next']].to_dict('records'),
            'bot30': df.tail(30)[::-1][['ts_code','name','pred_pct_chg_next']].to_dict('records'),
            'rows':  df[['ts_code','name','pred_pct_chg_next']].to_dict('records'),
        }
        # Rank prep for ensemble: assign rank 1 = best (highest pred), N = worst
        ranks = df.set_index('ts_code')['pred_pct_chg_next'].rank(ascending=False, method='average')
        if pred_frame is None:
            pred_frame = pd.DataFrame(index=ranks.index)
        pred_frame[f'{key}_rank'] = ranks
        pred_frame[f'{key}_pred'] = df.set_index('ts_code')['pred_pct_chg_next']

    ensemble = None
    consensus = None
    if pred_frame is not None and not pred_frame.empty:
        rank_cols = [c for c in pred_frame.columns if c.endswith('_rank')]
        # exclude xgb canonical (which is a copy of xgb_default) — avoids double-vote
        rank_cols_uniq = [c for c in rank_cols if c != 'xgb_rank']
        avg_rank = pred_frame[rank_cols_uniq].mean(axis=1)
        # Map avg_rank back to a "score" — higher = better. Center so scores
        # are interpretable as percentile-equivalents.
        N = len(pred_frame)
        ens_score = (N + 1 - avg_rank) / N    # 1 = best, ~0 = worst
        # Also compute a simple mean of predictions (in % units) for display
        pred_cols = [c for c in pred_frame.columns
                     if c.endswith('_pred') and c != 'xgb_pred']
        mean_pred = pred_frame[pred_cols].mean(axis=1)
        std_pred  = pred_frame[pred_cols].std(axis=1)

        ens_df = pd.DataFrame({
            'ts_code':         pred_frame.index,
            'pred_pct_chg_next': mean_pred.round(4),
            'pred_std':        std_pred.round(4),
            'ensemble_score':  ens_score.round(4),
            'avg_rank':        avg_rank.round(2),
        }).reset_index(drop=True)
        ens_df['name'] = ens_df['ts_code'].map(name_map).fillna('')
        ens_df = ens_df.sort_values('pred_pct_chg_next', ascending=False).reset_index(drop=True)
        ensemble = {
            'engine':       'ensemble',
            'engine_label': '集成 (6引擎均值)',
            'feature_date': next(iter(per_engine.values()))['feature_date'] if per_engine else '',
            'n_stocks':     len(ens_df),
            'n_engines':    len(rank_cols_uniq),
            'top30': ens_df.head(30)[['ts_code','name','pred_pct_chg_next','pred_std','ensemble_score']].to_dict('records'),
            'bot30': ens_df.tail(30)[::-1][['ts_code','name','pred_pct_chg_next','pred_std','ensemble_score']].to_dict('records'),
            'rows':  ens_df[['ts_code','name','pred_pct_chg_next','pred_std','ensemble_score']].to_dict('records'),
        }

        # Cross-engine consensus: count votes for top-K across engines
        top_K = 30
        votes = {}
        for key in per_engine:
            if key == 'xgb':   # canonical alias of xgb_default
                continue
            tops = set(r['ts_code'] for r in per_engine[key]['top30'][:top_K])
            for ts in tops:
                votes[ts] = votes.get(ts, 0) + 1
        # Build consensus list: stocks ranked by vote count, tie-break by mean_pred
        cons_rows = []
        for ts, v in sorted(votes.items(), key=lambda kv: (-kv[1], -mean_pred.get(kv[0], 0))):
            cons_rows.append({
                'ts_code':  ts,
                'name':     name_map.get(ts, ''),
                'votes':    v,
                'n_engines': len(rank_cols_uniq),
                'mean_pred': float(mean_pred.get(ts, 0)),
                'avg_rank':  float(avg_rank.get(ts, len(ens_df))),
            })
        consensus = {
            'top_k_window':  top_K,
            'n_engines':     len(rank_cols_uniq),
            'feature_date':  next(iter(per_engine.values()))['feature_date'] if per_engine else '',
            'rows':          cons_rows[:60],
        }

    _log(f"per_engine: {len(per_engine)} engines  ensemble={'yes' if ensemble else 'no'}  "
         f"consensus={consensus['rows'][:1] if consensus else 'no'}")

    # 4d. Multi-horizon predictions (xgb / lightgbm / catboost × t+1..t+5)
    multihorizon = {}
    mh_engines = ['xgb', 'lightgbm', 'catboost']
    mh_engine_labels = {'xgb': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}
    for eng in mh_engines:
        mh_path = ROOT / f'stock_predictions_multihorizon_{eng}.csv'
        if not mh_path.exists():
            continue
        mh_df = pd.read_csv(mh_path)
        # Convert trade_date to ISO string
        mh_df['trade_date'] = pd.to_datetime(mh_df['trade_date']).dt.strftime('%Y-%m-%d')
        # Round preds to 4 decimals for compactness
        for c in ('pred_d1','pred_d2','pred_d3','pred_d4','pred_d5','mean5','max5','slope'):
            if c in mh_df.columns:
                mh_df[c] = mh_df[c].astype('float32').round(4)
        # Attach Chinese name from D['live_table'] if available
        name_map = {r['ts_code']: r.get('name', '') for r in D.get('live_table', [])}
        mh_df['name'] = mh_df['ts_code'].map(name_map).fillna('')
        # Pre-compute alternate sortings
        mh_records = mh_df.to_dict('records')
        multihorizon[eng] = {
            'engine':         eng,
            'engine_label':   mh_engine_labels[eng],
            'feature_date':   mh_df['trade_date'].iloc[0] if len(mh_df) else '',
            'rows':           mh_records,
            'n_stocks':       len(mh_df),
            'top30_by_mean5': mh_df.sort_values('mean5', ascending=False).head(30)['ts_code'].tolist(),
            'top30_by_max5':  mh_df.sort_values('max5',  ascending=False).head(30)['ts_code'].tolist(),
            'top30_by_slope': mh_df.sort_values('slope', ascending=False).head(30)['ts_code'].tolist(),
            'top30_by_d1':    mh_df.sort_values('pred_d1', ascending=False).head(30)['ts_code'].tolist(),
        }
        _log(f"multihorizon {eng}: {len(mh_df):,} stocks  feature_date={multihorizon[eng]['feature_date']}")
    if not multihorizon:
        multihorizon = None

    payload = {
        'predictions': {
            'feature_date':     feature_date,
            'prediction_date':  D.get('prediction_date'),
            'n_live_stocks':    D.get('n_live_stocks'),
            'overall_metrics':  D['overall_metrics'],
            'daily_rank_ic':    D['daily_rank_ic'],
            'decile':           D['decile_analysis'],
            'live_topn':        D['live_topn'],
            'live_table':       D.get('live_table', []),
            'live_distribution': D.get('live_distribution'),
            'group_by_province': D.get('group_by_province'),
            'group_by_board':    D.get('group_by_board'),
            'group_by_exchange': D.get('group_by_exchange'),
        },
        'model_meta':       D.get('model_meta'),
        'feature_importance_top50': meta.get('feature_importance_top50', []),
        'feature_catalog': {
            'features':  [lookup(f) for f in (CATALOG.keys())],
            'groups':    [{'key': g[0], 'name_zh': g[1], 'desc': g[2]} for g in GROUPS],
        },
        'feature_descriptions': feature_descriptions(),
        'trends_top':           trends_top,
        'trends_bottom':        trends_bottom,
        'slim_trends':          slim_trends,
        'slim_feature_labels':  SLIM_FEATURE_LABELS,
        'live':                 live,
        'live_with_st':         live_with_st,
        'live_expanded':        live_expanded,
        'model_comparison':     model_comparison,
        'backtest_comparison':  backtest_comparison,
        'multihorizon':         multihorizon,
        'per_engine_singlehorizon': per_engine,
        'ensemble_singlehorizon':   ensemble,
        'consensus_singlehorizon':  consensus,
    }

    out_json = OUT / 'predictions_data.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
    _log(f"wrote {out_json}  ({out_json.stat().st_size / 1e6:.2f} MB)")

    # 5. Self-contained single-file
    src_html = OUT / 'predictions.html'
    if not src_html.exists():
        _log(f"WARNING {src_html} missing — skipping single-file bundle")
        return
    html = src_html.read_text(encoding='utf-8')
    embedded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    needle = '</head>'
    inject = f'<script id="predictions-data-embed">\nwindow.PREDICTIONS_DATA = {embedded};\n</script>\n'
    bundled = html.replace(needle, inject + needle, 1)
    bundled_p = OUT / 'index_predictions.html'
    bundled_p.write_text(bundled, encoding='utf-8')
    _log(f"wrote {bundled_p}  ({bundled_p.stat().st_size / 1e6:.2f} MB)")


if __name__ == '__main__':
    main()
