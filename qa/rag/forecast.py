"""
ForecastProvider — surface modelfactory's daily predictions to the QA layer.

When a user asks "未来走势" / "前景" / "预测" / "涨跌概率" etc. for a
resolved ts_code, the context_builder appends a "模型预测" block built
by this class. The block contains:

  - Vote summary across 30 model-engine combos (3 engines × 10 labels):
    how many predict UP, mean up-probability across the binary classifiers.
  - Averaged regression forecasts (next-day return %, 5-day return %).
  - Top-N feature importance from a canonical model (XGB regression on
    next-day close-to-close return).
  - Recent values of those top features for the queried ts_code, so the
    LLM has concrete corroborating data alongside the abstract prediction.

The actual phrasing + disclaimer ("基于模型预测，仅供参考，不构成投资建议")
is rendered by ``qa.rag.context_builder._fmt_forecast``; this class only
provides the underlying data.

Inputs (filesystem):
  stock_data/modelfactory/live/latest_{task}__{engine}__{label}.csv
  stock_data/modelfactory/runs/_LATEST  → name of the latest run
  stock_data/modelfactory/runs/<run>/<task>/<engine>__<label>/feature_importance.csv
  stock_data/sh/{code}.csv  / stock_data/sz/{code}.csv  (OHLCV)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'stock_data'

_LIVE_FILE_RE = re.compile(
    r'^latest_(?P<task>regression|classification)__'
    r'(?P<engine>[a-z]+)__'
    r'(?P<label>.+)\.csv$'
)


def _stock_csv_path(ts_code: str) -> Path:
    code, suf = ts_code.split('.')
    sub = 'sh' if suf.upper() == 'SH' else 'sz'
    return DATA / sub / f'{code}.csv'


class ForecastProvider:
    def __init__(self,
                 live_dir: str | Path = DATA / 'modelfactory' / 'live',
                 runs_dir: str | Path = DATA / 'modelfactory' / 'runs',
                 canonical_label: str = 'r_close_close',
                 canonical_engine: str = 'xgb',
                 n_top_features: int = 3):
        self.live_dir = Path(live_dir)
        self.runs_dir = Path(runs_dir)
        self.canonical_label  = canonical_label
        self.canonical_engine = canonical_engine
        self.n_top_features   = n_top_features

        # frames[(task, engine, label)] -> DataFrame indexed by ts_code
        self.frames: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._top_features: list[tuple[str, float]] = []

        self._load_live()
        self._load_top_features()
        print(f"[forecast] loaded {len(self.frames):,} live model frames "
              f"covering {self._n_unique_ts_codes():,} ts_codes; "
              f"top-{n_top_features} features = "
              f"{[f for f, _ in self._top_features]}",
              flush=True)

    # ─── Loading ──────────────────────────────────────────────────────────
    def _load_live(self):
        if not self.live_dir.exists():
            print(f"[forecast] live dir missing: {self.live_dir}", flush=True)
            return
        for p in sorted(self.live_dir.glob('latest_*.csv')):
            m = _LIVE_FILE_RE.match(p.name)
            if not m: continue
            task, engine, label = m.group('task'), m.group('engine'), m.group('label')
            try:
                df = pd.read_csv(p, dtype={'ts_code': str})
            except Exception as e:
                print(f"[forecast] failed to load {p.name}: {e}", flush=True)
                continue
            if 'ts_code' not in df.columns:
                continue
            df = df.set_index('ts_code')
            self.frames[(task, engine, label)] = df

    def _candidate_runs(self) -> list[str]:
        """Return run dir names sorted newest-first. Honour _LATEST as
        a hint but fall through to other runs if it's missing the
        canonical artifact."""
        runs = sorted(
            (p.name for p in self.runs_dir.glob('*') if p.is_dir()),
            reverse=True,
        )
        latest_marker = self.runs_dir / '_LATEST'
        if latest_marker.exists():
            try:
                hinted = latest_marker.read_text().strip()
                if hinted in runs:
                    runs = [hinted] + [r for r in runs if r != hinted]
            except Exception:
                pass
        return runs

    def _load_top_features(self):
        # Load deeper than n_top_features so summary() can pick the
        # first N that are *renderable* per stock — many highly-ranked
        # features (cs_*, day_of_month, csi500_pb, idx_*) are
        # cross-sectional / macro and don't have meaningful per-stock
        # recent values.
        DEEP_K = 40
        for run in self._candidate_runs():
            fi_path = (self.runs_dir / run / 'regression'
                       / f'{self.canonical_engine}__{self.canonical_label}'
                       / 'feature_importance.csv')
            if not fi_path.exists():
                continue
            try:
                fi = pd.read_csv(fi_path)
                fi = fi.sort_values('importance', ascending=False)
                self._top_features = [
                    (str(r['feature']), float(r['importance']))
                    for _, r in fi.head(DEEP_K).iterrows()
                ]
                print(f"[forecast] top features loaded from run {run} "
                      f"(deep_k={DEEP_K})", flush=True)
                return
            except Exception as e:
                print(f"[forecast] failed to read {fi_path}: {e}",
                      flush=True)
        print(f"[forecast] no usable feature_importance.csv found "
              f"(searched {self.runs_dir})", flush=True)

    def _n_unique_ts_codes(self) -> int:
        seen = set()
        for df in self.frames.values():
            seen.update(df.index)
        return len(seen)

    # ─── Vote aggregation ─────────────────────────────────────────────────
    def vote_for(self, ts_code: str) -> dict:
        """Aggregate every loaded model's verdict for ``ts_code``.

        Returns a dict suitable for direct rendering into the context.
        ``covered=False`` means no live model file mentions this ts_code.
        """
        result = {
            'ts_code':          ts_code,
            'covered':          False,
            'trade_date':       None,
            'n_models_total':   0,
            'n_models_up':      0,
            'binary_up_prob':   None,
            'reg_next_day_pct': None,
            'reg_5d_pct':       None,
        }

        binary_probs:  list[float] = []
        reg_1d_preds:  list[float] = []
        reg_5d_preds:  list[float] = []
        n_total = 0
        n_up    = 0
        latest_date = None

        for (task, engine, label), df in self.frames.items():
            if ts_code not in df.index:
                continue
            row = df.loc[ts_code]
            # Some files have duplicate ts_code rows; take the first.
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            n_total += 1

            td = str(row.get('trade_date', '')).strip()
            if td and (latest_date is None or td > latest_date):
                latest_date = td

            if task == 'regression':
                pred = float(row['pred'])
                if pred > 0: n_up += 1
                if label == 'r_close_close':
                    reg_1d_preds.append(pred)
                elif label == '5d_close_close' or label == 'r_5d_close_close':
                    reg_5d_preds.append(pred)
            else:  # classification
                # proba_K columns: pick "up" semantics per label
                proba_cols = [c for c in df.columns if c.startswith('proba_')]
                if not proba_cols:
                    continue
                k = len(proba_cols)
                if label == 'c_binary':
                    p_up = float(row.get('proba_1', 0.0))
                    binary_probs.append(p_up)
                    if p_up >= 0.5: n_up += 1
                elif label == 'c_3class':
                    # class 0 down / 1 flat / 2 up — sum upper half = up
                    p_up = float(row.get('proba_2', 0.0))
                    if p_up >= 0.34: n_up += 1
                else:  # c_5class_quantile / c_5class_overnight
                    # top 2 quintiles count as up
                    pred_class = int(row.get('pred_class', 0))
                    if pred_class >= (k // 2):
                        n_up += 1

        if n_total == 0:
            return result

        result.update({
            'covered':          True,
            'trade_date':       latest_date,
            'n_models_total':   n_total,
            'n_models_up':      n_up,
            'binary_up_prob':   round(sum(binary_probs)/len(binary_probs), 3)
                                  if binary_probs else None,
            'reg_next_day_pct': round(sum(reg_1d_preds)/len(reg_1d_preds), 3)
                                  if reg_1d_preds else None,
            'reg_5d_pct':       round(sum(reg_5d_preds)/len(reg_5d_preds), 3)
                                  if reg_5d_preds else None,
        })
        return result

    # ─── Top features + recent values ─────────────────────────────────────
    def top_features(self) -> list[tuple[str, float]]:
        return list(self._top_features)

    def recent_indicator(self, ts_code: str, feat_name: str,
                          n_days: int = 14) -> list[tuple[str, float]]:
        """Return [(date_str, value)] for ``feat_name`` over the last
        ``n_days`` trading days. Empty list if not computable.

        Supports anything ``xgbmodel.features.compute_price_features``
        produces (rsi_14, macd, bbpct_20, atr_14_pct, momentum_*,
        ret_lag_*, parkinson_*, etc.) plus the raw OHLCV inputs.
        Cross-sectional features (cs_*) and macro/index features are
        out of scope here — caller falls back gracefully.
        """
        fp = _stock_csv_path(ts_code)
        if not fp.exists():
            return []
        try:
            df = pd.read_csv(fp, dtype={'trade_date': str})
        except Exception:
            return []
        if df.empty: return []
        df = df.sort_values('trade_date').reset_index(drop=True)
        # Cap the history we feed to compute_price_features — RSI / MACD
        # need ~30 lookback to converge; 60 is plenty.
        df = df.tail(60).reset_index(drop=True)

        # Try direct OHLCV column first (close, vol, pct_chg etc.)
        if feat_name in df.columns:
            tail = df.tail(n_days)
            return [(str(r['trade_date']), float(r[feat_name]))
                    for _, r in tail.iterrows()
                    if pd.notna(r[feat_name])]

        # Otherwise route through compute_price_features
        try:
            from xgbmodel.features import compute_price_features
        except Exception:
            return []
        try:
            feats = compute_price_features(df)
        except Exception:
            return []
        if feat_name not in feats.columns:
            return []
        # compute_price_features returns a DataFrame aligned with df rows;
        # attach the date column for output.
        out_df = pd.DataFrame({
            'trade_date': df['trade_date'],
            feat_name:    feats[feat_name],
        }).tail(n_days)
        return [(str(r['trade_date']), float(r[feat_name]))
                for _, r in out_df.iterrows()
                if pd.notna(r[feat_name])]

    # ─── Convenience: bundle for one stock ────────────────────────────────
    def summary(self, ts_code: str, n_days: int = 14,
                 max_features: int = 5,
                 max_with_values: int = 2) -> dict:
        """Bundle vote + top features for ``ts_code``.

        Returns up to ``max_features`` highest-importance features, of
        which up to ``max_with_values`` have recent per-stock values
        attached (when computable from OHLCV / compute_price_features).
        The rest are name + importance + short explanation only.

        For this codebase's actual model ranking the high-importance
        features are dominated by cross-sectional (`cs_*`), macro
        (`csi500_*`, `n225_*`, `idx_*`) and calendar (`dow`,
        `day_of_month`) signals — none of which decompose per stock.
        We surface those by name to anchor the LLM's commentary and
        backfill 1–2 per-stock TA features (rsi_14, momentum_20,
        etc.) further down the ranking that DO have recent values.
        """
        v = self.vote_for(ts_code)
        if not v['covered']:
            return {'ts_code': ts_code, 'covered': False}

        feats: list[dict] = []
        with_value_count = 0
        # First pass: top by importance, value if possible
        for name, importance in self._top_features:
            if len(feats) >= max_features: break
            entry = {
                'name':       name,
                'importance': importance,
                'kind':       _classify_feature(name),
                'desc':       _FEATURE_DESC.get(name, ''),
                'recent':     [],
            }
            if with_value_count < max_with_values:
                recent = self.recent_indicator(ts_code, name, n_days=n_days)
                if recent:
                    entry['recent'] = recent
                    with_value_count += 1
            feats.append(entry)

        # Second pass: if no per-stock-value features made it, scan
        # deeper for renderable TA fallbacks so the LLM has at least
        # one concrete recent series to anchor on.
        if with_value_count == 0:
            already = {f['name'] for f in feats}
            for ta in _TA_FALLBACKS:
                if ta in already: continue
                recent = self.recent_indicator(ts_code, ta, n_days=n_days)
                if not recent: continue
                feats.append({
                    'name':       ta,
                    'importance': None,        # not in top-K
                    'kind':       'price_ta',
                    'desc':       _FEATURE_DESC.get(ta, ''),
                    'recent':     recent,
                })
                with_value_count += 1
                if with_value_count >= max_with_values: break

        return {**v, 'top_features': feats}


# ─── Feature taxonomy + descriptions ─────────────────────────────────────
def _classify_feature(name: str) -> str:
    if name.startswith('cs_'):  return 'cross_section'
    if name.startswith('idx_'): return 'index_ta'
    if name.endswith('_pct_chg_lag1') or name in (
        'csi500_pb', 'csi500_pe_ttm', 'csi500_pct_chg', 'csi500_open',
        'csi500_close', 'csi500_turnover',
        'csi300_turnover', 'sse_close', 'sse_pe_ttm',
        'sse50_close', 'sse50_pb', 'sse50_turnover',
        'szse_turnover', 'chinext_turnover',
    ):                          return 'macro'
    if name.endswith('_vol_ratio_lag1'): return 'macro'
    if name in ('dow', 'day_of_month', 'month_of_year'): return 'calendar'
    return 'price_or_other'


_FEATURE_DESC = {
    'cs_market_breadth':  '全市场上涨股票占比',
    'cs_daily_dispersion': '当日横截面收益离散度',
    'dow':                '一周中的第几天 (0=周一)',
    'day_of_month':       '当月第几日',
    'csi500_pb':          '中证500指数市净率',
    'csi500_pct_chg':     '中证500指数涨跌幅',
    'csi500_pe_ttm':      '中证500指数滚动市盈率',
    'n225_pct_chg_lag1':  '日经225昨日涨跌幅',
    'hsi_pct_chg_lag1':   '恒生昨日涨跌幅',
    'spx_pct_chg_lag1':   '标普500昨日涨跌幅',
    'rsi_14':             '14日相对强弱指数',
    'macd':               '指数平滑异同移动平均',
    'momentum_20':        '20日动量 (%)',
    'bbpct_20':           '布林带 %B (20日)',
    'atr_14_pct':         '14日真实波动率 (%)',
    'close_ma_20_ratio':  '收盘价 / 20日均线',
    'hl_ratio':           '当日 (high - low) / close',
    'turnover_rate_f':    '换手率 (流通)',
    'up_limit_ratio':     '股价距涨停板距离',
    'down_limit_ratio':   '股价距跌停板距离',
}

_TA_FALLBACKS = [
    'rsi_14', 'momentum_20', 'momentum_5', 'bbpct_20', 'atr_14_pct',
    'close_ma_20_ratio', 'macd',
]


def _self_test():
    fp = ForecastProvider()
    for ts in ('600519.SH', '300750.SZ', '002594.SZ', '000001.SZ'):
        s = fp.summary(ts)
        print(f"\n=== {ts} ===")
        if not s.get('covered'):
            print("  not covered")
            continue
        print(f"  trade_date {s['trade_date']}  votes {s['n_models_up']}/{s['n_models_total']} up  "
              f"binary_up={s['binary_up_prob']}  next_day={s['reg_next_day_pct']}%  "
              f"5d={s['reg_5d_pct']}%")
        for f in s['top_features']:
            imp = f['importance']
            imp_str = f"imp={imp:.0f}" if imp is not None else "(fallback)"
            recent = f.get('recent') or []
            tail_str = ' → '.join(f"{v:.2f}" for _, v in recent[-5:]) if recent else '(no series)'
            print(f"  - {f['name']:<24} {imp_str:<12} kind={f['kind']:<14} "
                  f"recent[-5]: {tail_str}")


if __name__ == '__main__':
    _self_test()
