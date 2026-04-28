"""
Static leakage attestation for the xgbmodel + Markowitz backtest.

Returns a structured dict that the dashboard renders in the
"无前视数据保证" (no-look-ahead data guarantee) section.

The attestation is checked by tracing every place the backtest reads data:

1. xgbmodel walk-forward training (xgbmodel/split.py):
   For every fold F, the structure is:
       train_window | purge=5d | val_window | embargo=2d | test_window
   The model that produces test predictions for any date `t ∈ test_window`
   is trained using rows where `trade_date ≤ train_end < t`. The 5-day purge
   prevents the (forward-shifted) target on the last train rows from
   overlapping with features in the val/test windows.

2. xgbmodel feature engineering (xgbmodel/features.py, cross_section.py):
   - All lag/rolling features use only past values (.shift(k≥0), .rolling(w))
     and are computed strictly on data ≤ trade_date.
   - Cross-sectional rank/demean is per trade_date — uses only that day's
     panel, never future days'.
   - Quarterly fundamentals (merge_fina_point_in_time): merge_asof with
     direction='backward' and left_on=trade_date, right_on=ann_date — so
     only fina rows whose ann_date ≤ trade_date are merged in.
   - Global indices are explicitly lagged 1 day (`<idx>_pct_chg_lag1`,
     `<idx>_vol_ratio_lag1`) to avoid using same-day Asia-Europe overlap.
   - Target uses `tgt.shift(-forward_window)` to put t+1's pct_chg into row t,
     then drops rows where target is NaN — so no target ever uses future
     data of the row it labels.

3. backtest/xgb_markowitz.py:
   - compute_rolling_sigma:
       df['pct_chg'].rolling(60, min_periods=20).std().shift(1)
     The .shift(1) makes σ_t depend only on pct_chg from t-window..t-1.
   - compute_rolling_adv: same pattern, .shift(1).
   - QP solver covariance panel:
       pf['pct_chg'].loc[:day].iloc[:-1].tail(cov_window)
     `.loc[:day]` includes day; `.iloc[:-1]` drops day; .tail(60) → only
     past 60 trading days strictly before `day`.
   - Entry filter: skip stocks whose abs(pct_chg) on `day` ≥ limit_pct.
     pct_chg(day) is known by the close of `day`; entry happens at close.
   - Entry price: prices.loc[day, 'close'] — the close of `day`. This is
     the standard "close-to-close" execution model: place the order at or
     near close after seeing the day's data; the realised return is from
     close(day) → close(day+1).
   - Exit logic: TP / SL trigger from day's high / low; horizon exits at
     close. All values are intra-day on the actual exit date — no future
     bars consulted.
   - Markowitz QP weights w only depend on (mu, Σ) computed from data
     ≤ close(day); they do NOT use day+1+ information.

The execution model is a clean, no-leakage simulation. The only optimistic
assumption is that orders can be filled at exactly the day's close — a
realistic A-share approximation for a small portfolio (10 names, 5% ADV
cap), but a non-trivial assumption for very large NAVs.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'stock_data'


def _check_data_freshness() -> dict:
    """Inventory: latest available date per data source.

    Used both to validate inference inputs and to surface what's missing/stale
    in the dashboard.
    """
    import pandas as pd

    out = {'sources': []}

    # 1. Per-stock OHLCV
    for sub in ('sh', 'sz'):
        folder = DATA_DIR / sub
        if not folder.exists():
            continue
        files = list(folder.glob('*.csv'))[:60]
        last_dates = []
        for f in files:
            try:
                df = pd.read_csv(f, usecols=['trade_date'])
                last_dates.append(int(str(df['trade_date'].astype(str).max())))
            except Exception:
                continue
        if last_dates:
            out['sources'].append({
                'name':  f'stock_data/{sub} OHLCV',
                'desc':  f'{("沪市" if sub=="sh" else "深市")}日线 OHLCV (按股一文件)',
                'latest_date': str(max(last_dates)),
                'min_in_sample': str(min(last_dates)),
                'samples': len(last_dates),
            })

    # 2. Per-date data files
    for prefix, folder, desc in [
        ('daily_basic',  DATA_DIR / 'daily_basic', '估值因子: 换手率/PE/PB/市值'),
        ('moneyflow',    DATA_DIR / 'moneyflow',   '小/中/大/特大单 资金流向'),
        ('block_trade',  DATA_DIR / 'block_trade', '大宗交易'),
        ('stk_limit',    DATA_DIR / 'stk_limit',   '涨跌停价格'),
    ]:
        if not folder.exists():
            continue
        files = sorted([f for f in os.listdir(folder)
                        if f.startswith(prefix + '_') and f.endswith('.csv')])
        if files:
            latest = files[-1].replace(prefix + '_', '').replace('.csv', '')
            out['sources'].append({
                'name': f'stock_data/{prefix}/',
                'desc': desc,
                'latest_date': latest,
                'samples': len(files),
            })

    # 3. CSI300 + index dailybasic
    try:
        fp = next((DATA_DIR / 'index' / 'idx_factor_pro').glob('000300*.csv'))
        df = pd.read_csv(fp, usecols=['trade_date'])
        out['sources'].append({
            'name': 'index/idx_factor_pro/000300_SH',
            'desc': '沪深300 技术因子 (BIAS/RSI/KDJ/...)',
            'latest_date': str(df['trade_date'].astype(str).max()),
        })
    except StopIteration:
        pass

    # 4. Global indices
    for fp in sorted((DATA_DIR / 'index' / 'index_global').glob('*.csv')):
        try:
            df = pd.read_csv(fp, usecols=['trade_date'])
            out['sources'].append({
                'name': f'index/index_global/{fp.stem}',
                'desc': f'{fp.stem} 全球指数 (滞后1日)',
                'latest_date': str(df['trade_date'].astype(str).max()),
            })
        except Exception:
            continue

    # 5. Fina indicator coverage
    fina_dir = DATA_DIR / 'fina_indicator'
    fina_files = list(fina_dir.glob('*.csv'))
    if fina_files:
        recent_anns = []
        for fp in fina_files[:50]:
            try:
                df = pd.read_csv(fp, usecols=['ann_date'])
                recent_anns.append(int(str(df['ann_date'].astype(str).max())))
            except Exception:
                continue
        out['sources'].append({
            'name': 'fina_indicator/',
            'desc': f'季报基本面 (forward-fill, {len(fina_files)} 只股)',
            'latest_date': str(max(recent_anns)) if recent_anns else 'unknown',
            'samples': len(fina_files),
        })

    # 6. Latest in OOF test predictions
    test_csv = DATA_DIR / 'models' / 'xgb_preds' / 'test.csv'
    if test_csv.exists():
        try:
            df = pd.read_csv(test_csv, usecols=['trade_date'])
            out['sources'].append({
                'name': 'models/xgb_preds/test.csv',
                'desc': 'XGB walk-forward OOF 预测 (用于回测的 μ)',
                'latest_date': str(df['trade_date'].astype(str).max()),
            })
        except Exception:
            pass

    return out


# Static narrative (matches the docstring above)
LEAKAGE_GUARANTEES = [
    {
        'topic': 'XGBoost 模型训练 (walk-forward)',
        'guarantee':
            '每折结构：训练窗 → purge gap (5日) → 验证窗 → embargo gap (2日) → 测试窗。'
            '某折测试日 t 的预测来自训练截止日严格早于 t 的模型；purge gap 防止训练集'
            '最后一行的 forward-shifted target 与验证集前几行的特征发生重叠泄露。',
        'evidence': 'xgbmodel/split.py:walk_forward_folds, train.py:_walk_forward',
    },
    {
        'topic': '特征工程',
        'guarantee':
            '所有滞后/滚动特征仅使用过去数据 (.shift(k≥0), .rolling(w))；'
            '横截面排名按当日全市场计算，不跨日；季报基本面通过 merge_asof '
            'direction=backward 仅合并 ann_date ≤ trade_date 的记录；'
            '全球指数显式滞后 1 日 (lag1)；目标 = pct_chg.shift(-1)，'
            '生成后立即丢弃 target=NaN 的行。',
        'evidence': 'xgbmodel/features.py, cross_section.py, data_loader.py:merge_fina_point_in_time',
    },
    {
        'topic': '回测 σ 与 ADV 估计',
        'guarantee':
            'σ_t = pct_chg.rolling(60).std().shift(1) — σ_t 仅依赖 t-60..t-1 的实际收益；'
            'ADV_t = amount.rolling(20).mean().shift(1) — 同理，只看 t-1 之前数据。',
        'evidence': 'backtest/xgb_markowitz.py:compute_rolling_sigma, compute_rolling_adv',
    },
    {
        'topic': 'QP 协方差矩阵',
        'guarantee':
            'pf[\'pct_chg\'].loc[:day].iloc[:-1].tail(60) — '
            '.loc[:day] 包括 day，.iloc[:-1] 去掉 day 当行，.tail(60) 取过去 60 个交易日。'
            'Σ 严格由 day 之前的数据估计，使用 Ledoit-Wolf 收缩。',
        'evidence': 'backtest/xgb_markowitz.py:run_backtest (QP solver branch)',
    },
    {
        'topic': '入场决策',
        'guarantee':
            '在 day 收盘后基于 pred(day)、σ(day)、Σ(day-1..day-60) 求解 QP 得到权重 w，'
            '以 close(day) 价格买入。pred(day) 由训练截止日 < day 的模型生成，'
            '其特征均为 day 当日及之前的可观测数据，不含未来信息。',
        'evidence': 'backtest/xgb_markowitz.py:run_backtest L405-L386',
    },
    {
        'topic': '出场决策',
        'guarantee':
            '止盈/止损以出场日的 high/low 触发；强平在第 5 日收盘卖出。'
            '所有判定值均为出场日当日 (历史已知)，不消费未来 bar。',
        'evidence': 'backtest/xgb_markowitz.py:run_backtest L324-L353',
    },
    {
        'topic': '执行假设',
        'guarantee':
            '唯一乐观假设：订单可在 day 收盘价精确成交。这是 close-to-close 执行模型，'
            '不是数据泄露。对小规模组合 (top-K=10, ADV 5% 上限) 在 A 股是合理近似；'
            '对极大规模 NAV 需引入更激进的市场冲击模型。',
        'evidence': 'backtest/xgb_markowitz.py:run_backtest 入场使用 close 字段',
    },
]


def build_audit() -> dict:
    """Top-level entry: return both the static guarantees and dynamic data
    freshness inventory."""
    return {
        'guarantees':       LEAKAGE_GUARANTEES,
        'data_freshness':   _check_data_freshness(),
    }


if __name__ == '__main__':
    import json
    print(json.dumps(build_audit(), ensure_ascii=False, indent=2))
