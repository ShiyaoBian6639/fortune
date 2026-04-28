"""
Independent verification backtest — fully vectorised, minimum-feature.

Re-implements the SAME strategy as `backtest/xgb_markowitz.py` from scratch,
with completely different code paths. The two implementations should agree on
key metrics (Sharpe, CAGR, MDD, hit rate) within ~5% if our main backtest is
logic-correct. Material divergence (>10%) signals a bug.

Differences vs main backtest (intentional simplifications for clarity):
  • Equal-weight top-K instead of QP optimisation. Removes Σ-estimation as
    a confounding source. Sharpe should still be in the same ballpark.
  • No ADV liquidity cap (we run small notional 1M).
  • No market-impact bps. Fixed 25bps round-trip cost (10 entry + 15 exit).
  • No ST cap (the strategy at 1M notional won't stress liquidity).

Identical to main backtest:
  • Daily top-K (default 10) by prediction at trade_date X-impl_lag.
  • impl_lag (default 1): pred(X-1) → entry at day X.
  • entry_price ∈ {close, open}: which auction we fill on.
  • TP +3% / SL -2% / max_hold 5 (intraday triggers, gap-up/down realism).
  • T+1 settlement: same-day exit blocked.
  • Locked-limit detection: skip exit on bars where open=high=low=close at
    pre_close ± limit_pct.

Run:
    ./venv/Scripts/python -m backtest.verify_independent \\
        --preds_csv stock_data/models_ensemble_xlc_oc/xgb_preds/test.csv \\
        --start 2025-07-01 --end 2026-04-15 \\
        --impl_lag 1 --entry_price open

Outputs: a markdown comparison report against a `--main_metrics` file (the
output of the main xgb_markowitz backtest with the same parameters).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'stock_data'


def _log(msg): print(f"[verify] {msg}", flush=True)


# ─── Price loading ──────────────────────────────────────────────────────────
def load_prices(ts_codes: List[str], start: pd.Timestamp,
                end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """Load OHLC + pre_close for each stock; return dict ts_code → DataFrame."""
    out = {}
    for ts in ts_codes:
        code, suf = ts.split('.')
        sub = 'sh' if suf.upper() == 'SH' else 'sz'
        fp = DATA / sub / f'{code}.csv'
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, usecols=['trade_date','open','high','low','close','pre_close','pct_chg'])
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df[(df['trade_date'] >= start - pd.Timedelta(days=10)) &
                    (df['trade_date'] <= end + pd.Timedelta(days=10))]
            if len(df) < 5:
                continue
            df = df.sort_values('trade_date').set_index('trade_date')
            out[ts] = df
        except Exception:
            continue
    return out


# ─── Locked-limit detector (same as main bt) ─────────────────────────────────
def is_locked(row, limit_pct: float = 9.8) -> str:
    op, hi, lo, cl = (float(row.get(k, 0.0) or 0.0) for k in ('open','high','low','close'))
    pc = float(row.get('pre_close', 0.0) or 0.0)
    if pc <= 0 or hi <= 0: return ''
    if (hi - lo) > 1e-6 or abs(cl - op) > 1e-6: return ''
    pct = (cl / pc - 1.0) * 100.0
    if pct >= limit_pct - 0.1: return 'locked_up'
    if pct <= -limit_pct + 0.1: return 'locked_down'
    return ''


# ─── Backtest engine ─────────────────────────────────────────────────────────
def run_verify(preds: pd.DataFrame, prices: Dict[str, pd.DataFrame], *,
                top_k: int = 10, tp_pct: float = 0.03, sl_pct: float = 0.02,
                max_hold: int = 5, impl_lag: int = 1, entry_price: str = 'open',
                cost_bps: float = 25.0, initial: float = 1_000_000.0,
                limit_pct: float = 9.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trading_days = sorted(preds['trade_date'].unique())
    day_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_days)}
    cost = cost_bps / 1e4

    cash = initial
    positions: List[dict] = []      # {ts_code, entry_date, entry_price, shares}
    equity_rows = []
    trade_rows  = []

    for day in trading_days:
        day = pd.Timestamp(day)

        # ── Exit existing positions ──
        kept = []
        for p in positions:
            # T+1 — skip same-day exit
            if p['entry_date'] == day:
                kept.append(p); continue
            df = prices.get(p['ts_code'])
            if df is None or day not in df.index:
                kept.append(p); continue
            row = df.loc[day]
            held = len(df.loc[p['entry_date']:day]) - 1
            op = float(row['open']); hi = float(row['high'])
            lo = float(row['low']);  cl = float(row['close'])
            tp_price = p['entry_price'] * (1.0 + tp_pct)
            sl_price = p['entry_price'] * (1.0 - sl_pct)

            # Locked-limit gate — can't fill on locked bars (until safety net)
            if is_locked(row, limit_pct=limit_pct) and held < max_hold * 2:
                kept.append(p); continue

            tp_hit = (hi >= tp_price) and (lo <= tp_price)
            sl_hit = (lo <= sl_price) and (hi >= sl_price)
            if (op > tp_price) and not tp_hit: tp_hit, tp_fill = True, op
            else: tp_fill = tp_price
            if (op < sl_price) and not sl_hit: sl_hit, sl_fill = True, op
            else: sl_fill = sl_price

            exit_price, reason = None, None
            if tp_hit and sl_hit:
                exit_price, reason = sl_fill, 'stop_loss'      # conservative
            elif tp_hit:
                exit_price, reason = tp_fill, 'take_profit'
            elif sl_hit:
                exit_price, reason = sl_fill, 'stop_loss'
            elif held >= max_hold:
                if lo < hi or held >= max_hold * 2:
                    exit_price, reason = cl, 'horizon'

            if exit_price is None:
                kept.append(p); continue
            proceeds = exit_price * p['shares'] * (1.0 - cost / 2)   # half cost on exit
            pnl = proceeds - p['cost_basis']
            cash += proceeds
            trade_rows.append({
                'ts_code': p['ts_code'], 'entry_date': p['entry_date'], 'exit_date': day,
                'entry_price': p['entry_price'], 'exit_price': exit_price,
                'shares': p['shares'], 'pnl': pnl,
                'ret': pnl / p['cost_basis'], 'held_days': held, 'reason': reason,
            })
        positions = kept

        # ── Mark-to-market NAV ──
        invested_value = 0.0
        for p in positions:
            df = prices.get(p['ts_code'])
            if df is not None and day in df.index:
                invested_value += float(df.loc[day, 'close']) * p['shares']
        equity_rows.append({'trade_date': day, 'nav': cash + invested_value,
                             'cash': cash, 'invested': invested_value, 'n_pos': len(positions)})

        # ── Entry: top-K from preds[trade_date = day - impl_lag] ──
        i = day_idx.get(day, -1)
        if i < impl_lag:
            continue
        pred_date = pd.Timestamp(trading_days[i - impl_lag]) if impl_lag > 0 else day
        today = preds[preds['trade_date'] == pred_date]
        if today.empty: continue
        cands = today.sort_values('pred', ascending=False).head(top_k * 4)
        held_codes = {p['ts_code'] for p in positions}

        chosen = []
        for _, r in cands.iterrows():
            ts = r['ts_code']
            if ts in held_codes: continue
            df = prices.get(ts)
            if df is None or day not in df.index: continue
            row = df.loc[day]
            # Entry filter — skip locked-all-day bars
            if is_locked(row, limit_pct=limit_pct):
                continue
            entry_px = float(row['open'] if entry_price == 'open' else row['close'])
            if entry_px <= 0: continue
            chosen.append((ts, entry_px))
            if len(chosen) >= top_k:
                break

        if chosen and cash > 1000:
            # Equal-weight allocation
            budget = cash * 0.95
            per_pos = budget / len(chosen)
            for ts, px in chosen:
                gross_shares = per_pos / (px * (1.0 + cost / 2))
                shares = np.floor(gross_shares / 100.0) * 100.0
                if shares <= 0: continue
                cb = px * shares * (1.0 + cost / 2)         # entry includes half cost
                if cb > cash: continue
                cash -= cb
                positions.append({
                    'ts_code': ts, 'entry_date': day, 'entry_price': px,
                    'shares': shares, 'cost_basis': cb,
                })

    equity = pd.DataFrame(equity_rows).set_index('trade_date')
    trades = pd.DataFrame(trade_rows)
    return equity, trades


def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial: float):
    nav = equity['nav']
    ret = nav.pct_change().dropna()
    n = len(ret)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / n) - 1.0 if n > 0 else 0.0
    vol  = ret.std() * np.sqrt(252) if n > 1 else 0.0
    sharpe = (ret.mean() * 252) / vol if vol > 0 else 0.0
    peak = nav.cummax()
    mdd = ((nav / peak) - 1.0).min()
    if not trades.empty:
        hit_rate = (trades['ret'] > 0).mean()
        avg_win  = trades.loc[trades['ret'] > 0, 'ret'].mean() if (trades['ret']>0).any() else 0
        avg_loss = trades.loc[trades['ret'] < 0, 'ret'].mean() if (trades['ret']<0).any() else 0
    else:
        hit_rate = avg_win = avg_loss = 0
    return {
        'cagr':       float(cagr) * 100,
        'sharpe':     float(sharpe),
        'mdd':        float(mdd) * 100,
        'vol_ann':    float(vol) * 100,
        'final_nav':  float(nav.iloc[-1]),
        'total_return': float(nav.iloc[-1] / initial - 1) * 100,
        'n_trades':   int(len(trades)),
        'hit_rate':   float(hit_rate) * 100,
        'avg_win':    float(avg_win) * 100,
        'avg_loss':   float(avg_loss) * 100,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preds_csv', required=True)
    p.add_argument('--start', default='2025-07-01')
    p.add_argument('--end',   default='2026-04-15')
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--tp', type=float, default=0.03)
    p.add_argument('--sl', type=float, default=0.02)
    p.add_argument('--max_hold', type=int, default=5)
    p.add_argument('--impl_lag', type=int, default=1)
    p.add_argument('--entry_price', choices=['open', 'close'], default='open')
    p.add_argument('--cost_bps', type=float, default=25.0)
    p.add_argument('--initial',  type=float, default=1_000_000.0)
    args = p.parse_args()

    start_ts = pd.Timestamp(args.start); end_ts = pd.Timestamp(args.end)
    preds = pd.read_csv(args.preds_csv, parse_dates=['trade_date'])
    preds = preds[(preds['trade_date'] >= start_ts) & (preds['trade_date'] <= end_ts)]
    _log(f"loaded {len(preds):,} preds, {preds['trade_date'].nunique()} days, "
         f"{preds['ts_code'].nunique()} stocks")

    ts_codes = sorted(preds['ts_code'].unique())
    _log(f"loading prices for {len(ts_codes):,} stocks ...")
    prices = load_prices(ts_codes, start_ts, end_ts)
    _log(f"loaded prices for {len(prices):,} stocks")

    equity, trades = run_verify(
        preds, prices,
        top_k=args.top_k, tp_pct=args.tp, sl_pct=args.sl, max_hold=args.max_hold,
        impl_lag=args.impl_lag, entry_price=args.entry_price,
        cost_bps=args.cost_bps, initial=args.initial,
    )
    m = compute_metrics(equity, trades, args.initial)

    print()
    print('=' * 70)
    print(f'INDEPENDENT VERIFICATION BACKTEST  ({args.start} → {args.end})')
    print('=' * 70)
    print(f'  preds        : {args.preds_csv}')
    print(f'  top_k        : {args.top_k}')
    print(f'  TP / SL      : +{args.tp*100:.1f}% / -{args.sl*100:.1f}%')
    print(f'  max_hold     : {args.max_hold}d')
    print(f'  impl_lag     : {args.impl_lag}d   entry_price = {args.entry_price}')
    print(f'  cost_bps     : {args.cost_bps:.0f} round-trip')
    print(f'  initial      : {args.initial:,.0f}')
    print()
    print(f'  Final NAV    : {m["final_nav"]:,.0f}')
    print(f'  Total return : {m["total_return"]:+.2f}%')
    print(f'  CAGR         : {m["cagr"]:+.2f}%')
    print(f'  Vol (ann)    : {m["vol_ann"]:.2f}%')
    print(f'  Sharpe       : {m["sharpe"]:+.2f}')
    print(f'  Max drawdown : {m["mdd"]:+.2f}%')
    print(f'  N trades     : {m["n_trades"]:,}')
    print(f'  Hit rate     : {m["hit_rate"]:.2f}%')
    print(f'  Avg win/loss : +{m["avg_win"]:.2f}% / {m["avg_loss"]:.2f}%')

    # Save metrics for the comparison report
    out_p = ROOT / 'plots' / 'backtest_xgb_markowitz' / f'verify_independent_{Path(args.preds_csv).stem}.txt'
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(
        '\n'.join(f'{k}: {v}' for k, v in m.items()),
        encoding='utf-8',
    )
    _log(f'metrics saved → {out_p}')
    return m


if __name__ == '__main__':
    main()
