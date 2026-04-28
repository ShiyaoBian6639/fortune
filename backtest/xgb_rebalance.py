"""
Rebalance strategy: rolling top-K Markowitz portfolio with daily rebalance.

Key differences from `backtest/xgb_markowitz.py` (the top-K + TP/SL strategy):

  • **Open-time tradeability filter**: each morning, drop stocks whose open
    price gapped to the daily limit (≥ +9.5% locked-up = no sellers, ≤ -9.5%
    locked-down = panic). The selection ranks AVAILABLE stocks, not the
    full universe. Stocks held at limit-down get carried (can't sell) but
    are not allowed to enter as new positions.

  • **Rebalance, not full reset**: a stock that was top-10 yesterday and is
    still top-10 today STAYS in the portfolio. Only the deltas trade —
    drop-outs are sold, new entrants are bought, weight changes adjust
    shares. Daily turnover collapses from 100 % (full reset) to typically
    20-40 %, drastically cutting transaction costs.

  • **No TP/SL**: a position holds while it remains in the top-K. Exit
    triggers ONLY when a stock falls out of the daily ranking. The "buy 10
    today, sell all tomorrow, buy 10 again" pattern that bleeds 50%+/yr to
    transaction costs is eliminated.

  • **T+1 settlement still enforced**: positions opened today (entry_date
    == day) cannot be closed today.

  • **Same Markowitz QP weighting**: top-K available stocks → run the
    long-only mean-variance QP with Ledoit-Wolf shrunk Σ → target weights.

CLI matches xgb_markowitz.py where applicable (--solver, --max_st_per_day,
--impl_lag, --entry_price, --top_k, --preds_csv, --tag, --start/--end).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .xgb_markowitz import (
    load_predictions, load_price_panel, load_benchmark,
    compute_rolling_adv, compute_rolling_sigma,
    estimate_covariance, markowitz_qp_weights, markowitz_diagonal_weights,
    _is_locked_at_limit,
    compute_metrics, trade_summary,
    DATA_DIR, ROOT,
)


# ── ST detection helper (st_intervals is {ts_code: [(start, end, kind), ...]}) ──
def _is_st(ts_code: str, day: pd.Timestamp,
           st_intervals: Dict[str, list] = None) -> bool:
    if st_intervals is None:
        return False
    iv = st_intervals.get(ts_code)
    if not iv:
        return False
    day_str = day.strftime('%Y%m%d') if hasattr(day, 'strftime') else str(day)
    for start, end, _kind in iv:
        if start <= day_str <= end:
            return True
    return False


def load_st_intervals():
    """Build the ST {ts_code: [(start, end, kind), ...]} index from
    api/st_history.csv. Returns None if the roster isn't available."""
    try:
        from api.st_history import load_roster, build_daily_index
        roster = load_roster()
        return build_daily_index(roster)
    except Exception as e:
        print(f"[rebalance] WARNING: ST roster unavailable ({e})")
        return None


# ── Open-time tradeability helper ─────────────────────────────────────────────
def is_open_at_limit(row, limit_pct: float = 9.5) -> str:
    """Return 'limit_up_open' / 'limit_down_open' / '' based on the OPEN price.

    Tighter than the all-day locked-limit test: a stock that opens at +9.5%+
    is essentially un-buyable (no sellers willing to give up the gain). One
    that opens at -9.5% is in panic — buying may be possible but the spread
    is huge and the move has often already happened. We skip both directions
    on the BUY side.

    A stock that LATER hits limit during the day but opens normally is fine
    to buy at the open auction.
    """
    op = float(row.get('open',  0.0) or 0.0)
    pc = float(row.get('pre_close', 0.0) or 0.0)
    if op <= 0 or pc <= 0:
        return ''
    op_pct = (op / pc - 1.0) * 100.0
    if op_pct >= limit_pct:
        return 'limit_up_open'
    if op_pct <= -limit_pct:
        return 'limit_down_open'
    return ''


# ── Position record ───────────────────────────────────────────────────────────
class Position:
    __slots__ = ('ts_code', 'entry_date', 'entry_price', 'shares',
                 'cost_basis', 'cum_div')

    def __init__(self, ts_code, entry_date, entry_price, shares, cost_basis):
        self.ts_code     = ts_code
        self.entry_date  = entry_date
        self.entry_price = entry_price
        self.shares      = shares
        self.cost_basis  = cost_basis
        self.cum_div     = 0.0


# ── Rebalance backtest engine ─────────────────────────────────────────────────
def run_rebalance_backtest(
    preds:          pd.DataFrame,
    prices:         Dict[str, pd.DataFrame],
    sigmas:         Dict[str, pd.Series],
    advs:           Dict[str, pd.Series] = None,
    top_k:          int   = 10,
    entry_bps:      float = 10.0,
    exit_bps:       float = 15.0,
    initial:        float = 1_000_000.0,
    open_limit_pct: float = 9.5,
    max_pos_adv:    float = 0.05,
    nav_cap:        float = 0.0,
    impact_bps:     float = 0.0,
    solver:         str   = 'qp',
    cov_window:     int   = 60,
    risk_aversion:  float = 1.0,
    max_st_per_day: int   = -1,
    st_intervals:   Dict[str, list] = None,
    impl_lag:       int   = 1,
    entry_price:    str   = 'open',
    rebalance_threshold: float = 0.005,
    hold_buffer_k:  int   = 20,            # Stickiness buffer: a held stock
                                            # stays in the portfolio while it
                                            # is in top (top_k + hold_buffer_k)
                                            # of today's preds. Only sells
                                            # when it falls past that.
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling daily rebalance to a Markowitz QP portfolio of top-K available.

    Each trading day:
      1. EXIT phase: any held position whose ts_code is NOT in today's target
         is sold at open (subject to T+1 and locked-down protection).
      2. ENTRY phase: target = QP weights × (cash + invested) on top-K
         available. Existing holdings adjusted to target weight; new entrants
         bought; over-weight positions trimmed.

    `rebalance_threshold` (default 0.5%): minimum |target − current| / NAV
    fraction before triggering a trade for an existing holding. Avoids tiny
    nuisance trades that cost more than they help.
    """
    trading_days = sorted(preds['trade_date'].unique())
    day_index    = {pd.Timestamp(d): i for i, d in enumerate(trading_days)}
    print(f"[rebalance] simulating {len(trading_days)} trading days "
          f"({pd.Timestamp(trading_days[0]).date()} → "
          f"{pd.Timestamp(trading_days[-1]).date()})  "
          f"impl_lag={impl_lag} entry={entry_price}")

    cash = initial
    positions: List[Position] = []
    equity_rows = []
    trade_rows  = []

    ec = entry_bps / 1e4
    xc = exit_bps  / 1e4

    def _exec_price(row, is_buy: bool) -> float:
        """Pick fill price per the entry_price convention."""
        op = float(row.get('open')  or 0.0)
        cl = float(row.get('close') or 0.0)
        if entry_price == 'open':
            return op if op > 0 else cl
        if entry_price == 'vwap':
            return 0.5 * ((op or cl) + cl)
        return cl

    for day in trading_days:
        day = pd.Timestamp(day)

        # ── 1. Build target portfolio from pred at (day - impl_lag) ──
        i_today = day_index.get(day, -1)
        if i_today < impl_lag:
            equity_rows.append({'trade_date': day, 'nav': cash, 'cash': cash,
                                 'invested': 0.0, 'n_pos': 0,
                                 'turnover_today': 0.0,
                                 'pnl_realized_day': 0.0})
            continue
        pred_date = pd.Timestamp(trading_days[i_today - impl_lag])
        today_preds = preds[preds['trade_date'] == pred_date]

        # ── 2. Filter to TRADEABLE stocks at today's open ──
        available: List[Tuple[str, float, float]] = []   # (ts, pred, sigma)
        held_codes = {p.ts_code for p in positions}
        held_st = sum(1 for p in positions if _is_st(p.ts_code, day, st_intervals))
        st_budget = (top_k if max_st_per_day < 0 else max(0, max_st_per_day - held_st))
        st_added = 0
        skipped = {'no_data': 0, 'limit_open': 0, 'no_sigma': 0, 'st_cap': 0,
                    'locked_all': 0}

        for _, r in today_preds.sort_values('pred', ascending=False).iterrows():
            ts = r['ts_code']
            df = prices.get(ts)
            if df is None or day not in df.index:
                skipped['no_data'] += 1; continue
            day_row = df.loc[day]
            # Open-time limit filter (the new gate)
            ol = is_open_at_limit(day_row, limit_pct=open_limit_pct)
            if ol and ts not in held_codes:
                skipped['limit_open'] += 1; continue
            # Also skip if locked-all-day (no liquidity)
            if _is_locked_at_limit(day_row) and ts not in held_codes:
                skipped['locked_all'] += 1; continue
            s_series = sigmas.get(ts)
            if s_series is None or day not in s_series.index:
                skipped['no_sigma'] += 1; continue
            s = float(s_series.loc[day])
            if not np.isfinite(s):
                skipped['no_sigma'] += 1; continue
            # ST budget — count already-held + new
            if max_st_per_day >= 0 and _is_st(ts, day, st_intervals):
                if ts not in held_codes and st_added >= st_budget:
                    skipped['st_cap'] += 1; continue
                if ts not in held_codes:
                    st_added += 1
            available.append((ts, float(r['pred']), s))
            if len(available) >= top_k * 4:   # collect a buffer; will trim to top_k
                break

        # ── 3. Pick top_k available, compute QP weights ──
        target_top = available[:top_k]
        target_codes = [t[0] for t in target_top]
        target_set = set(target_codes)

        # Build the "hold-set": the wider universe a held stock can stay in
        # without being sold. A stock that was top-K yesterday and is now
        # ranked at top_k + 1 .. top_k + hold_buffer_k stays put — we don't
        # churn it out for marginal ranking changes.
        hold_extended = [t[0] for t in available[: top_k + hold_buffer_k]]
        hold_set = set(hold_extended)

        # Compute QP weights for the target subset
        target_weights = {}
        if target_top:
            mu = np.array([t[1] for t in target_top], dtype=np.float64)
            sg = np.array([t[2] for t in target_top], dtype=np.float64)
            if solver == 'qp':
                panel_rows = []
                for u in target_top:
                    pf = prices.get(u[0])
                    if pf is None:
                        panel_rows.append(np.full(cov_window, np.nan))
                        continue
                    s = pf['pct_chg'].loc[:day].iloc[:-1].tail(cov_window)
                    arr = s.to_numpy(dtype=np.float64)
                    if arr.size < cov_window:
                        arr = np.concatenate([np.full(cov_window - arr.size, np.nan), arr])
                    panel_rows.append(arr)
                R = np.column_stack(panel_rows)
                cov = estimate_covariance(R, min_sigma=0.5)
                w = markowitz_qp_weights(mu, cov,
                                          risk_aversion=risk_aversion,
                                          long_only=True)
            else:
                w = markowitz_diagonal_weights(mu, sg, long_only=True)
            for ts, wi in zip(target_codes, w):
                target_weights[ts] = float(wi)

        # ── 4. Compute current dollar holdings (mark-to-market at today's open) ──
        invested_value = 0.0
        parked_value   = 0.0    # positions in hold_set but NOT in target_set
        current_dollars: Dict[str, float] = {}
        for p in positions:
            df = prices.get(p.ts_code)
            if df is None or day not in df.index:
                px = float(p.entry_price)
            else:
                row = df.loc[day]
                px = float(row.get('open') or row.get('close') or p.entry_price)
            v = px * p.shares
            current_dollars[p.ts_code] = v
            invested_value += v
            if p.ts_code not in target_set and p.ts_code in hold_set:
                parked_value += v

        nav = cash + invested_value
        if nav_cap > 0:
            nav = min(nav, nav_cap)
        # Tradeable NAV = full NAV minus dollars locked in parked (hold-buffer)
        # positions. Target weights apply to this fraction so the ENTRY phase
        # doesn't try to buy a full NAV-weight of top-K names while the
        # parked names are still consuming portfolio space.
        tradeable_nav = max(nav - parked_value, 0.0)

        # ── 5. EXIT phase: sell positions that fell out of target ──
        realized_today = 0.0
        kept: List[Position] = []
        turnover_today = 0.0

        for p in positions:
            target = target_weights.get(p.ts_code, 0.0) * tradeable_nav
            current = current_dollars.get(p.ts_code, 0.0)
            df = prices.get(p.ts_code)
            if df is None or day not in df.index:
                kept.append(p); continue
            row = df.loc[day]
            # T+1: same-day exits blocked
            if p.entry_date == day:
                kept.append(p); continue
            # Locked-down protection: can't fill on locked-down bar
            if _is_locked_at_limit(row) == 'locked_down':
                kept.append(p); continue

            # Sell only when the position drops out of the EXTENDED hold-set
            # (top_k + hold_buffer_k). Stocks that slip from rank 9 → 15 stay
            # parked; the "rebalance" trims weights but doesn't churn names.
            if p.ts_code not in hold_set:
                # FULL EXIT — fell past the stickiness buffer
                exit_px = _exec_price(row, is_buy=False)
                if exit_px <= 0:
                    kept.append(p); continue
                proceeds = exit_px * p.shares * (1.0 - xc)
                pnl = proceeds - p.cost_basis
                cash += proceeds
                realized_today += pnl
                turnover_today += proceeds
                trade_rows.append({
                    'ts_code': p.ts_code, 'entry_date': p.entry_date,
                    'exit_date': day, 'entry_price': p.entry_price,
                    'exit_price': exit_px, 'shares': p.shares,
                    'cost_basis': p.cost_basis, 'proceeds': proceeds,
                    'pnl': pnl, 'ret': pnl / p.cost_basis if p.cost_basis else 0,
                    'held_days': len(df.loc[p.entry_date:day]) - 1,
                    'reason': 'rebalance_drop',
                })
                continue

            # PARTIAL TRIM is only sensible for stocks that ARE in today's
            # target (top_k). For stocks in the buffer band (top_k+1..top_k+hold_buffer_k)
            # we DON'T trim — they're parked. Trimming them would churn names
            # we're trying to hold.
            if p.ts_code not in target_set:
                kept.append(p); continue
            diff = target - current
            if diff < -rebalance_threshold * nav:
                # Sell shares to reduce position
                exit_px = _exec_price(row, is_buy=False)
                if exit_px <= 0:
                    kept.append(p); continue
                shares_to_sell = np.floor(min(p.shares,
                                                 (-diff) / exit_px) / 100.0) * 100.0
                if shares_to_sell <= 0:
                    kept.append(p); continue
                proceeds = exit_px * shares_to_sell * (1.0 - xc)
                cost_per_share = p.cost_basis / p.shares if p.shares else 0
                booked_cost = cost_per_share * shares_to_sell
                pnl = proceeds - booked_cost
                cash += proceeds
                realized_today += pnl
                turnover_today += proceeds
                trade_rows.append({
                    'ts_code': p.ts_code, 'entry_date': p.entry_date,
                    'exit_date': day, 'entry_price': p.entry_price,
                    'exit_price': exit_px, 'shares': shares_to_sell,
                    'cost_basis': booked_cost, 'proceeds': proceeds,
                    'pnl': pnl, 'ret': pnl / booked_cost if booked_cost else 0,
                    'held_days': len(df.loc[p.entry_date:day]) - 1,
                    'reason': 'rebalance_trim',
                })
                p.shares     -= shares_to_sell
                p.cost_basis -= booked_cost
                kept.append(p)
                continue
            kept.append(p)
        positions = kept

        # ── 6. ENTRY / TOP-UP phase: buy under-allocated names ──
        # Recompute tradeable NAV after the EXIT phase (some positions may have
        # been sold, increasing cash; parked dollars unchanged).
        parked_now = sum(
            current_dollars.get(p.ts_code, 0.0)
            for p in positions
            if p.ts_code not in target_set and p.ts_code in hold_set
        )
        nav_now           = cash + sum(current_dollars.get(p.ts_code, 0.0)
                                        for p in positions)
        tradeable_nav_now = max(nav_now - parked_now, 0.0)
        held_codes = {p.ts_code for p in positions}
        for ts in target_codes:
            target = target_weights.get(ts, 0.0) * tradeable_nav_now
            current = sum(p.shares * (
                _exec_price(prices[ts].loc[day], is_buy=True) if ts in prices and day in prices[ts].index else 0.0
            ) for p in positions if p.ts_code == ts)
            diff = target - current
            if diff <= rebalance_threshold * tradeable_nav_now:
                continue
            # Position is under target → buy more
            df = prices.get(ts)
            if df is None or day not in df.index: continue
            row = df.loc[day]
            entry_px = _exec_price(row, is_buy=True)
            if entry_px <= 0: continue
            alloc = diff
            # ADV liquidity cap
            if advs is not None:
                adv_series = advs.get(ts)
                if adv_series is not None and day in adv_series.index:
                    adv_val = float(adv_series.loc[day])
                    if np.isfinite(adv_val) and adv_val > 0:
                        alloc = min(alloc, adv_val * max_pos_adv)
            gross_shares = alloc / (entry_px * (1.0 + ec))
            shares = np.floor(gross_shares / 100.0) * 100.0
            if shares <= 0: continue
            impact = 0.0
            if impact_bps > 0 and advs is not None:
                adv_series = advs.get(ts)
                if adv_series is not None and day in adv_series.index:
                    adv_val = float(adv_series.loc[day])
                    if np.isfinite(adv_val) and adv_val > 0:
                        impact = (impact_bps / 1e4) * np.sqrt(entry_px * shares / adv_val)
            cost = entry_px * shares * (1.0 + ec + impact)
            if cost > cash: continue
            cash -= cost
            turnover_today += cost
            # Add to or merge with existing position
            existing = next((p for p in positions if p.ts_code == ts), None)
            if existing is None:
                positions.append(Position(
                    ts_code=ts, entry_date=day, entry_price=entry_px,
                    shares=shares, cost_basis=cost,
                ))
            else:
                # Update entry_price as VWAP, keep oldest entry_date for T+1 reasoning
                total_shares = existing.shares + shares
                vwap = (existing.entry_price * existing.shares + entry_px * shares) / total_shares
                existing.entry_price = vwap
                existing.shares = total_shares
                existing.cost_basis += cost
                # entry_date stays the same so T+1 still applies to the OLDEST lot

        # ── 7. Equity row ──
        # Recompute invested at end-of-day close for fair NAV mark
        invested_close = 0.0
        for p in positions:
            df = prices.get(p.ts_code)
            if df is None or day not in df.index:
                invested_close += p.cost_basis
            else:
                invested_close += float(df.loc[day, 'close']) * p.shares
        nav_eod = cash + invested_close
        equity_rows.append({
            'trade_date': day, 'nav': nav_eod, 'cash': cash,
            'invested': invested_close, 'n_pos': len(positions),
            'turnover_today': turnover_today,
            'pnl_realized_day': realized_today,
        })

    equity = pd.DataFrame(equity_rows).set_index('trade_date')
    trades = pd.DataFrame(trade_rows)
    return equity, trades


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start',    default='2025-07-01')
    p.add_argument('--end',      default='2026-04-21')
    p.add_argument('--top_k',    type=int,   default=10)
    p.add_argument('--entry_bps', type=float, default=10.0)
    p.add_argument('--exit_bps',  type=float, default=15.0)
    p.add_argument('--initial',  type=float, default=1_000_000.0)
    p.add_argument('--open_limit_pct', type=float, default=9.5,
                   help='filter stocks whose open is within this %% of the daily limit '
                        '(default 9.5: skip stocks opening ≥ +9.5%% or ≤ -9.5%%).')
    p.add_argument('--max_pos_adv', type=float, default=0.05)
    p.add_argument('--nav_cap', type=float, default=0.0)
    p.add_argument('--impact_bps', type=float, default=10.0)
    p.add_argument('--solver', choices=['diag', 'qp'], default='qp')
    p.add_argument('--cov_window', type=int, default=60)
    p.add_argument('--risk_aversion', type=float, default=1.0)
    p.add_argument('--max_st_per_day', type=int, default=-1)
    p.add_argument('--impl_lag', type=int, default=1)
    p.add_argument('--entry_price', choices=['close', 'open', 'vwap'], default='open',
                   help='Entry execution model. Default = open (realistic with impl_lag=1).')
    p.add_argument('--rebalance_threshold', type=float, default=0.005,
                   help='minimum |target − current| as fraction of NAV before '
                        'triggering a trade. Default 0.005 = 0.5%% of NAV.')
    p.add_argument('--pred_smooth_days', type=int, default=5,
                   help='Smooth predictions with N-day EMA before ranking. '
                        'Stabilises top-K membership when raw daily preds '
                        'are noisy. 1 = no smoothing (raw), 5 = default, '
                        '10 = aggressive smoothing.')
    p.add_argument('--hold_buffer_k', type=int, default=20,
                   help='Stickiness buffer: a held position stays in the '
                        'portfolio while it is in top (top_k + hold_buffer_k) '
                        'of the daily ranking. Only sells when it falls past '
                        'that. Default 20 → top-K=10 holds while ranked 1-30. '
                        '0 = strict top-K-only churn.')
    p.add_argument('--preds_csv', default=None)
    p.add_argument('--tag',      default='rebalance',
                   help='suffix for output filenames')
    args = p.parse_args()

    preds = load_predictions(args.start, args.end, preds_csv=args.preds_csv)
    print(f"[rebalance] {len(preds):,} predictions "
          f"covering {preds['trade_date'].nunique()} trading days")

    # Optional EMA smoothing of predictions — stabilises rank order so the
    # rebalance strategy doesn't sell winners that briefly drop in the noise.
    if args.pred_smooth_days > 1:
        print(f"[rebalance] applying {args.pred_smooth_days}-day EMA to predictions ...")
        preds = preds.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        alpha = 2.0 / (args.pred_smooth_days + 1)
        preds['pred'] = preds.groupby('ts_code')['pred'].transform(
            lambda s: s.ewm(alpha=alpha, adjust=False).mean()
        )

    start_ts = preds['trade_date'].min()
    end_ts   = preds['trade_date'].max()
    ts_codes = sorted(preds['ts_code'].unique())

    print(f"[rebalance] loading prices for {len(ts_codes):,} stocks ...")
    prices = load_price_panel(ts_codes, start_ts, end_ts)
    sigmas = compute_rolling_sigma(prices, window=60)
    advs   = compute_rolling_adv(prices,   window=20)

    st_intervals = None
    if args.max_st_per_day >= 0:
        st_intervals = load_st_intervals()
        print(f"[rebalance] ST roster loaded: {len(st_intervals):,} stocks "
              f"(cap = {args.max_st_per_day} per day)")

    equity, trades = run_rebalance_backtest(
        preds, prices, sigmas, advs=advs,
        top_k=args.top_k, entry_bps=args.entry_bps, exit_bps=args.exit_bps,
        initial=args.initial,
        open_limit_pct=args.open_limit_pct,
        max_pos_adv=args.max_pos_adv, nav_cap=args.nav_cap,
        impact_bps=args.impact_bps,
        solver=args.solver, cov_window=args.cov_window,
        risk_aversion=args.risk_aversion,
        max_st_per_day=args.max_st_per_day,
        st_intervals=st_intervals,
        impl_lag=args.impl_lag,
        entry_price=args.entry_price,
        rebalance_threshold=args.rebalance_threshold,
        hold_buffer_k=args.hold_buffer_k,
    )

    bench   = load_benchmark(start_ts, end_ts)
    metrics = compute_metrics(equity, bench, initial=args.initial)
    tsum    = trade_summary(trades)

    # Turnover summary
    avg_turnover = float(equity['turnover_today'].mean())
    nav_avg      = float(equity['nav'].mean())
    daily_to_pct = (avg_turnover / nav_avg * 100) if nav_avg > 0 else 0.0
    annual_to_pct = daily_to_pct * 252

    print()
    print("=" * 70)
    print(f"REBALANCE BACKTEST  ({pd.Timestamp(args.start).date()} → {pd.Timestamp(args.end).date()})")
    print("=" * 70)
    print(f"  top_k       : {args.top_k}     open_limit_pct: {args.open_limit_pct}")
    print(f"  cost        : {args.entry_bps:.0f} bps entry / {args.exit_bps:.0f} bps exit  + impact_bps {args.impact_bps:.0f}")
    print(f"  impl_lag    : {args.impl_lag}    entry_price: {args.entry_price}")
    print(f"  rebal thr   : {args.rebalance_threshold * 100:.2f}% of NAV")
    print(f"  initial     : {args.initial:,.0f}")
    print()
    def _fmt(v, suffix='', mult=1.0):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return '   -   '
        return f'{v * mult:+.2f}{suffix}'
    # Percentage-like metrics are returned as fractions (0.50 = 50%); scale ×100 for display.
    PCT_KEYS = {'total_return','cagr','vol_ann','mdd','alpha_ann'}
    print("-- Strategy --")
    for k in ('total_return','cagr','vol_ann','sharpe','mdd','calmar'):
        v = metrics.get(k)
        if k in PCT_KEYS:
            print(f"  {k:14}: {_fmt(v, '%', 100)}")
        else:
            print(f"  {k:14}: {_fmt(v)}")
    fnav = metrics.get('final_nav')
    print(f"  final NAV     : {fnav:,.0f}" if fnav else "  final NAV     :   -")
    print()
    print("-- vs CSI300 --")
    for k in ('alpha_ann','beta','info_ratio'):
        v = metrics.get(k)
        if k in PCT_KEYS:
            print(f"  {k:14}: {_fmt(v, '%', 100)}")
        else:
            print(f"  {k:14}: {_fmt(v)}")
    print()
    print("-- Trades --")
    for k in ('n_trades','hit_rate','avg_win_pct','avg_loss_pct','median_held_days'):
        v = tsum.get(k)
        # tsum already in % units (hit_rate, avg_win_pct etc.)
        suf = '%' if 'pct' in k or 'rate' in k else ''
        print(f"  {k:18}: {_fmt(v, suf)}")
    print()
    print("-- Turnover --")
    print(f"  avg daily turnover : {avg_turnover:,.0f} CNY ({daily_to_pct:.2f}% of NAV/day)")
    print(f"  annualised turnover: {annual_to_pct:.0f}% of NAV/year")

    # Save artefacts
    plot_dir = ROOT / 'plots' / 'backtest_xgb_markowitz'
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or 'rebalance'
    equity.to_csv(plot_dir / f'equity_{tag}.csv')
    trades.to_csv(plot_dir / f'trades_{tag}.csv', index=False)
    with open(plot_dir / f'metrics_{tag}.txt', 'w', encoding='utf-8') as f:
        f.write(f"REBALANCE BACKTEST  {args.start} → {args.end}\n")
        f.write(f"top_k: {args.top_k}\n")
        f.write(f"open_limit_pct: {args.open_limit_pct}\n")
        f.write(f"impl_lag: {args.impl_lag}\n")
        f.write(f"entry_price: {args.entry_price}\n")
        f.write(f"rebalance_threshold: {args.rebalance_threshold}\n")
        f.write('\n')
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        for k, v in tsum.items():
            f.write(f"{k}: {v}\n")
        f.write(f"avg_daily_turnover_pct: {daily_to_pct:.4f}\n")
        f.write(f"annualised_turnover_pct: {annual_to_pct:.4f}\n")
    print(f"\n[rebalance] artefacts → plots/backtest_xgb_markowitz/")


if __name__ == '__main__':
    main()
