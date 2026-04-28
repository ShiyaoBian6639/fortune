"""
Backtest: xgbmodel → Markowitz mean-variance top-10 long-only portfolio
=======================================================================

Strategy
--------
At each trading day t:
  1. Score every stock with the walk-forward OOF prediction from the xgbmodel
     (stock_data/models/xgb_preds/test.csv, one fold per ~2 trading weeks).
  2. Select the top K = 10 stocks by predicted next-day excess return mu_i.
  3. Build a long-only Markowitz portfolio:
       - mu_i  = xgb point prediction
       - Sigma = diagonal matrix of per-stock uncertainty sigma_i^2
                 (proxied by the rolling-60d std of realized pct_chg, which is
                 the "probability dispersion" implied by the model's residual
                 distribution applied per stock)
       - closed-form diagonal Markowitz: w_i ∝ max(mu_i, 0) / sigma_i^2
       - normalize so sum(w_i) = 1 (fully invested if any mu_i > 0, else cash)
  4. Position management
       - Entry: close(t), with cost = entry_cost_bps
       - Intraday exit on any subsequent day d:
            * if high(d) >= entry * (1 + tp_pct)  → sell at tp_pct  (take profit)
            * elif low(d) <= entry * (1 - sl_pct) → sell at -sl_pct (stop loss)
            * elif d - t == max_hold_days         → sell at close(d) (horizon)
       - Ties broken TP > SL if both trigger same bar (generous; see DISCUSSION).
  5. Rebalance: at each close we free cash from exited positions and enter the
     new top-10 basket with that cash (cash-drag when positions are still held).

No look-ahead
-------------
Every OOF pred is from a fold whose training window ended strictly before the
fold's test window (with purge + embargo). Rolling realized vol is computed on
data strictly before trade_date.

Costs & slippage
----------------
entry_cost_bps + exit_cost_bps applied round-trip. Chinese A-share retail cost
is typically ~10-30 bps round-trip including commission + stamp duty.

Metrics
-------
CAGR, annualised volatility, Sharpe (rf=0), max drawdown, Calmar ratio,
hit-rate, avg win / avg loss, turnover, vs. CSI300 alpha and beta.

Usage
-----
    ./venv/Scripts/python -m backtest.xgb_markowitz \\
        --start 2021-04-22 --end 2026-04-21 \\
        --top_k 10 --tp 0.03 --sl 0.02 --max_hold 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / 'stock_data'
PREDS_CSV = DATA_DIR / 'models' / 'xgb_preds' / 'test.csv'
OUT_DIR   = ROOT / 'plots' / 'backtest_xgb_markowitz'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSI300_CSV = DATA_DIR / 'index' / 'idx_factor_pro' / '000300_SH.csv'


# ── Data loading ──────────────────────────────────────────────────────────────
def load_predictions(start: str, end: str,
                      preds_csv: Path = None) -> pd.DataFrame:
    """Load walk-forward OOF predictions between start and end (inclusive).

    `preds_csv` defaults to PREDS_CSV (the canonical xgbmodel output) but can
    point at any other model's `xgb_preds/test.csv` produced by model_compare.
    """
    p = Path(preds_csv) if preds_csv else PREDS_CSV
    df = pd.read_csv(p, usecols=['ts_code', 'trade_date', 'pred', 'target'])
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df[(df['trade_date'] >= pd.Timestamp(start)) &
            (df['trade_date'] <= pd.Timestamp(end))].copy()
    df = df.sort_values(['trade_date', 'pred'], ascending=[True, False]).reset_index(drop=True)
    return df


def load_price_panel(ts_codes: List[str], start: pd.Timestamp, end: pd.Timestamp,
                      warmup_days: int = 90) -> Dict[str, pd.DataFrame]:
    """Load OHLC + pct_chg + amount for each stock covering [start - warmup, end].

    Returns dict: ts_code → DataFrame(trade_date, open, high, low, close,
    pct_chg, amount) indexed by trade_date (pd.Timestamp) for fast lookup.
    `amount` is in CNY thousands (tushare convention) — we convert to CNY.
    """
    start_fetch = start - pd.Timedelta(days=warmup_days * 2)   # calendar buffer
    out: Dict[str, pd.DataFrame] = {}
    missing = 0
    for ts_code in ts_codes:
        code, suffix = ts_code.split('.')
        sub = 'sh' if suffix.upper() == 'SH' else 'sz'
        fp = DATA_DIR / sub / f'{code}.csv'
        if not fp.exists():
            missing += 1
            continue
        try:
            df = pd.read_csv(fp, usecols=['trade_date', 'open', 'high', 'low',
                                           'close', 'pre_close', 'pct_chg', 'amount'])
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df[(df['trade_date'] >= start_fetch) &
                    (df['trade_date'] <= end + pd.Timedelta(days=5))]
            df = df.sort_values('trade_date').reset_index(drop=True)
            if len(df) < 5:
                missing += 1
                continue
            df['amount_cny'] = df['amount'].astype(float) * 1000.0   # tushare: 千元
            df = df.set_index('trade_date')
            out[ts_code] = df
        except Exception:
            missing += 1
            continue
    print(f"[backtest] loaded prices for {len(out):,} stocks, missing {missing}")
    return out


def compute_rolling_adv(prices: Dict[str, pd.DataFrame],
                         window: int = 20) -> Dict[str, pd.Series]:
    """Rolling mean daily turnover (CNY) per stock, shifted by 1 so date t
    sees only amount up to t-1 (no look-ahead)."""
    out = {}
    for ts_code, df in prices.items():
        s = df['amount_cny'].rolling(window, min_periods=5).mean().shift(1)
        out[ts_code] = s
    return out


def load_benchmark(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """CSI300 close for the backtest window."""
    df = pd.read_csv(CSI300_CSV, encoding='utf-8-sig',
                     usecols=['trade_date', 'close'])
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    df = df[(df['trade_date'] >= start - pd.Timedelta(days=10)) &
            (df['trade_date'] <= end + pd.Timedelta(days=10))]
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    return df.set_index('trade_date')


# ── Strategy pieces ───────────────────────────────────────────────────────────
def compute_rolling_sigma(prices: Dict[str, pd.DataFrame],
                           window: int = 60) -> Dict[str, pd.Series]:
    """Rolling-window std of daily pct_chg (in %) per stock, indexed by date.

    sigma_i,t is computed on data strictly before t (shift by 1 so t cannot
    see its own realized return).
    """
    out = {}
    for ts_code, df in prices.items():
        s = df['pct_chg'].rolling(window, min_periods=20).std().shift(1)
        out[ts_code] = s
    return out


def markowitz_diagonal_weights(mu: np.ndarray, sigma: np.ndarray,
                                long_only: bool = True,
                                min_sigma: float = 0.5) -> np.ndarray:
    """Closed-form diagonal Markowitz: w_i ∝ mu_i / sigma_i^2 (long-only).

    Floored sigma at `min_sigma`% to avoid pathological tiny-denom blow-ups for
    stocks with very short history. With fewer than long_only=False a full
    Markowitz optimiser would be required; here diagonal Sigma has closed form.
    """
    s = np.maximum(sigma, min_sigma).astype(np.float64)
    m = mu.astype(np.float64)
    if long_only:
        m = np.clip(m, 0.0, None)
    raw = m / (s ** 2)
    total = raw.sum()
    if total <= 0:
        return np.zeros_like(raw)
    return raw / total


# ── Full covariance long-only QP ──────────────────────────────────────────────
def estimate_covariance(returns_panel: np.ndarray,
                         min_sigma: float = 0.5) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance from a (T, K) returns panel (in %).

    Falls back to a diagonal sample-variance matrix if T < 2K or LW fails.
    Diagonal floored at min_sigma**2 (in %**2) to keep Σ positive definite when
    a stock has almost no realised dispersion (recent IPO, halted name).
    """
    R = np.asarray(returns_panel, dtype=np.float64)
    T, K = R.shape
    floor2 = float(min_sigma) ** 2
    if T < max(20, 2 * K):
        var = np.nanvar(R, axis=0, ddof=1)
        var = np.where(np.isfinite(var) & (var > floor2), var, floor2)
        return np.diag(var)
    R_clean = R - np.nanmean(R, axis=0, keepdims=True)
    R_clean = np.nan_to_num(R_clean, nan=0.0)
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf(assume_centered=True).fit(R_clean).covariance_
    except Exception:
        cov = np.cov(R_clean, rowvar=False, ddof=1)
    diag = np.maximum(np.diag(cov), floor2)
    np.fill_diagonal(cov, diag)
    return cov


def markowitz_qp_weights(mu: np.ndarray, cov: np.ndarray,
                          risk_aversion: float = 1.0,
                          long_only: bool = True) -> np.ndarray:
    """Long-only mean-variance QP solved with SLSQP.

        minimize    0.5 * λ * wᵀ Σ w  −  μᵀ w
        subject to  Σ w_i = 1,  w_i ≥ 0  (if long_only)

    For diagonal Σ this collapses to the closed-form solution; for full Σ it
    accounts for cross-stock correlations (e.g. two highly-correlated banks
    each get a smaller weight than they would in the diagonal case).

    Falls back to the diagonal closed form if all μᵢ ≤ 0 or the solver fails.
    """
    from scipy.optimize import minimize
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    K = mu.size
    if K == 0:
        return np.zeros(0, dtype=np.float64)
    if long_only and (mu <= 0).all():
        return np.zeros(K, dtype=np.float64)

    sym_cov = 0.5 * (cov + cov.T)

    def obj(w):
        return 0.5 * risk_aversion * float(w @ sym_cov @ w) - float(mu @ w)

    def grad(w):
        return risk_aversion * (sym_cov @ w) - mu

    pos_mask = mu > 0
    if pos_mask.any():
        w0 = pos_mask.astype(np.float64)
        w0 = w0 / w0.sum()
    else:
        w0 = np.full(K, 1.0 / K)

    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0,
                    'jac': lambda w: np.ones_like(w)}]
    bounds = [(0.0, 1.0) if long_only else (-1.0, 1.0)] * K

    try:
        res = minimize(obj, w0, jac=grad, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
        if res.success:
            w = np.clip(res.x, 0.0 if long_only else -1.0, 1.0)
            s = w.sum()
            if s > 1e-9:
                return w / s
    except Exception:
        pass
    sigma_diag = np.sqrt(np.maximum(np.diag(sym_cov), 1e-8))
    return markowitz_diagonal_weights(mu, sigma_diag, long_only=long_only)


# ── Locked-limit detection ────────────────────────────────────────────────────
def _is_locked_at_limit(row, limit_pct: float = 9.8) -> str:
    """Return 'locked_down', 'locked_up', or '' for a daily price row.

    A bar is locked-at-limit when there's NO intraday range (open == high
    == low == close) AND the close sits at the up-limit or down-limit price.
    On these bars retail orders cannot fill (limit-down: no buyers; limit-up:
    no sellers), so TP/SL/horizon exits must wait one more day.

    The detector accepts a small tick tolerance (0.1%) to absorb pre_close
    rounding artefacts.
    """
    op = float(row.get('open',  0.0) or 0.0)
    hi = float(row.get('high',  0.0) or 0.0)
    lo = float(row.get('low',   0.0) or 0.0)
    cl = float(row.get('close', 0.0) or 0.0)
    pc = float(row.get('pre_close', 0.0) or 0.0)
    if pc <= 0 or hi <= 0 or lo <= 0:
        return ''
    # No movement at all — required precondition for "locked all day"
    if (hi - lo) > 1e-6 or abs(cl - op) > 1e-6:
        return ''
    pct = (cl / pc - 1.0) * 100.0
    if pct >= limit_pct - 0.1:
        return 'locked_up'
    if pct <= -limit_pct + 0.1:
        return 'locked_down'
    return ''


# ── Backtest loop ─────────────────────────────────────────────────────────────
class Position:
    __slots__ = ('ts_code', 'entry_date', 'entry_price', 'shares', 'cost_basis',
                 'weight', 'pred', 'sigma')

    def __init__(self, ts_code, entry_date, entry_price, shares, cost_basis,
                 weight, pred, sigma):
        self.ts_code     = ts_code
        self.entry_date  = entry_date
        self.entry_price = entry_price
        self.shares      = shares
        self.cost_basis  = cost_basis      # cash spent including entry cost
        self.weight      = weight
        self.pred        = pred
        self.sigma       = sigma


def run_backtest(
    preds:       pd.DataFrame,
    prices:      Dict[str, pd.DataFrame],
    sigmas:      Dict[str, pd.Series],
    advs:        Dict[str, pd.Series] = None,
    top_k:       int   = 10,
    tp_pct:      float = 0.03,
    sl_pct:      float = 0.02,
    max_hold:    int   = 5,
    entry_bps:   float = 10.0,
    exit_bps:    float = 15.0,        # includes stamp duty on sell
    initial:     float = 1_000_000.0,
    limit_pct:   float = 9.8,         # skip entries that gapped near limit
    max_pos_adv: float = 0.05,        # cap each position at 5% of 20d ADV
    nav_cap:     float = 0.0,         # 0 = no cap; else hard cap on deployable NAV
    impact_bps:  float = 0.0,         # extra market-impact bps per √(pos/ADV)
    solver:      str   = 'diag',      # 'diag' (closed form) or 'qp' (full Σ)
    cov_window:  int   = 60,          # rolling window for full Σ estimation
    risk_aversion: float = 1.0,       # λ in 0.5·λ·wᵀΣw − μᵀw
    max_st_per_day: int = -1,         # cap ST entries per day; -1 = no cap
    st_intervals:   Dict[str, list] = None,  # {ts_code: [(start, end, kind), ...]}
    entry_price:    str = 'close',    # 'close' (default) or 'vwap' (=(open+close)/2)
    impl_lag:       int = 1,          # trading-day lag between prediction and entry.
                                      # 1 = realistic (pred at X used to enter at close
                                      # of X+1, since the pred can only be computed
                                      # AFTER close(X)). 0 = legacy (instant entry,
                                      # used X's close as a feature *and* entered at
                                      # close X — same-time decision and execution).
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the day-by-day simulation.

    Returns
    -------
    equity : DataFrame(trade_date, nav, cash, invested, n_pos, pnl_realized_day)
    trades : DataFrame per-trade records
    """
    trading_days = sorted(preds['trade_date'].unique())
    day_index    = {pd.Timestamp(d): i for i, d in enumerate(trading_days)}
    print(f"[backtest] simulating {len(trading_days)} trading days "
          f"({pd.Timestamp(trading_days[0]).date()} → {pd.Timestamp(trading_days[-1]).date()})  "
          f"impl_lag={impl_lag} day(s)")

    cash = initial
    positions: List[Position] = []
    equity_rows = []
    trade_rows  = []

    ec = entry_bps / 1e4
    xc = exit_bps  / 1e4

    for day in trading_days:
        day = pd.Timestamp(day)
        realized_today = 0.0

        # ── 1. Exit logic on existing positions ────────────────────────────────
        kept: List[Position] = []
        for p in positions:
            # skip same-day exits for a position that was entered at close today
            if p.entry_date == day:
                kept.append(p)
                continue
            df = prices.get(p.ts_code)
            if df is None or day not in df.index:
                kept.append(p)
                continue
            row = df.loc[day]

            held_days = len(df.loc[p.entry_date:day]) - 1
            op = float(row['open']) if 'open' in row else float(row['close'])
            hi = float(row['high']); lo = float(row['low']); cl = float(row['close'])

            tp_price = p.entry_price * (1.0 + tp_pct)
            sl_price = p.entry_price * (1.0 - sl_pct)

            # ── Locked-limit gate (T+N realism) ──
            # If the stock is locked at limit-up or limit-down for the entire
            # session, retail orders cannot fill: at limit-down there are no
            # buyers; at limit-up there are no sellers willing to give up the
            # gain. We hold the position and re-evaluate tomorrow — the
            # position may keep losing (or gaining) as additional locked days
            # stack up. After max_hold * 2 days we force-exit at close even on
            # a locked bar (assumed to clear by then; rare in practice).
            locked = _is_locked_at_limit(row, limit_pct=limit_pct)
            if locked and held_days < max_hold * 2:
                kept.append(p)
                continue

            exit_price = None
            reason     = None
            # Realistic exit-pricing model:
            #   T+1 rule: same-day exits already blocked above (line 337).
            #   Gap-down realism: an order placed at the open fills at OPEN
            #     price, not at the SL trigger. So a -2% SL on a stock that
            #     gaps to -8% open and locks at -10% fills at -8%, not -2%.
            #     If the stock locks all day → no fill (handled by the
            #     locked-limit gate above).
            #   TP triggers if low ≤ tp_price ≤ high (stock CROSSED the TP)
            #     • gap-up open (open > tp_price): fill = open (not tp_price)
            #     • normal: fill = tp_price
            #   SL symmetric: triggers if low ≤ sl_price ≤ high
            #     • gap-down open (open < sl_price): fill = open
            #     • normal: fill = sl_price
            #   If both could trigger, prefer SL (more conservative).
            tp_hit = (hi >= tp_price) and (lo <= tp_price)
            sl_hit = (lo <= sl_price) and (hi >= sl_price)
            if (op > tp_price) and not tp_hit:
                tp_hit  = True
                tp_fill = op
            else:
                tp_fill = tp_price
            if (op < sl_price) and not sl_hit:
                sl_hit  = True
                sl_fill = op
            else:
                sl_fill = sl_price

            if tp_hit and sl_hit:
                exit_price = sl_fill
                reason     = 'stop_loss'
            elif tp_hit:
                exit_price = tp_fill
                reason     = 'take_profit'
            elif sl_hit:
                exit_price = sl_fill
                reason     = 'stop_loss'
            elif held_days >= max_hold:
                # Horizon: sell at close, unless locked-at-limit (handled above).
                # The lo<hi guard remains as a belt-and-braces check.
                if lo < hi or held_days < max_hold * 2:
                    exit_price = cl
                    reason     = 'horizon'

            if exit_price is None:
                kept.append(p)
                continue

            gross   = exit_price * p.shares
            proceed = gross * (1.0 - xc)
            pnl     = proceed - p.cost_basis
            cash   += proceed
            realized_today += pnl

            trade_rows.append({
                'ts_code':     p.ts_code,
                'entry_date':  p.entry_date,
                'exit_date':   day,
                'entry_price': p.entry_price,
                'exit_price':  exit_price,
                'shares':      p.shares,
                'cost_basis':  p.cost_basis,
                'proceeds':    proceed,
                'pnl':         pnl,
                'ret':         pnl / p.cost_basis,
                'held_days':   held_days,
                'reason':      reason,
                'pred':        p.pred,
                'sigma':       p.sigma,
                'weight':      p.weight,
            })
        positions = kept

        # ── 2. Mark-to-market NAV (positions use today's close) ────────────────
        invested_value = 0.0
        n_pos = 0
        for p in positions:
            df = prices.get(p.ts_code)
            if df is None or day not in df.index:
                invested_value += p.cost_basis         # stale mark
            else:
                invested_value += float(df.loc[day, 'close']) * p.shares * (1.0 - xc)
            n_pos += 1

        # ── 3. Entry: top-K by pred, only if we have cash ──────────────────────
        # Implementation lag: a prediction at trade_date P uses features through
        # close(P). The earliest realistic entry timestamp is close(P + impl_lag).
        # impl_lag=1 → enter at close(day) using preds at trade_date (day - 1).
        i_today = day_index.get(day, -1)
        if i_today < impl_lag:
            # Not enough history of preds yet — append empty equity row + continue
            nav = cash + 0.0
            equity_rows.append({'trade_date': day, 'nav': nav, 'cash': cash,
                                 'invested': 0.0, 'n_pos': 0,
                                 'pnl_realized_day': realized_today})
            continue
        pred_date = pd.Timestamp(trading_days[i_today - impl_lag]) if impl_lag > 0 else day
        today = preds[preds['trade_date'] == pred_date]
        if today.empty:
            # append equity row before continuing
            nav = cash + invested_value
            equity_rows.append({'trade_date': day, 'nav': nav, 'cash': cash,
                                 'invested': invested_value, 'n_pos': n_pos,
                                 'pnl_realized_day': realized_today})
            continue

        # sort + filter: need price at day, need a rolling sigma, need to not
        # already hold the name, and guard vs limit-up bar
        candidates = today.sort_values('pred', ascending=False).head(top_k * 4)

        already_held = {p.ts_code for p in positions}
        # ST count cap: count both already-held ST and new entries against
        # max_st_per_day. -1 disables the cap (default behaviour preserved).
        day_str = day.strftime('%Y%m%d')

        def _is_st(ts_code: str) -> bool:
            if st_intervals is None:
                return False
            iv = st_intervals.get(ts_code)
            if not iv:
                return False
            for start, end, _kind in iv:
                if start <= day_str <= end:
                    return True
            return False

        held_st = sum(1 for p in positions if _is_st(p.ts_code))
        st_budget = (top_k if max_st_per_day < 0 else max(0, max_st_per_day - held_st))
        st_added  = 0

        usable = []
        skipped_st_cap = []   # for diagnostics
        for _, r in candidates.iterrows():
            ts_code = r['ts_code']
            if ts_code in already_held:
                continue
            pf = prices.get(ts_code)
            if pf is None or day not in pf.index:
                continue
            day_row = pf.loc[day]
            # Buyability filter — was the stock buyable somewhere during the day?
            # OLD (over-strict): if |pct_chg(day)| >= 9.8 → block.
            #   This blocks stocks that closed at limit, even if they traded
            #   below the limit during the day (intraday dip = buy opportunity).
            # NEW: filter only if the stock was locked at limit ALL DAY,
            #   i.e. low(day) ≥ up_limit_price ≈ pre_close × (1 + limit/100).
            #   That's the only case where intraday execution is impossible.
            pre_close = float(day_row.get('pre_close') or 0.0)
            low       = float(day_row.get('low')       or 0.0)
            if pre_close > 0 and low > 0:
                # convert low to "pct_chg-from-pre_close" %
                low_pct = (low / pre_close - 1.0) * 100.0
                if low_pct >= limit_pct - 0.1:  # 0.1% tolerance for tick rounding
                    continue                    # locked at upper limit all day
                if low_pct <= -limit_pct + 0.1:
                    continue                    # locked at lower limit all day
            else:
                # Fallback to old filter if pre_close/low missing
                if abs(float(day_row['pct_chg'])) >= limit_pct:
                    continue
            s_series = sigmas.get(ts_code)
            if s_series is None or day not in s_series.index:
                continue
            s = s_series.loc[day]
            if not np.isfinite(s):
                continue
            # Apply ST cap
            if max_st_per_day >= 0 and _is_st(ts_code):
                if st_added >= st_budget:
                    skipped_st_cap.append(ts_code)
                    continue
                st_added += 1
            # Entry-price model:
            #   close (legacy) — buy at day's close auction (15:00). Best when
            #     the prediction was made BEFORE close (i.e. impl_lag=0). Slight
            #     look-ahead because pred(day) used close(day) as a feature.
            #   open  (recommended for impl_lag>=1) — buy at day's open auction
            #     (09:30). pred(day-impl_lag) was computed overnight and a
            #     market-on-open order fills near open(day). Captures the
            #     intra-day portion of the predicted move.
            #   vwap  (~mid-day proxy) — average of open and close.
            entry_close = float(day_row['close'])
            entry_open  = float(day_row.get('open') or entry_close)
            if entry_price == 'open':
                entry_px = entry_open
            elif entry_price == 'vwap':
                entry_px = 0.5 * (entry_open + entry_close)
            else:
                entry_px = entry_close
            usable.append((ts_code, float(r['pred']), float(s), entry_px))
            if len(usable) >= top_k:
                break

        if usable and cash > 1000:
            mu = np.array([u[1] for u in usable], dtype=np.float64)
            sg = np.array([u[2] for u in usable], dtype=np.float64)
            if solver == 'qp':
                # Build (T × K) realised pct_chg panel from data strictly before `day`
                panel_rows = []
                for u in usable:
                    pf = prices.get(u[0])
                    if pf is None:
                        panel_rows.append(np.full(cov_window, np.nan))
                        continue
                    s = pf['pct_chg'].loc[:day].iloc[:-1].tail(cov_window)
                    arr = s.to_numpy(dtype=np.float64)
                    if arr.size < cov_window:
                        arr = np.concatenate([np.full(cov_window - arr.size, np.nan), arr])
                    panel_rows.append(arr)
                R = np.column_stack(panel_rows)        # (T, K)
                cov = estimate_covariance(R, min_sigma=0.5)
                w = markowitz_qp_weights(mu, cov,
                                          risk_aversion=risk_aversion,
                                          long_only=True)
            else:
                w = markowitz_diagonal_weights(mu, sg, long_only=True)

            # Deployable budget: min(95% of cash, nav_cap) so an unbounded NAV
            # can't swallow the whole A-share universe's small-cap liquidity
            budget = cash * 0.95
            if nav_cap > 0:
                budget = min(budget, nav_cap)
            for (ts_code, pred_i, sig_i, close_i), wi in zip(usable, w):
                if wi <= 0 or not np.isfinite(close_i) or close_i <= 0:
                    continue
                alloc = budget * wi
                # Per-stock ADV liquidity cap
                if advs is not None:
                    adv_series = advs.get(ts_code)
                    if adv_series is not None and day in adv_series.index:
                        adv_val = float(adv_series.loc[day])
                        if np.isfinite(adv_val) and adv_val > 0:
                            alloc = min(alloc, adv_val * max_pos_adv)
                gross_shares = alloc / (close_i * (1.0 + ec))
                shares = np.floor(gross_shares / 100.0) * 100.0   # A-shares 100-lot
                if shares <= 0:
                    continue
                # Market impact: extra cost scaled by order-to-ADV ratio
                impact = 0.0
                if impact_bps > 0 and advs is not None:
                    adv_series = advs.get(ts_code)
                    if adv_series is not None and day in adv_series.index:
                        adv_val = float(adv_series.loc[day])
                        if np.isfinite(adv_val) and adv_val > 0:
                            order_val = close_i * shares
                            impact = (impact_bps / 1e4) * np.sqrt(order_val / adv_val)
                cost = close_i * shares * (1.0 + ec + impact)
                if cost > cash:
                    continue
                cash -= cost
                positions.append(Position(
                    ts_code=ts_code, entry_date=day, entry_price=close_i,
                    shares=shares, cost_basis=cost,
                    weight=float(wi), pred=pred_i, sigma=sig_i,
                ))

        # ── 4. Recompute invested value after entries (entry cash moved out) ───
        invested_value = 0.0
        n_pos = len(positions)
        for p in positions:
            df = prices.get(p.ts_code)
            if df is None or day not in df.index:
                invested_value += p.cost_basis
            else:
                invested_value += float(df.loc[day, 'close']) * p.shares * (1.0 - xc)

        nav = cash + invested_value
        equity_rows.append({'trade_date': day, 'nav': nav, 'cash': cash,
                             'invested': invested_value, 'n_pos': n_pos,
                             'pnl_realized_day': realized_today})

    equity = pd.DataFrame(equity_rows).set_index('trade_date')
    trades = pd.DataFrame(trade_rows)
    return equity, trades


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(equity: pd.DataFrame, bench: pd.DataFrame,
                     initial: float) -> Dict[str, float]:
    nav = equity['nav'].values
    idx = equity.index

    # Align benchmark to same trading days
    b = bench.reindex(idx, method='ffill')
    b_norm = b['close'] / b['close'].iloc[0] * initial

    # Daily returns
    ret  = pd.Series(nav, index=idx).pct_change().fillna(0.0)
    bret = b['close'].pct_change().fillna(0.0)

    years = (idx[-1] - idx[0]).days / 365.25
    total_return = nav[-1] / initial - 1.0
    cagr = (nav[-1] / initial) ** (1.0 / max(years, 1e-6)) - 1.0

    vol  = ret.std(ddof=1) * np.sqrt(252)
    sharpe = (ret.mean() * 252) / (vol + 1e-12)

    # Max drawdown
    roll_max = pd.Series(nav, index=idx).cummax()
    dd       = (pd.Series(nav, index=idx) / roll_max - 1.0)
    mdd      = dd.min()
    mdd_date = dd.idxmin()
    peak_before = roll_max.loc[:mdd_date].idxmax()

    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    # CSI300 stats
    b_total   = b['close'].iloc[-1] / b['close'].iloc[0] - 1.0
    b_cagr    = (b['close'].iloc[-1] / b['close'].iloc[0]) ** (1.0 / max(years, 1e-6)) - 1.0
    b_vol     = bret.std(ddof=1) * np.sqrt(252)
    b_sharpe  = (bret.mean() * 252) / (b_vol + 1e-12)
    b_roll    = b['close'].cummax()
    b_dd      = (b['close'] / b_roll - 1.0)
    b_mdd     = b_dd.min()

    # Beta / alpha vs CSI300
    cov = np.cov(ret.values, bret.values, ddof=1)
    beta  = cov[0, 1] / (cov[1, 1] + 1e-12)
    alpha = (ret.mean() - beta * bret.mean()) * 252

    # Information ratio
    active = ret - bret
    ir = (active.mean() * 252) / (active.std(ddof=1) * np.sqrt(252) + 1e-12)

    return {
        'years':          years,
        'total_return':   total_return,
        'cagr':           cagr,
        'vol_ann':        vol,
        'sharpe':         sharpe,
        'max_drawdown':   mdd,
        'mdd_peak_date':  str(peak_before.date()),
        'mdd_trough_date': str(mdd_date.date()),
        'calmar':         calmar,
        'bench_total':    b_total,
        'bench_cagr':     b_cagr,
        'bench_vol':      b_vol,
        'bench_sharpe':   b_sharpe,
        'bench_mdd':      b_mdd,
        'alpha_ann':      alpha,
        'beta':           beta,
        'info_ratio':     ir,
        'final_nav':      nav[-1],
    }


def trade_summary(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {}
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] <= 0]
    summary = {
        'n_trades':        len(trades),
        'hit_rate':        len(wins) / len(trades),
        'avg_win_pct':     float(wins['ret'].mean() * 100) if len(wins)  else 0.0,
        'avg_loss_pct':    float(losses['ret'].mean() * 100) if len(losses) else 0.0,
        'avg_ret_pct':     float(trades['ret'].mean() * 100),
        'median_hold_days': float(trades['held_days'].median()),
        'pct_tp':          float((trades['reason'] == 'take_profit').mean()),
        'pct_sl':          float((trades['reason'] == 'stop_loss').mean()),
        'pct_horizon':     float((trades['reason'] == 'horizon').mean()),
    }
    return summary


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(equity: pd.DataFrame, bench: pd.DataFrame,
                  metrics: Dict[str, float], initial: float, out_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    b = bench.reindex(equity.index, method='ffill')
    b_norm = b['close'] / b['close'].iloc[0] * initial
    nav = equity['nav']
    roll_max = nav.cummax()
    dd       = (nav / roll_max - 1.0) * 100
    b_dd     = (b['close'] / b['close'].cummax() - 1.0) * 100

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                              gridspec_kw={'height_ratios': [2.5, 1, 1]})

    # Equity curve
    ax = axes[0]
    ax.plot(equity.index, nav, label='xgb Markowitz top-10',
            color='#1f77b4', linewidth=1.5)
    ax.plot(equity.index, b_norm, label='CSI300 buy&hold',
            color='#888', linewidth=1.2, alpha=0.8)
    ax.set_ylabel('NAV (CNY)')
    ax.set_title(f"Equity curve  "
                 f"CAGR {metrics['cagr']*100:+.2f}% vs CSI300 {metrics['bench_cagr']*100:+.2f}%  "
                 f"Sharpe {metrics['sharpe']:.2f}  MDD {metrics['max_drawdown']*100:+.1f}%")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # Drawdown
    ax = axes[1]
    ax.fill_between(equity.index, dd, 0, color='#d62728', alpha=0.5,
                     label='strategy DD')
    ax.plot(equity.index, b_dd, color='#888', linewidth=1, alpha=0.8,
            label='CSI300 DD')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    # Position count + cash
    ax = axes[2]
    ax.plot(equity.index, equity['n_pos'], color='#2ca02c',
            linewidth=1, label='# positions')
    ax2 = ax.twinx()
    ax2.plot(equity.index, equity['cash'] / 1e4, color='#ff7f0e',
             linewidth=1, alpha=0.7, label='cash (万 CNY)')
    ax.set_ylabel('# positions', color='#2ca02c')
    ax2.set_ylabel('cash (万 CNY)', color='#ff7f0e')
    ax.grid(alpha=0.3)
    ax.set_xlabel('date')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[backtest] saved plot → {out_path}")


# ── CLI / main ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start',    default='2021-04-22')
    p.add_argument('--end',      default='2026-04-21')
    p.add_argument('--top_k',    type=int,   default=10)
    p.add_argument('--tp',       type=float, default=0.03,
                   help='take-profit threshold (fractional, e.g. 0.03 = 3%%)')
    p.add_argument('--sl',       type=float, default=0.02,
                   help='stop-loss threshold (fractional)')
    p.add_argument('--max_hold', type=int,   default=5)
    p.add_argument('--entry_bps', type=float, default=10.0,
                   help='entry-side transaction cost (basis points)')
    p.add_argument('--exit_bps',  type=float, default=15.0,
                   help='exit-side transaction cost in bps (includes stamp tax)')
    p.add_argument('--initial',  type=float, default=1_000_000.0)
    p.add_argument('--sigma_window', type=int, default=60,
                   help='rolling window (days) for per-stock sigma estimate')
    p.add_argument('--max_pos_adv', type=float, default=0.05,
                   help='cap per-stock position at this fraction of 20d ADV '
                        '(default 0.05 = 5%%, realistic liquidity)')
    p.add_argument('--nav_cap', type=float, default=0.0,
                   help='hard cap on deployable NAV (0 = no cap). Useful to '
                        'see performance at a fund size the market can absorb.')
    p.add_argument('--impact_bps', type=float, default=10.0,
                   help='extra market-impact cost in bps × sqrt(order/ADV)')
    p.add_argument('--solver', choices=['diag', 'qp'], default='diag',
                   help='diag = closed-form diagonal Σ; '
                        'qp = full Ledoit-Wolf shrunk Σ via SLSQP long-only QP')
    p.add_argument('--max_st_per_day', type=int, default=-1,
                   help='cap ST/*ST positions in any daily portfolio at N. '
                        '-1 = no cap (preserves original results). 0 = exclude '
                        'ST entirely. Detected via stock_data/st_history.csv '
                        '(downloaded by api.st_history) — the authoritative '
                        'tushare namechange roster, NOT the 5%% band heuristic.')
    p.add_argument('--entry_price', choices=['close', 'open', 'vwap'], default='close',
                   help='Entry execution model. '
                        'close (default) = buy at day close (15:00 auction). '
                        'open  = buy at day open (09:30 auction) — realistic '
                        '         when pair with --impl_lag 1 (overnight pred '
                        '         deployed via market-on-open). '
                        'vwap  = (open+close)/2 mid-day proxy.')
    p.add_argument('--impl_lag', type=int, default=1,
                   help='Trading-day lag between prediction and entry. '
                        '1 (default, realistic) = pred at trade_date P uses '
                        'features through close(P), so the earliest realistic '
                        'entry is close(P+1) — gives the pipeline overnight to '
                        'compute predictions and submit market-on-close orders. '
                        '0 = legacy zero-lag (same-day pred and entry; uses '
                        "close(P) as a feature AND enters at close(P) — "
                        'instantaneous decision, optimistic).')
    p.add_argument('--cov_window', type=int, default=60,
                   help='rolling window for full Σ estimation (qp solver only)')
    p.add_argument('--risk_aversion', type=float, default=1.0,
                   help='λ in 0.5·λ·wᵀΣw − μᵀw (qp solver only). Higher = more '
                        'risk-averse, more diversified weights.')
    p.add_argument('--preds_csv', default=None,
                   help='Override predictions CSV. Default = canonical XGBoost. '
                        'Use stock_data/models_<engine>/xgb_preds/test.csv to '
                        'backtest against an alternative model.')
    p.add_argument('--tag',      default='',
                   help='suffix for output filenames (e.g. "aggressive")')
    args = p.parse_args()

    preds = load_predictions(args.start, args.end, preds_csv=args.preds_csv)
    print(f"[backtest] {len(preds):,} predictions "
          f"covering {preds['trade_date'].nunique()} trading days")

    start_ts = preds['trade_date'].min()
    end_ts   = preds['trade_date'].max()

    ts_codes = preds['ts_code'].unique().tolist()
    prices = load_price_panel(ts_codes, start_ts, end_ts,
                               warmup_days=args.sigma_window + 30)
    sigmas = compute_rolling_sigma(prices, window=args.sigma_window)
    advs   = compute_rolling_adv  (prices, window=20)

    # Optional: load the ST roster if --max_st_per_day is in effect
    st_intervals = None
    if args.max_st_per_day >= 0:
        try:
            from api.st_history import load_roster, build_daily_index
            roster = load_roster()
            st_intervals = build_daily_index(roster)
            print(f"[backtest] loaded ST roster: {len(st_intervals):,} stocks ever flagged ST "
                  f"(cap = {args.max_st_per_day} per day)")
        except Exception as e:
            print(f"[backtest] WARNING: ST roster unavailable ({e}); proceeding without ST cap")

    equity, trades = run_backtest(
        preds, prices, sigmas, advs=advs,
        top_k=args.top_k, tp_pct=args.tp, sl_pct=args.sl,
        max_hold=args.max_hold,
        entry_bps=args.entry_bps, exit_bps=args.exit_bps,
        initial=args.initial,
        max_pos_adv=args.max_pos_adv, nav_cap=args.nav_cap,
        impact_bps=args.impact_bps,
        solver=args.solver, cov_window=args.cov_window,
        risk_aversion=args.risk_aversion,
        max_st_per_day=args.max_st_per_day,
        st_intervals=st_intervals,
        entry_price=args.entry_price,
        impl_lag=args.impl_lag,
    )

    bench = load_benchmark(start_ts, end_ts)
    metrics = compute_metrics(equity, bench, initial=args.initial)
    tsum    = trade_summary(trades)

    suffix = f"_{args.tag}" if args.tag else ''
    equity_path  = OUT_DIR / f'equity{suffix}.csv'
    trades_path  = OUT_DIR / f'trades{suffix}.csv'
    metrics_path = OUT_DIR / f'metrics{suffix}.txt'
    plot_path    = OUT_DIR / f'equity_curve{suffix}.png'

    equity.to_csv(equity_path)
    trades.to_csv(trades_path, index=False)

    lines = []
    lines.append(f"==== xgbmodel Markowitz top-{args.top_k} backtest ====")
    lines.append(f"window        : {start_ts.date()} → {end_ts.date()}  ({metrics['years']:.2f} years)")
    lines.append(f"take-profit   : +{args.tp*100:.1f}%   stop-loss : -{args.sl*100:.1f}%   "
                 f"max_hold : {args.max_hold}d   sigma window : {args.sigma_window}d")
    lines.append(f"tx cost       : {args.entry_bps:.0f} bps entry / {args.exit_bps:.0f} bps exit "
                 f"+ {args.impact_bps:.0f} bps impact*√(pos/ADV)")
    lines.append(f"liquidity     : max position = {args.max_pos_adv*100:.1f}% of 20d ADV    "
                 f"nav cap = {args.nav_cap:,.0f}" if args.nav_cap > 0 else
                 f"liquidity     : max position = {args.max_pos_adv*100:.1f}% of 20d ADV    nav cap = none")
    lines.append(f"initial       : {args.initial:,.0f}")
    lines.append("")
    lines.append("-- Strategy ----------------------------------------------")
    lines.append(f"  final NAV      : {metrics['final_nav']:>14,.0f}")
    lines.append(f"  total return   : {metrics['total_return']*100:>+12.2f}%")
    lines.append(f"  CAGR           : {metrics['cagr']*100:>+12.2f}%")
    lines.append(f"  vol (ann)      : {metrics['vol_ann']*100:>12.2f}%")
    lines.append(f"  Sharpe (rf=0)  : {metrics['sharpe']:>12.2f}")
    lines.append(f"  max drawdown   : {metrics['max_drawdown']*100:>+12.2f}%")
    lines.append(f"  mdd peak → trough : {metrics['mdd_peak_date']} → {metrics['mdd_trough_date']}")
    lines.append(f"  Calmar         : {metrics['calmar']:>12.2f}")
    lines.append("")
    lines.append("-- CSI300 benchmark -------------------------------------")
    lines.append(f"  total return   : {metrics['bench_total']*100:>+12.2f}%")
    lines.append(f"  CAGR           : {metrics['bench_cagr']*100:>+12.2f}%")
    lines.append(f"  vol (ann)      : {metrics['bench_vol']*100:>12.2f}%")
    lines.append(f"  Sharpe         : {metrics['bench_sharpe']:>12.2f}")
    lines.append(f"  max drawdown   : {metrics['bench_mdd']*100:>+12.2f}%")
    lines.append("")
    lines.append("-- Cross-stats ------------------------------------------")
    lines.append(f"  alpha (ann)    : {metrics['alpha_ann']*100:>+12.2f}%")
    lines.append(f"  beta           : {metrics['beta']:>12.2f}")
    lines.append(f"  info ratio     : {metrics['info_ratio']:>12.2f}")
    lines.append("")
    if tsum:
        lines.append("-- Trades -----------------------------------------------")
        lines.append(f"  n_trades       : {tsum['n_trades']:>12,}")
        lines.append(f"  hit rate       : {tsum['hit_rate']*100:>12.2f}%")
        lines.append(f"  avg win        : {tsum['avg_win_pct']:>+12.2f}%")
        lines.append(f"  avg loss       : {tsum['avg_loss_pct']:>+12.2f}%")
        lines.append(f"  avg return     : {tsum['avg_ret_pct']:>+12.2f}%")
        lines.append(f"  median hold    : {tsum['median_hold_days']:>12.1f} d")
        lines.append(f"  take-profit %  : {tsum['pct_tp']*100:>12.1f}%")
        lines.append(f"  stop-loss %    : {tsum['pct_sl']*100:>12.1f}%")
        lines.append(f"  horizon exit % : {tsum['pct_horizon']*100:>12.1f}%")

    report = "\n".join(lines)
    print("\n" + report + "\n")
    with open(metrics_path, 'w') as fh:
        fh.write(report + "\n")

    plot_results(equity, bench, metrics, initial=args.initial, out_path=plot_path)
    print(f"[backtest] artefacts:  {equity_path}\n"
          f"                       {trades_path}\n"
          f"                       {metrics_path}\n"
          f"                       {plot_path}")


if __name__ == '__main__':
    main()
