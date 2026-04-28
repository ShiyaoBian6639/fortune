"""Advanced ensemble pipeline for BTC IC improvement (v3).

Beyond v2's vol-normalized XGB:

  * Engineered features
      - Lag features (1, 3, 5d) on top technicals
      - Interactions (vol×momentum, dominance×alts, funding×trend)
      - Intraday microstructure: semivariance up/down, jump count

  * Multi-horizon ensemble of regression models
      - XGB at h=3, 5, 7, 10  (each predicting vol-normalized target)
      - Ridge on quantile-transformed features at h=5

  * Confidence gate: XGB classifier on sign(h=5)

  * Sample weighting: exponential decay (half-life=250 trading days)

  * Early stopping: last 60 train bars as validation in every refit

  * Strategy uses ensemble expected return AND classifier prob > 0.5

Outputs btc_data/backtest_results.json (replaces v2 output, dashboard works
unchanged).

Usage:
    ./venv/Scripts/python -m btc.advanced_pipeline
"""

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore", category=UserWarning)

HORIZONS = [3, 5, 7, 10]
PRIMARY_HORIZON = 5
VOL_WINDOW = 20
COVERAGE_THRESHOLD = 0.90
INITIAL_TRAIN_FRAC = 0.5
REFIT_EVERY = 30
WEIGHT_HALF_LIFE = 250
ES_VAL_SIZE = 60

LAG_FEATURES = [
    "btc_rsi_14", "btc_macd_hist", "btc_log_ret_1", "btc_atr_pct",
    "btc_log_ret_5", "btc_bb_pctb", "intraday_rv", "intraday_taker_ratio",
    "intraday_semi_skew", "funding_rate", "btc_eth_corr_30", "perp_basis",
]
LAGS = [1, 3, 5]

INTERACTIONS = [
    ("btc_rsi_14", "btc_atr_pct"),
    ("btc_log_ret_5", "intraday_rv"),
    ("funding_rate", "btc_log_ret_5"),
    ("btc_eth_corr_30", "eth_log_ret_1"),
    ("btcdom_log_ret_5", "alts_mean_ret_5"),
    ("btc_macd_hist", "btc_rv_20"),
    ("intraday_semi_skew", "btc_log_ret_1"),
]

XGB_REG_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=8,
    tree_method="hist", random_state=42, n_jobs=-1,
    early_stopping_rounds=30,
)

XGB_CLS_PARAMS = dict(
    objective="binary:logistic",
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=8,
    tree_method="hist", random_state=42, n_jobs=-1,
    early_stopping_rounds=30,
)

RAW_PRICE_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
                  "trades", "taker_buy_base", "taker_buy_quote",
                  "btcdom_close", "perp_close", "perp_volume"]


@dataclass
class StrategyConfig:
    entry_threshold: float = 0.005
    exit_threshold: float = -0.002
    cls_min_prob: float = 0.50
    stop_loss: float = 0.03
    take_profit: float = 0.06
    max_hold_bars: int = 5
    cost_bps_per_side: float = 5.0
    initial_capital: float = 10_000.0
    regime_vol_quantile: float = 0.85
    regime_vol_lookback: int = 250
    regime_dom_threshold: float = 0.05


def engineer_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    extras = {}
    for f in LAG_FEATURES:
        if f in df.columns:
            for lag in LAGS:
                extras[f"{f}_lag{lag}"] = df[f].shift(lag)
    for a, b in INTERACTIONS:
        if a in df.columns and b in df.columns:
            extras[f"{a}__x__{b}"] = df[a] * df[b]
    if "btc_log_ret_1" in df.columns:
        extras["mom_5d_zscore"] = (
            df["btc_log_ret_1"].rolling(5).sum()
            / df["btc_log_ret_1"].rolling(20).std(ddof=0)
        )
        extras["mom_20d_zscore"] = (
            df["btc_log_ret_1"].rolling(20).sum()
            / df["btc_log_ret_1"].rolling(60).std(ddof=0)
        )
    if "intraday_rv_up" in df.columns and "intraday_rv_dn" in df.columns:
        extras["intraday_updown_ratio"] = (
            df["intraday_rv_up"] / df["intraday_rv_dn"].replace(0, np.nan)
        )
    extra_df = pd.DataFrame(extras, index=df.index)
    return pd.concat([df, extra_df], axis=1)


def build_targets(close: pd.Series, horizons=HORIZONS, vol_window=VOL_WINDOW):
    log_close = np.log(close)
    rv = log_close.diff().rolling(vol_window).std(ddof=0)
    rv_safe = rv.clip(lower=max(rv.quantile(0.05), 0.005))
    out = {"rv": rv_safe}
    for h in horizons:
        raw = log_close.shift(-h) - log_close
        out[f"target_raw_{h}"] = raw
        out[f"target_norm_{h}"] = (raw / (np.sqrt(h) * rv_safe)).clip(-5, 5)
    out[f"target_sign_{PRIMARY_HORIZON}"] = (
        out[f"target_raw_{PRIMARY_HORIZON}"] > 0
    ).astype(int)
    return pd.DataFrame(out, index=close.index)


def prepare_dataset(features_csv: Path, skip_extras: bool = False):
    df = pd.read_csv(features_csv, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    if not skip_extras:
        df = engineer_extra_features(df).copy()
    targets = build_targets(df["close"])
    full = pd.concat([df, targets], axis=1).copy()

    feat_cols = [c for c in full.columns
                 if c not in RAW_PRICE_COLS and not c.startswith("target_")
                 and c not in {"rv"}]
    coverage = full[feat_cols].notna().mean()
    keep = coverage[coverage >= COVERAGE_THRESHOLD].index.tolist()
    drop = sorted(set(feat_cols) - set(keep))
    print(f"  features: {len(feat_cols)} candidate (after extras), kept {len(keep)} "
          f"(>= {COVERAGE_THRESHOLD * 100:.0f}% coverage)")
    if drop:
        print(f"  dropped: {drop[:6]}{'...' if len(drop) > 6 else ''}")

    feat_df = full[keep]
    mask = feat_df.notna().all(axis=1)
    full = full[mask]
    print(f"  data shape after warmup drop: {full.shape}")
    print(f"  range: {full.index[0].date()} -> {full.index[-1].date()}")
    return full, keep


def exp_decay_weights(n: int, half_life: int = WEIGHT_HALF_LIFE) -> np.ndarray:
    decay = np.log(2) / half_life
    ages = np.arange(n)[::-1]
    return np.exp(-decay * ages).astype(np.float32)


def _val_ic(model, X_val, y_val, is_classifier=False):
    if len(X_val) < 5:
        return 0.0
    pred = (model.predict_proba(X_val)[:, 1] if is_classifier
            else model.predict(X_val))
    if np.std(pred) == 0 or np.std(y_val) == 0:
        return 0.0
    return float(np.corrcoef(pred, y_val)[0, 1])


def fit_xgb(model_cls, params, X, y, val_size=ES_VAL_SIZE, hl=WEIGHT_HALF_LIFE,
            is_classifier=False):
    n = len(X)
    val = min(val_size, max(n // 5, 10))
    if n - val < 50:
        p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        m = model_cls(**p)
        m.fit(X, y, sample_weight=exp_decay_weights(n, hl))
        return m, None, 0.0
    X_tr, y_tr = X[:-val], y[:-val]
    X_val, y_val = X[-val:], y[-val:]
    m = model_cls(**params)
    m.fit(X_tr, y_tr, sample_weight=exp_decay_weights(len(X_tr), hl),
          eval_set=[(X_val, y_val)], verbose=False)
    val_ic = _val_ic(m, X_val, y_val, is_classifier=is_classifier)
    return m, m.best_iteration, val_ic


def fit_ridge(X, y, val_size=ES_VAL_SIZE, hl=WEIGHT_HALF_LIFE, alpha=10.0):
    n = len(X)
    val = min(val_size, max(n // 5, 10))
    qt = QuantileTransformer(
        n_quantiles=min(n - val, 500),
        output_distribution="normal",
        subsample=min(n, 100_000), random_state=42)
    if n - val >= 50:
        qt.fit(X[:-val])
        X_tr_q = qt.transform(X[:-val])
        X_val_q = qt.transform(X[-val:])
        y_tr, y_val = y[:-val], y[-val:]
        m = Ridge(alpha=alpha)
        m.fit(X_tr_q, y_tr, sample_weight=exp_decay_weights(len(X_tr_q), hl))
        val_ic = _val_ic(m, X_val_q, y_val, is_classifier=False)
    else:
        qt.fit(X)
        m = Ridge(alpha=alpha)
        m.fit(qt.transform(X), y, sample_weight=exp_decay_weights(n, hl))
        val_ic = 0.0
    return qt, m, val_ic


def walk_forward_ensemble(X, dates, targets_df, initial_train, refit_every,
                          adaptive_sign: bool = True, sign_min_abs: float = 0.04,
                          oos_ic_lookback: int = 120, oos_ic_min_obs: int = 40):
    """Walk-forward ensemble with trailing-OOS-IC sign correction.

    For each test bar t and model m:
      1. Generate raw prediction p_t at time t (model fit on data < t).
      2. Look at trailing (raw_pred, actual) pairs for m where actual is
         already known at time t — i.e. for an h-step model, pairs at
         s ≤ t-h.
      3. Compute trailing-OOS IC over the last `oos_ic_lookback` such pairs.
         If |IC| > sign_min_abs, multiply p_t by sign(IC).

    This is honest: only past OOS evidence drives the sign. Until we have
    `oos_ic_min_obs` complete pairs, sign correction is disabled (sign=1).
    """
    n = len(X)
    raw_pred_xgb = {h: np.full(n, np.nan) for h in HORIZONS}
    raw_pred_ridge = np.full(n, np.nan)
    raw_pred_cls = np.full(n, np.nan)
    pred_xgb = {h: np.full(n, np.nan) for h in HORIZONS}
    pred_ridge = np.full(n, np.nan)
    pred_cls = np.full(n, np.nan)
    importances = []
    refit_meta = []
    val_ic_history = []
    sign_history = []  # sign decisions per refit
    last_refit = -1
    models = {}
    val_ics = {}

    target_arrays = {
        h: targets_df[f"target_norm_{h}"].to_numpy(dtype=np.float32)
        for h in HORIZONS
    }
    sign_array = targets_df[f"target_sign_{PRIMARY_HORIZON}"].to_numpy(dtype=np.float32)

    def trailing_ic(raw_preds, targets, end_excl, lookback):
        if end_excl <= 0:
            return None, 0
        a = raw_preds[:end_excl]
        b = targets[:end_excl]
        v = ~np.isnan(a) & ~np.isnan(b)
        if v.sum() < oos_ic_min_obs:
            return None, int(v.sum())
        idx = np.where(v)[0][-lookback:]
        a, b = a[idx], b[idx]
        if np.std(a) == 0 or np.std(b) == 0:
            return None, len(idx)
        return float(np.corrcoef(a, b)[0, 1]), len(idx)

    def model_sign_at(t, name):
        if not adaptive_sign:
            return 1.0, None, 0
        if name.startswith("xgb_h"):
            h = int(name.split("h")[-1])
            ic, n_obs = trailing_ic(raw_pred_xgb[h], target_arrays[h], t - h, oos_ic_lookback)
        elif name == "ridge":
            ic, n_obs = trailing_ic(raw_pred_ridge, target_arrays[PRIMARY_HORIZON],
                                    t - PRIMARY_HORIZON, oos_ic_lookback)
        elif name == "xgb_cls":
            ic, n_obs = trailing_ic(raw_pred_cls, sign_array,
                                    t - PRIMARY_HORIZON, oos_ic_lookback)
        else:
            return 1.0, None, 0
        if ic is None or abs(ic) < sign_min_abs:
            return 1.0, ic, n_obs
        return (-1.0 if ic < 0 else 1.0), ic, n_obs

    start = initial_train + max(HORIZONS)
    for t in range(start, n):
        if last_refit < 0 or (t - last_refit) >= refit_every:
            primary_imp = None
            best_iters = {}
            for h in HORIZONS:
                te = t - h
                X_tr = X[:te]
                y_tr = target_arrays[h][:te]
                v = ~np.isnan(y_tr)
                X_v, y_v = X_tr[v], y_tr[v]
                if len(X_v) < 100:
                    continue
                m, bi, vic = fit_xgb(xgb.XGBRegressor, XGB_REG_PARAMS, X_v, y_v)
                models[f"xgb_h{h}"] = m
                val_ics[f"xgb_h{h}"] = vic
                best_iters[f"xgb_h{h}"] = bi
                if h == PRIMARY_HORIZON:
                    primary_imp = m.feature_importances_.copy()

            te = t - PRIMARY_HORIZON
            X_tr = X[:te]
            y_tr = target_arrays[PRIMARY_HORIZON][:te]
            v = ~np.isnan(y_tr)
            if v.sum() >= 100:
                qt, ridge, vic = fit_ridge(X_tr[v], y_tr[v])
                models["ridge"] = (qt, ridge)
                val_ics["ridge"] = vic

            y_sign = sign_array[:te]
            v = ~np.isnan(y_sign)
            if v.sum() >= 100:
                m, bi, vic = fit_xgb(xgb.XGBClassifier, XGB_CLS_PARAMS,
                                     X_tr[v], y_sign[v].astype(int),
                                     is_classifier=True)
                models["xgb_cls"] = m
                val_ics["xgb_cls"] = vic
                best_iters["xgb_cls"] = bi

            if primary_imp is not None:
                importances.append(primary_imp)
                refit_meta.append({
                    "date": pd.Timestamp(dates[t]).strftime("%Y-%m-%d"),
                    "best_iters": best_iters,
                })
                val_ic_history.append({
                    "date": pd.Timestamp(dates[t]).strftime("%Y-%m-%d"),
                    **{k: float(v) for k, v in val_ics.items()},
                })
            last_refit = t

        X_t = X[t:t + 1]
        sign_record = {"date": pd.Timestamp(dates[t]).strftime("%Y-%m-%d")}
        for h in HORIZONS:
            key = f"xgb_h{h}"
            if key in models:
                raw = float(models[key].predict(X_t)[0])
                raw_pred_xgb[h][t] = raw
                sgn, ic, _ = model_sign_at(t, key)
                pred_xgb[h][t] = sgn * raw
                sign_record[f"{key}_sgn"] = sgn
                sign_record[f"{key}_ic"] = ic
        if "ridge" in models:
            qt, ridge = models["ridge"]
            raw = float(ridge.predict(qt.transform(X_t))[0])
            raw_pred_ridge[t] = raw
            sgn, ic, _ = model_sign_at(t, "ridge")
            pred_ridge[t] = sgn * raw
            sign_record["ridge_sgn"] = sgn
            sign_record["ridge_ic"] = ic
        if "xgb_cls" in models:
            raw = float(models["xgb_cls"].predict_proba(X_t)[0, 1])
            raw_pred_cls[t] = raw
            sgn, ic, _ = model_sign_at(t, "xgb_cls")
            pred_cls[t] = (1.0 - raw) if sgn < 0 else raw
            sign_record["xgb_cls_sgn"] = sgn
            sign_record["xgb_cls_ic"] = ic
        if (t - last_refit) == 1 or t == start:  # record signs at refit boundaries
            sign_history.append(sign_record)

    return (pred_xgb, pred_ridge, pred_cls, np.array(importances),
            refit_meta, val_ic_history, sign_history)


def build_regime_filter(full: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    rv = full["btc_rv_20"] if "btc_rv_20" in full.columns else full["rv"]
    rv_q = rv.rolling(config.regime_vol_lookback, min_periods=60).quantile(
        config.regime_vol_quantile)
    high_vol = (rv > rv_q).fillna(False)
    if "btcdom_log_ret_5" in full.columns:
        dom_extreme = (full["btcdom_log_ret_5"].abs()
                       > config.regime_dom_threshold).fillna(False)
    else:
        dom_extreme = pd.Series(False, index=full.index)
    return ~(high_vol | dom_extreme)


def run_strategy(dates, closes, expected_log_rets, cls_probs, tradable, config):
    n = len(dates)
    in_pos = False
    entry_price = entry_date = entry_expected = entry_cls = None
    entry_idx = -1
    n_units = 0.0
    cash = config.initial_capital
    cost = config.cost_bps_per_side / 10_000.0

    trades = []
    equity = np.zeros(n)
    position = np.zeros(n, dtype=int)

    for i in range(n):
        date = dates[i]
        close = float(closes[i])
        exp_ret = expected_log_rets[i]
        cp = cls_probs[i] if i < len(cls_probs) else np.nan
        is_tradable = bool(tradable[i]) if i < len(tradable) else True

        if in_pos:
            ret_unrealized = close / entry_price - 1
            bars_held = i - entry_idx
            exit_reason = None
            if ret_unrealized <= -config.stop_loss:
                exit_reason = "stop_loss"
            elif ret_unrealized >= config.take_profit:
                exit_reason = "take_profit"
            elif bars_held >= config.max_hold_bars:
                exit_reason = "time_stop"
            elif (not np.isnan(exp_ret) and exp_ret < config.exit_threshold):
                exit_reason = "signal_flip"

            if exit_reason is not None:
                cash = n_units * close * (1 - cost)
                gross_ret = close / entry_price - 1
                net_ret = (1 - cost) ** 2 * (close / entry_price) - 1
                trades.append({
                    "entry_date": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                    "exit_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "entry_price": float(entry_price),
                    "exit_price": close,
                    "bars_held": int(bars_held),
                    "exit_reason": exit_reason,
                    "pred_at_entry": float(entry_expected),
                    "cls_prob_at_entry": float(entry_cls) if entry_cls is not None else None,
                    "gross_return": float(gross_ret),
                    "net_return": float(net_ret),
                })
                in_pos = False
                n_units = 0.0
                entry_price = entry_date = entry_expected = entry_cls = None
                entry_idx = -1

        if (not in_pos and is_tradable
                and not np.isnan(exp_ret) and exp_ret > config.entry_threshold
                and (np.isnan(cp) or cp >= config.cls_min_prob)):
            n_units = (cash * (1 - cost)) / close
            cash = 0.0
            entry_price = close
            entry_date = date
            entry_expected = exp_ret
            entry_cls = cp
            entry_idx = i
            in_pos = True

        equity[i] = (n_units * close) if in_pos else cash
        position[i] = 1 if in_pos else 0

    return trades, equity, position


def compute_metrics(equity, dates, closes, trades, initial_capital):
    eq = np.asarray(equity, dtype=float)
    rets = np.diff(np.log(eq + 1e-9))
    bh_eq = closes / closes[0] * initial_capital
    bh_rets = np.diff(np.log(bh_eq))

    days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    years = max(days / 365.25, 1e-9)

    total = float(eq[-1] / initial_capital - 1)
    cagr = float((eq[-1] / initial_capital) ** (1 / years) - 1)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = (float(rets.mean() / downside.std() * np.sqrt(252))
               if len(downside) and downside.std() > 0 else 0.0)
    cm = np.maximum.accumulate(eq)
    max_dd = float(((eq - cm) / cm).min())

    bh_total = float(bh_eq[-1] / initial_capital - 1)
    bh_cagr = float((bh_eq[-1] / initial_capital) ** (1 / years) - 1)
    bh_sharpe = (float(bh_rets.mean() / bh_rets.std() * np.sqrt(252))
                 if bh_rets.std() > 0 else 0.0)
    bh_cm = np.maximum.accumulate(bh_eq)
    bh_max_dd = float(((bh_eq - bh_cm) / bh_cm).min())

    if trades:
        wins = [t for t in trades if t["net_return"] > 0]
        losses = [t for t in trades if t["net_return"] <= 0]
        win_rate = len(wins) / len(trades)
        avg_win = float(np.mean([t["net_return"] for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t["net_return"] for t in losses])) if losses else 0.0
        sw = sum(t["net_return"] for t in wins)
        sl = abs(sum(t["net_return"] for t in losses))
        pf = float(sw / sl) if sl > 0 else float("inf")
        avg_hold = float(np.mean([t["bars_held"] for t in trades]))
        eb = pd.Series([t["exit_reason"] for t in trades]).value_counts().to_dict()
    else:
        win_rate = avg_win = avg_loss = pf = avg_hold = 0.0
        eb = {}

    return {
        "total_return": total, "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": max_dd, "win_rate": float(win_rate),
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": pf if np.isfinite(pf) else None,
        "n_trades": len(trades), "avg_holding_days": avg_hold,
        "buy_hold_return": bh_total, "buy_hold_cagr": bh_cagr,
        "buy_hold_sharpe": bh_sharpe, "buy_hold_max_dd": bh_max_dd,
        "exit_breakdown": eb,
    }


def importance_stability(imp_matrix, feat_cols, top_k=10):
    n_refits, _ = imp_matrix.shape
    mean = imp_matrix.mean(axis=0)
    std = imp_matrix.std(axis=0)
    top_idx = np.argsort(-imp_matrix, axis=1)[:, :top_k]
    top_freq = np.zeros(len(feat_cols), dtype=int)
    for row in top_idx:
        for idx in row:
            top_freq[idx] += 1
    rows = []
    for i, f in enumerate(feat_cols):
        rows.append({
            "feature": f,
            "mean_importance": float(mean[i]),
            "std_importance": float(std[i]),
            f"top{top_k}_freq": int(top_freq[i]),
            f"top{top_k}_rate": float(top_freq[i] / max(n_refits, 1)),
        })
    rows.sort(key=lambda r: r["mean_importance"], reverse=True)
    return rows


def trailing_ic_matrix(preds_dict, actuals_dict, primary_h, lookback=120, min_obs=40):
    """Per-bar trailing IC of each model (lookahead-free).

    For each bar t and model m, compute IC over the last `lookback`
    completed (pred_s, actual_s) pairs where target_s is known at time t.
    For h-step model: usable pairs have s + h <= t.
    """
    out = {}
    for name, p in preds_dict.items():
        h = int(name.split("h")[-1]) if name.startswith("xgb_h") else primary_h
        a = actuals_dict[name]
        n = len(p)
        ic_arr = np.full(n, np.nan)
        for t in range(n):
            end = t - h
            if end <= 0:
                continue
            mask = ~np.isnan(p[:end]) & ~np.isnan(a[:end])
            if mask.sum() < min_obs:
                continue
            idx = np.where(mask)[0][-lookback:]
            pp, aa = p[idx], a[idx]
            if np.std(pp) == 0 or np.std(aa) == 0:
                continue
            ic_arr[t] = float(np.corrcoef(pp, aa)[0, 1])
        out[name] = ic_arr
    return out


def blend_ensemble(method, preds_dict, ic_matrix, top_k=None, ic_floor=0.0):
    """Blend per-model predictions per bar according to method.

    method:
      'mean'         equal-weight average (current default)
      'ic_weighted'  weight by |trailing IC|, fallback uniform until IC available
      'ic_filter'    keep only models with |IC| >= ic_floor (uniform among kept)
      'top_k'        at each bar, keep only top_k models by |IC|, equal-weight
    """
    names = list(preds_dict.keys())
    pred_stack = np.array([preds_dict[k] for k in names])
    ic_stack = np.array([ic_matrix.get(k, np.full_like(pred_stack[0], np.nan))
                         for k in names])
    n_t = pred_stack.shape[1]

    if method == "mean":
        weights = np.ones_like(pred_stack)
    elif method == "ic_weighted":
        weights = np.where(~np.isnan(ic_stack), np.abs(ic_stack), 0.0)
        no_ic_yet = np.isnan(ic_stack).all(axis=0)
        weights[:, no_ic_yet] = 1.0
    elif method == "ic_filter":
        weights = np.where(np.abs(ic_stack) >= ic_floor, 1.0, 0.0)
        no_ic_yet = np.isnan(ic_stack).all(axis=0)
        weights[:, no_ic_yet] = 1.0
    elif method == "top_k":
        weights = np.zeros_like(pred_stack)
        for t in range(n_t):
            ics_t = ic_stack[:, t]
            preds_t = pred_stack[:, t]
            valid = ~np.isnan(ics_t) & ~np.isnan(preds_t)
            if valid.sum() == 0:
                vmask = ~np.isnan(preds_t)
                weights[vmask, t] = 1.0
                continue
            valid_idx = np.where(valid)[0]
            ranks = np.argsort(-np.abs(ics_t[valid_idx]))[:top_k]
            for r in ranks:
                weights[valid_idx[r], t] = 1.0
    else:
        raise ValueError(f"Unknown method: {method}")

    weights = np.where(np.isnan(pred_stack), 0.0, weights)
    pred_clean = np.where(np.isnan(pred_stack), 0.0, pred_stack)
    weight_sum = weights.sum(axis=0)
    blended = (pred_clean * weights).sum(axis=0)
    return np.where(weight_sum > 0, blended / weight_sum, np.nan)


def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    v = ~np.isnan(a) & ~np.isnan(b)
    if v.sum() < 2:
        return float("nan")
    if np.std(a[v]) == 0 or np.std(b[v]) == 0:
        return float("nan")
    return float(np.corrcoef(a[v], b[v])[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", default="btc_data/features_BTCUSDT_1d.csv")
    parser.add_argument("--out", default="btc_data/backtest_results.json")
    parser.add_argument("--no_regime", action="store_true")
    parser.add_argument("--no_extras", action="store_true",
                        help="skip lag+interaction feature engineering (ablation)")
    parser.add_argument("--contrarian", action="store_true",
                        help="flip ensemble sign at strategy level (use as contrarian)")
    parser.add_argument("--cls_min_prob", type=float, default=0.50,
                        help="classifier probability gate (use 0.0 to disable)")
    parser.add_argument("--entry_threshold", type=float, default=0.005)
    parser.add_argument("--no_adaptive_sign", action="store_true",
                        help="disable per-refit val-IC sign correction")
    args = parser.parse_args()

    print("Loading & engineering features...")
    full, feat_cols = prepare_dataset(Path(args.features_csv), skip_extras=args.no_extras)

    X = full[feat_cols].to_numpy(dtype=np.float32)
    closes = full["close"].to_numpy(dtype=np.float64)
    dates = full.index.to_numpy()
    rv = full["rv"].to_numpy(dtype=np.float64)
    target_norm5 = full[f"target_norm_{PRIMARY_HORIZON}"].to_numpy(dtype=np.float32)
    target_raw5 = full[f"target_raw_{PRIMARY_HORIZON}"].to_numpy(dtype=np.float32)
    target_sign5 = full[f"target_sign_{PRIMARY_HORIZON}"].to_numpy(dtype=np.float32)
    targets_df = full[[f"target_norm_{h}" for h in HORIZONS]
                      + [f"target_sign_{PRIMARY_HORIZON}"]]

    initial_train = int(len(full) * INITIAL_TRAIN_FRAC)
    print(f"\nWalk-forward ensemble: initial_train={initial_train}, "
          f"refit_every={REFIT_EVERY}, horizons={HORIZONS}")
    print("  base models per refit: 4× XGB-reg + 1× Ridge-QT + 1× XGB-cls")
    print("  sample weights: exp decay (half-life=250d), early stopping (val=60d)")

    pred_xgb, pred_ridge, pred_cls, imp_mat, refit_meta, val_ic_history, sign_history = walk_forward_ensemble(
        X, dates, targets_df, initial_train, REFIT_EVERY,
        adaptive_sign=not args.no_adaptive_sign,
    )

    preds_dict = {f"xgb_h{h}": pred_xgb[h] for h in HORIZONS}
    preds_dict["ridge"] = pred_ridge
    actuals_dict = {f"xgb_h{h}": targets_df[f"target_norm_{h}"].to_numpy(np.float32)
                    for h in HORIZONS}
    actuals_dict["ridge"] = targets_df[f"target_norm_{PRIMARY_HORIZON}"].to_numpy(np.float32)

    print("\nComputing trailing-OOS-IC matrix for blending...")
    ic_matrix = trailing_ic_matrix(preds_dict, actuals_dict, PRIMARY_HORIZON,
                                   lookback=120, min_obs=40)

    method_specs = [
        ("mean", dict(method="mean")),
        ("ic_weighted", dict(method="ic_weighted")),
        ("ic_filter_0.05", dict(method="ic_filter", ic_floor=0.05)),
        ("top_1", dict(method="top_k", top_k=1)),
        ("top_2", dict(method="top_k", top_k=2)),
        ("top_3", dict(method="top_k", top_k=3)),
    ]

    config = StrategyConfig(
        cls_min_prob=args.cls_min_prob,
        entry_threshold=args.entry_threshold,
    )
    if args.no_regime:
        tradable = pd.Series(True, index=full.index)
        regime_blocked = 0
    else:
        tradable = build_regime_filter(full, config)
        regime_blocked = int((~tradable).sum())
    tradable_arr = tradable.to_numpy()

    sign = -1.0 if args.contrarian else 1.0
    cls_prob_adj = (1.0 - pred_cls) if args.contrarian else pred_cls

    method_results = {}
    print("\n=== Ensemble method comparison ===")
    print(f"  {'method':18s} {'IC_raw':>8s} {'IC_norm':>8s} {'Sharpe':>7s} "
          f"{'Total':>8s} {'PF':>5s} {'MaxDD':>7s} {'N':>4s}")
    for label, kwargs in method_specs:
        ens_norm = blend_ensemble(preds_dict=preds_dict, ic_matrix=ic_matrix, **kwargs)
        exp_ret = sign * ens_norm * np.sqrt(PRIMARY_HORIZON) * rv
        ic_raw_m = safe_corr(exp_ret, target_raw5)
        ic_norm_m = safe_corr(ens_norm, target_norm5)
        trades_m, equity_m, position_m = run_strategy(
            dates, closes, exp_ret, cls_prob_adj, tradable_arr, config
        )
        metrics_m = compute_metrics(equity_m, dates, closes, trades_m,
                                    config.initial_capital)
        method_results[label] = {
            "ensemble_norm": ens_norm,
            "expected_ret": exp_ret,
            "ic_raw": ic_raw_m,
            "ic_norm": ic_norm_m,
            "metrics": metrics_m,
            "trades": trades_m,
            "equity": equity_m,
            "position": position_m,
        }
        pf = metrics_m["profit_factor"]
        pf_str = "inf" if pf is None else f"{pf:.2f}"
        print(f"  {label:18s} {ic_raw_m:+.4f}  {ic_norm_m:+.4f}  "
              f"{metrics_m['sharpe']:+.3f}  {metrics_m['total_return']:+.2%}  "
              f"{pf_str:>5s}  {metrics_m['max_drawdown']:+.2%}  "
              f"{metrics_m['n_trades']:>4d}")

    eligible = [(lbl, r) for lbl, r in method_results.items()
                if r["metrics"]["n_trades"] >= 15]
    if not eligible:
        eligible = list(method_results.items())
    best_label, best_res = max(
        eligible,
        key=lambda kv: (kv[1]["metrics"]["sharpe"], kv[1]["metrics"]["total_return"])
    )
    print(f"\nBest method (by Sharpe, ≥15 trades): {best_label}")

    ensemble_norm = best_res["ensemble_norm"]
    expected_ret = best_res["expected_ret"]
    ic_norm = best_res["ic_norm"]
    ic_raw = best_res["ic_raw"]
    trades = best_res["trades"]
    equity = best_res["equity"]
    position = best_res["position"]
    metrics = best_res["metrics"]

    per_model_ic = {}
    for h in HORIZONS:
        per_model_ic[f"xgb_h{h}"] = safe_corr(pred_xgb[h], target_norm5)
    per_model_ic["ridge_h5"] = safe_corr(pred_ridge, target_norm5)
    per_model_ic["ensemble_z"] = safe_corr(ensemble_norm, target_norm5)
    per_model_ic["xgb_cls_vs_sign"] = safe_corr(pred_cls, target_sign5)

    ic_norm = per_model_ic["ensemble_z"]
    ic_raw = safe_corr(expected_ret, target_raw5)
    valid_raw = ~np.isnan(expected_ret) & ~np.isnan(target_raw5)
    rmse_raw = (float(np.sqrt(((expected_ret[valid_raw] - target_raw5[valid_raw]) ** 2).mean()))
                if valid_raw.sum() else float("nan"))

    print("\nPer-model IC (vs target_norm_5):")
    for k, v in per_model_ic.items():
        print(f"  {k:24s}: {v:+.4f}")
    print(f"  ensemble IC(raw):       : {ic_raw:+.4f}")
    print(f"  RMSE(raw):              : {rmse_raw:.4f}")
    print(f"  refits: {len(refit_meta)}")

    if val_ic_history:
        print("\nVal IC summary across refits (only used pre-OOS warmup):")
        keys = [k for k in val_ic_history[0].keys() if k != "date"]
        for k in keys:
            vals = [r[k] for r in val_ic_history if k in r and r[k] is not None]
            if vals:
                arr = np.array(vals)
                neg_frac = float((arr < 0).mean())
                print(f"  {k:12s}  mean={arr.mean():+.3f}  std={arr.std():.3f}  "
                      f"neg_frac={neg_frac:.0%}")

    print(f"\nRegime filter: {regime_blocked} bars blocked "
          f"({regime_blocked / len(full) * 100:.1f}%)")
    print(f"Strategy: {config}")
    print(f"\nFinal metrics ({best_label}):")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k:22s}: {v}")
        elif isinstance(v, float):
            print(f"  {k:22s}: {v:+.4f}")
        else:
            print(f"  {k:22s}: {v}")

    imp_stab = importance_stability(imp_mat, feat_cols, top_k=10) if len(imp_mat) else []
    final_imp = sorted(
        [{"feature": f, "importance": float(imp)}
         for f, imp in zip(feat_cols, imp_mat[-1])],
        key=lambda x: x["importance"], reverse=True
    ) if len(imp_mat) else []

    cm = np.maximum.accumulate(equity)
    dd = (equity - cm) / cm
    bh_eq = closes / closes[0] * config.initial_capital
    equity_curve = [
        {"date": pd.Timestamp(d).strftime("%Y-%m-%d"),
         "strategy": float(eq), "buy_hold": float(bh),
         "drawdown": float(dd_), "in_position": int(p), "tradable": int(t)}
        for d, eq, bh, dd_, p, t in zip(dates, equity, bh_eq, dd, position, tradable_arr)
    ]

    predictions = [
        {"date": pd.Timestamp(d).strftime("%Y-%m-%d"),
         "close": float(c),
         "predicted_5d": (None if np.isnan(p) else float(p)),
         "actual_5d": (None if np.isnan(yi) else float(yi)),
         "predicted_norm": (None if np.isnan(pn) else float(pn)),
         "actual_norm": (None if np.isnan(yn) else float(yn)),
         "cls_prob": (None if np.isnan(cp) else float(cp))}
        for d, c, p, yi, pn, yn, cp in zip(
            dates, closes, expected_ret, target_raw5,
            ensemble_norm, target_norm5, pred_cls)
    ]

    out = {
        "config": asdict(config),
        "horizon": PRIMARY_HORIZON,
        "horizons": HORIZONS,
        "vol_window": VOL_WINDOW,
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "ic_norm": ic_norm,
        "ic_raw": ic_raw,
        "rmse_raw": rmse_raw,
        "refits": len(refit_meta),
        "per_model_ic": per_model_ic,
        "metrics": metrics,
        "feature_importance": final_imp,
        "importance_stability": imp_stab,
        "best_params": {k: v for k, v in XGB_REG_PARAMS.items() if k != "n_jobs"},
        "regime_filter_enabled": not args.no_regime,
        "regime_blocked_bars": regime_blocked,
        "predictions": predictions,
        "trades": trades,
        "equity_curve": equity_curve,
        "date_range": {"start": pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
                       "end": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")},
        "ensemble_meta": {
            "regression_models": [f"xgb_h{h}" for h in HORIZONS] + ["ridge_h5"],
            "classifier": "xgb_cls",
            "weight_half_life": WEIGHT_HALF_LIFE,
            "es_val_size": ES_VAL_SIZE,
            "adaptive_sign": not args.no_adaptive_sign,
            "contrarian": args.contrarian,
        },
        "val_ic_history": val_ic_history,
        "sign_history": sign_history,
        "best_method": best_label,
        "method_comparison": [
            {
                "method": lbl,
                "ic_raw": r["ic_raw"], "ic_norm": r["ic_norm"],
                "sharpe": r["metrics"]["sharpe"],
                "total_return": r["metrics"]["total_return"],
                "profit_factor": r["metrics"]["profit_factor"],
                "max_drawdown": r["metrics"]["max_drawdown"],
                "n_trades": r["metrics"]["n_trades"],
                "win_rate": r["metrics"]["win_rate"],
            }
            for lbl, r in method_results.items()
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
