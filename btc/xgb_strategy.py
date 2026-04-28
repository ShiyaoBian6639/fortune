"""XGBoost-based BTC price prediction with walk-forward training and backtest.

Improvements (v2):
  1. Vol-normalized target  : predict z-score = fwd_log_ret / (sqrt(h)*rv);
     denormalize to expected log return for trading decisions.
  3. Intraday features       : aggregated from 5m klines (RV, bipower, jump,
     skew, kurt, taker imbalance, intraday drawup/down, vol concentration).
  4. Regime gating           : skip new entries when realized vol is in top
     15% of trailing 250d, or BTCDOM 5d-move is extreme.
  5. Hyperparameter tuning   : TimeSeriesSplit grid search on first 50% of
     data; tuned params used for all walk-forward refits.
  6. Importance stability    : track full importance matrix across refits;
     surface mean / std / top-10 frequency for dashboard.

Usage:
    ./venv/Scripts/python -m btc.xgb_strategy
    ./venv/Scripts/python -m btc.xgb_strategy --no_tune        # skip HP tuning
    ./venv/Scripts/python -m btc.xgb_strategy --no_regime      # disable gate
"""

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

HORIZON = 5
VOL_WINDOW = 20
COVERAGE_THRESHOLD = 0.90
INITIAL_TRAIN_FRAC = 0.5
REFIT_EVERY = 30

XGB_BASE = dict(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

XGB_DEFAULT = dict(XGB_BASE,
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
)

RAW_PRICE_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
                  "trades", "taker_buy_base", "taker_buy_quote",
                  "btcdom_close", "perp_close", "perp_volume"]


@dataclass
class StrategyConfig:
    entry_threshold: float = 0.005
    exit_threshold: float = -0.002
    stop_loss: float = 0.03
    take_profit: float = 0.06
    max_hold_bars: int = 5
    cost_bps_per_side: float = 5.0
    initial_capital: float = 10_000.0
    regime_vol_quantile: float = 0.85
    regime_vol_lookback: int = 250
    regime_dom_threshold: float = 0.05  # |btcdom 5d ret| >= 5% blocks entry


def prepare_dataset(features_csv: Path, horizon: int = HORIZON, vol_window: int = VOL_WINDOW):
    df = pd.read_csv(features_csv, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    close = df["close"]
    log_close = np.log(close)
    target_raw = log_close.shift(-horizon) - log_close
    rv = log_close.diff().rolling(vol_window).std(ddof=0)
    rv_floor = max(rv.quantile(0.05), 0.005)
    rv_safe = rv.clip(lower=rv_floor)
    target_norm = (target_raw / (np.sqrt(horizon) * rv_safe)).clip(-5, 5)

    df["__target_raw__"] = target_raw
    df["__target_norm__"] = target_norm
    df["__rv__"] = rv_safe

    feat_cols = [c for c in df.columns
                 if c not in RAW_PRICE_COLS and not c.startswith("__")]
    coverage = df[feat_cols].notna().mean()
    keep = coverage[coverage >= COVERAGE_THRESHOLD].index.tolist()
    drop = sorted(set(feat_cols) - set(keep))
    print(f"  features: {len(feat_cols)} candidate, kept {len(keep)} "
          f"(>= {COVERAGE_THRESHOLD * 100:.0f}% coverage)")
    if drop:
        print(f"  dropped (low coverage): {drop[:6]}{'...' if len(drop) > 6 else ''}")

    feat_df = df[keep].copy()
    mask = feat_df.notna().all(axis=1)
    full = pd.concat([feat_df, df["close"], df["__target_raw__"],
                      df["__target_norm__"], df["__rv__"]], axis=1)
    full = full.rename(columns={"__target_raw__": "target_raw",
                                "__target_norm__": "target_norm",
                                "__rv__": "rv"})
    full = full[mask]
    print(f"  data shape after warmup drop: {full.shape}")
    print(f"  range: {full.index[0].date()} -> {full.index[-1].date()}")
    return full, keep


def build_regime_filter(full: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """Return boolean Series: True = entries allowed, False = entries blocked."""
    rv = full["btc_rv_20"] if "btc_rv_20" in full.columns else full["rv"]
    rv_q = rv.rolling(config.regime_vol_lookback, min_periods=60).quantile(
        config.regime_vol_quantile
    )
    high_vol = (rv > rv_q).fillna(False)

    if "btcdom_log_ret_5" in full.columns:
        dom_extreme = (full["btcdom_log_ret_5"].abs() > config.regime_dom_threshold).fillna(False)
    else:
        dom_extreme = pd.Series(False, index=full.index)

    block = high_vol | dom_extreme
    return ~block


def tune_hyperparams(X, y, horizon: int, n_splits: int = 4, verbose: bool = True):
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    if len(X) < 200:
        if verbose:
            print("  too few rows, returning default params")
        return XGB_DEFAULT, []

    grid = list(product(
        [3, 4, 6],            # max_depth
        [0.02, 0.05],         # learning_rate
        [200, 400],           # n_estimators
    ))

    test_size = max(60, len(X) // (n_splits + 2))
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    results = []
    for i, (md, lr, ne) in enumerate(grid):
        params = dict(XGB_BASE, max_depth=md, learning_rate=lr, n_estimators=ne,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5)
        ics = []
        for tr_idx, te_idx in tscv.split(X):
            tr_idx = tr_idx[:max(len(tr_idx) - horizon, 1)]
            model = xgb.XGBRegressor(**params)
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[te_idx])
            if np.std(preds) > 0 and np.std(y[te_idx]) > 0:
                ics.append(float(np.corrcoef(preds, y[te_idx])[0, 1]))
        mean_ic = float(np.nanmean(ics)) if ics else float("nan")
        results.append({"max_depth": md, "learning_rate": lr,
                        "n_estimators": ne, "mean_ic": mean_ic,
                        "fold_ics": ics})
        if verbose:
            print(f"  [{i + 1}/{len(grid)}] depth={md} lr={lr} n={ne}  IC={mean_ic:+.4f}")

    results.sort(key=lambda r: r["mean_ic"] if not np.isnan(r["mean_ic"]) else -1, reverse=True)
    best = results[0]
    if np.isnan(best["mean_ic"]):
        return XGB_DEFAULT, results
    best_params = dict(XGB_BASE,
        max_depth=best["max_depth"], learning_rate=best["learning_rate"],
        n_estimators=best["n_estimators"],
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
    )
    if verbose:
        print(f"  best: depth={best['max_depth']} lr={best['learning_rate']} "
              f"n={best['n_estimators']}  IC={best['mean_ic']:+.4f}")
    return best_params, results


def walk_forward_predict(X, y, dates, initial_train, refit_every, horizon, params):
    n = len(X)
    preds = np.full(n, np.nan, dtype=np.float64)
    importance_matrix = []  # list of np arrays
    refit_dates = []
    last_refit = -1
    model = None

    start = initial_train + horizon
    for t in range(start, n):
        if last_refit < 0 or (t - last_refit) >= refit_every:
            train_end = t - horizon
            X_tr = X[:train_end]
            y_tr = y[:train_end]
            valid = ~np.isnan(y_tr)
            X_tr = X_tr[valid]
            y_tr = y_tr[valid]
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            importance_matrix.append(model.feature_importances_.copy())
            refit_dates.append(pd.Timestamp(dates[t]).strftime("%Y-%m-%d"))
            last_refit = t
        preds[t] = float(model.predict(X[t:t + 1])[0])

    return preds, np.array(importance_matrix), refit_dates


def run_strategy(dates, closes, expected_log_rets, raw_preds, tradable, config: StrategyConfig):
    n = len(dates)
    in_pos = False
    entry_price = entry_date = entry_pred_norm = entry_expected = None
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
        pred_norm = raw_preds[i]
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
            elif not np.isnan(exp_ret) and exp_ret < config.exit_threshold:
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
                    "pred_norm_at_entry": float(entry_pred_norm),
                    "gross_return": float(gross_ret),
                    "net_return": float(net_ret),
                })
                in_pos = False
                n_units = 0.0
                entry_price = entry_date = entry_pred_norm = entry_expected = None
                entry_idx = -1

        if (not in_pos and is_tradable and not np.isnan(exp_ret)
                and exp_ret > config.entry_threshold):
            n_units = (cash * (1 - cost)) / close
            cash = 0.0
            entry_price = close
            entry_date = date
            entry_pred_norm = pred_norm
            entry_expected = exp_ret
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

    total_ret = float(eq[-1] / initial_capital - 1)
    cagr = float((eq[-1] / initial_capital) ** (1 / years) - 1)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = (float(rets.mean() / downside.std() * np.sqrt(252))
               if len(downside) and downside.std() > 0 else 0.0)
    cummax = np.maximum.accumulate(eq)
    max_dd = float(((eq - cummax) / cummax).min())

    bh_total = float(bh_eq[-1] / initial_capital - 1)
    bh_cagr = float((bh_eq[-1] / initial_capital) ** (1 / years) - 1)
    bh_sharpe = float(bh_rets.mean() / bh_rets.std() * np.sqrt(252)) if bh_rets.std() > 0 else 0.0
    bh_cummax = np.maximum.accumulate(bh_eq)
    bh_max_dd = float(((bh_eq - bh_cummax) / bh_cummax).min())

    if trades:
        wins = [t for t in trades if t["net_return"] > 0]
        losses = [t for t in trades if t["net_return"] <= 0]
        win_rate = len(wins) / len(trades)
        avg_win = float(np.mean([t["net_return"] for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t["net_return"] for t in losses])) if losses else 0.0
        sum_w = sum(t["net_return"] for t in wins)
        sum_l = abs(sum(t["net_return"] for t in losses))
        profit_factor = float(sum_w / sum_l) if sum_l > 0 else float("inf")
        avg_hold = float(np.mean([t["bars_held"] for t in trades]))
        exit_breakdown = pd.Series([t["exit_reason"] for t in trades]).value_counts().to_dict()
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0.0
        exit_breakdown = {}

    return {
        "total_return": total_ret, "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": max_dd, "win_rate": float(win_rate),
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor if np.isfinite(profit_factor) else None,
        "n_trades": len(trades), "avg_holding_days": avg_hold,
        "buy_hold_return": bh_total, "buy_hold_cagr": bh_cagr,
        "buy_hold_sharpe": bh_sharpe, "buy_hold_max_dd": bh_max_dd,
        "exit_breakdown": exit_breakdown,
    }


def importance_stability(imp_matrix: np.ndarray, feat_cols: list, top_k: int = 10):
    """Per-feature mean/std + top-K frequency across refits."""
    n_refits, n_feats = imp_matrix.shape
    mean = imp_matrix.mean(axis=0)
    std = imp_matrix.std(axis=0)
    cv = std / np.where(mean > 1e-9, mean, np.nan)
    top_indices_per_refit = np.argsort(-imp_matrix, axis=1)[:, :top_k]
    top_freq = np.zeros(n_feats, dtype=int)
    for row in top_indices_per_refit:
        for idx in row:
            top_freq[idx] += 1
    rows = []
    for i, f in enumerate(feat_cols):
        rows.append({
            "feature": f,
            "mean_importance": float(mean[i]),
            "std_importance": float(std[i]),
            "cv": float(cv[i]) if np.isfinite(cv[i]) else None,
            f"top{top_k}_freq": int(top_freq[i]),
            f"top{top_k}_rate": float(top_freq[i] / n_refits),
        })
    rows.sort(key=lambda r: r["mean_importance"], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", default="btc_data/features_BTCUSDT_1d.csv")
    parser.add_argument("--out", default="btc_data/backtest_results.json")
    parser.add_argument("--no_tune", action="store_true")
    parser.add_argument("--no_regime", action="store_true")
    args = parser.parse_args()

    print("Loading & preparing data...")
    full, feat_cols = prepare_dataset(Path(args.features_csv))

    X = full[feat_cols].to_numpy(dtype=np.float32)
    y_raw = full["target_raw"].to_numpy(dtype=np.float32)
    y_norm = full["target_norm"].to_numpy(dtype=np.float32)
    rv = full["rv"].to_numpy(dtype=np.float64)
    closes = full["close"].to_numpy(dtype=np.float64)
    dates = full.index.to_numpy()

    initial_train = int(len(full) * INITIAL_TRAIN_FRAC)

    if args.no_tune:
        print("\nUsing default XGB params (HP tuning skipped)")
        best_params = XGB_DEFAULT
        tuning_results = []
    else:
        print(f"\nTuning hyperparameters on first {initial_train} bars (TimeSeriesSplit, 4 folds)...")
        best_params, tuning_results = tune_hyperparams(
            X[:initial_train], y_norm[:initial_train], HORIZON
        )

    print(f"\nWalk-forward XGB on normalized target:")
    print(f"  initial_train={initial_train}, refit_every={REFIT_EVERY}, horizon={HORIZON}d")
    preds_norm, imp_matrix, refit_dates = walk_forward_predict(
        X, y_norm, dates, initial_train, REFIT_EVERY, HORIZON, best_params
    )

    expected_log_rets = preds_norm * np.sqrt(HORIZON) * rv

    n_pred = int((~np.isnan(preds_norm)).sum())
    valid = ~np.isnan(preds_norm) & ~np.isnan(y_norm)
    ic_norm = float(pd.Series(preds_norm[valid]).corr(pd.Series(y_norm[valid]))) if valid.sum() > 1 else float("nan")
    valid_raw = ~np.isnan(expected_log_rets) & ~np.isnan(y_raw)
    ic_raw = float(pd.Series(expected_log_rets[valid_raw]).corr(pd.Series(y_raw[valid_raw]))) if valid_raw.sum() > 1 else float("nan")
    rmse_raw = float(np.sqrt(((expected_log_rets[valid_raw] - y_raw[valid_raw]) ** 2).mean())) if valid_raw.sum() else float("nan")
    print(f"  preds: {n_pred}  IC(norm): {ic_norm:+.4f}  IC(raw): {ic_raw:+.4f}  RMSE: {rmse_raw:.4f}  refits: {len(refit_dates)}")

    config = StrategyConfig()
    if args.no_regime:
        tradable = pd.Series(True, index=full.index)
        regime_blocked = 0
    else:
        tradable = build_regime_filter(full, config)
        regime_blocked = int((~tradable).sum())
    print(f"\nRegime filter: {regime_blocked} bars blocked ({regime_blocked / len(full) * 100:.1f}%)")

    print(f"Strategy: {config}")
    trades, equity, position = run_strategy(
        dates, closes, expected_log_rets, preds_norm, tradable.to_numpy(), config
    )
    print(f"  trades: {len(trades)}")

    metrics = compute_metrics(equity, dates, closes, trades, config.initial_capital)
    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k:22s}: {v}")
        elif isinstance(v, float):
            print(f"  {k:22s}: {v:+.4f}")
        else:
            print(f"  {k:22s}: {v}")

    imp_stability = importance_stability(imp_matrix, feat_cols, top_k=10)
    final_imp = sorted(
        [{"feature": f, "importance": float(imp)}
         for f, imp in zip(feat_cols, imp_matrix[-1])],
        key=lambda x: x["importance"], reverse=True,
    )

    cummax = np.maximum.accumulate(equity)
    dd = (equity - cummax) / cummax
    bh_eq = closes / closes[0] * config.initial_capital
    tradable_arr = tradable.to_numpy()
    equity_curve = [
        {"date": pd.Timestamp(d).strftime("%Y-%m-%d"),
         "strategy": float(eq), "buy_hold": float(bh),
         "drawdown": float(dd_), "in_position": int(p),
         "tradable": int(t)}
        for d, eq, bh, dd_, p, t in zip(dates, equity, bh_eq, dd, position, tradable_arr)
    ]

    predictions = [
        {"date": pd.Timestamp(d).strftime("%Y-%m-%d"),
         "close": float(c),
         "predicted_5d": (None if np.isnan(p) else float(p)),
         "actual_5d": (None if np.isnan(yi) else float(yi)),
         "predicted_norm": (None if np.isnan(pn) else float(pn)),
         "actual_norm": (None if np.isnan(yn) else float(yn))}
        for d, c, p, yi, pn, yn in zip(dates, closes, expected_log_rets, y_raw, preds_norm, y_norm)
    ]

    out = {
        "config": asdict(config),
        "horizon": HORIZON, "vol_window": VOL_WINDOW,
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "ic_norm": ic_norm, "ic_raw": ic_raw, "rmse_raw": rmse_raw,
        "refits": len(refit_dates),
        "metrics": metrics,
        "feature_importance": final_imp,
        "importance_stability": imp_stability,
        "tuning_results": tuning_results,
        "best_params": {k: v for k, v in best_params.items() if k != "n_jobs"},
        "regime_filter_enabled": not args.no_regime,
        "regime_blocked_bars": regime_blocked,
        "predictions": predictions,
        "trades": trades,
        "equity_curve": equity_curve,
        "date_range": {"start": pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
                       "end": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
