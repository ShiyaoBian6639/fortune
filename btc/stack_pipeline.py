"""Stacked + quantile-sized BTC pipeline.

Builds on multi_model_pipeline:
  1. Fit base models (xgb, lgb, mlp, dcnn, transformer, catboost, tabnet)
     walk-forward, generate sign-corrected predictions for each.
  2. Train a Ridge meta-learner on (base_preds, target) using a strict
     walk-forward expanding window — meta-learner only ever sees actually
     completed (pred, actual) pairs from the past.
  3. Quantile LightGBM is fit separately to estimate q25/q50/q75; the
     interquartile spread acts as an uncertainty signal for position sizing.
  4. Strategy variants tested:
        single best base model         (mlp, etc.)
        equal-weight mean              (current best top1 — but here mean)
        ridge-stacked predictions
        ridge-stacked + quantile sizing
     Position size scales with confidence = 1 - normalized_spread, clipped
     to [0.25, 1.0]. (Smaller positions when uncertainty is high.)

Output -> btc_data/backtest_results.json (overwrites; consumed by dashboard).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from btc.advanced_pipeline import (
    HORIZONS, PRIMARY_HORIZON, INITIAL_TRAIN_FRAC, REFIT_EVERY,
    WEIGHT_HALF_LIFE, ES_VAL_SIZE,
    StrategyConfig, prepare_dataset, build_regime_filter,
    compute_metrics, exp_decay_weights, safe_corr,
)
from btc.multi_model_pipeline import (
    walk_forward_single, apply_sign_correction, hit_rate,
)
from xgbmodel import btc_models

warnings.filterwarnings("ignore", category=UserWarning)


def walk_forward_quantile(X, y_norm, dates, initial_train, refit_every,
                          horizon, hl=WEIGHT_HALF_LIFE):
    """Walk-forward QuantileLGB to produce q25/q50/q75 streams."""
    n = len(X)
    q25 = np.full(n, np.nan)
    q50 = np.full(n, np.nan)
    q75 = np.full(n, np.nan)
    last_refit = -1
    model = None
    start = initial_train + horizon
    for t in range(start, n):
        if last_refit < 0 or (t - last_refit) >= refit_every:
            te = t - horizon
            valid = ~np.isnan(y_norm[:te])
            if valid.sum() < 100:
                last_refit = t
                continue
            sw = exp_decay_weights(te, hl).astype(np.float32)
            sw = np.where(valid, sw, 0.0).astype(np.float32)
            y_pass = y_norm.copy()
            y_pass[:te] = np.where(valid, y_norm[:te], 0.0)
            model = btc_models.QuantileLGBReg()
            model.fit(X, y_pass, end_idx=te, sample_weight=sw,
                      val_size=ES_VAL_SIZE)
            last_refit = t
        if model is not None:
            try:
                a, b, c = model.predict_quantiles(X, t)
                q25[t], q50[t], q75[t] = a, b, c
            except Exception:
                pass
    return q25, q50, q75


def stack_meta_learner(base_preds_dict, target_norm, horizon,
                       lookback_min=80, refit_every=15, alpha=5.0):
    """Walk-forward Ridge meta-learner on base-model predictions.

    For each test bar t, fit Ridge on completed (base_preds[s], target[s])
    pairs for s such that target_s is known at time t (i.e. s+h <= t-1).
    Refit every `refit_every` bars to amortize.

    Returns: meta_pred (n,) and per-bar weight matrix (rows = refits).
    """
    names = list(base_preds_dict.keys())
    P = np.array([base_preds_dict[k] for k in names])  # (M, n)
    n_t = P.shape[1]
    meta_pred = np.full(n_t, np.nan)
    last_refit = -1
    coefs = None
    intercept = None
    coef_history = []

    for t in range(n_t):
        usable_end = t - horizon
        if usable_end <= 0:
            continue
        valid = ~np.isnan(P[:, :usable_end]).any(axis=0) & ~np.isnan(target_norm[:usable_end])
        if valid.sum() < lookback_min:
            continue

        if last_refit < 0 or (t - last_refit) >= refit_every:
            X_meta = P[:, :usable_end].T[valid]
            y_meta = target_norm[:usable_end][valid]
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_meta, y_meta)
            coefs = ridge.coef_.copy()
            intercept = float(ridge.intercept_)
            coef_history.append({
                "date_idx": int(t),
                "coefs": dict(zip(names, [float(c) for c in coefs])),
                "intercept": intercept,
            })
            last_refit = t

        if coefs is not None and not np.isnan(P[:, t]).any():
            meta_pred[t] = float(coefs @ P[:, t] + intercept)

    return meta_pred, coef_history, names


def run_strategy_with_size(dates, closes, expected_log_rets, position_sizes,
                           tradable, config: StrategyConfig):
    """Variant of run_strategy with per-bar position sizing in [0, 1]."""
    n = len(dates)
    in_pos = False
    entry_price = entry_date = entry_expected = None
    entry_idx = -1
    entry_size = 0.0
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
        is_tradable = bool(tradable[i]) if i < len(tradable) else True
        size = float(position_sizes[i]) if i < len(position_sizes) else 1.0
        size = max(0.0, min(1.0, size))

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
                cash += n_units * close * (1 - cost)
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
                    "size_at_entry": float(entry_size),
                    "gross_return": float(gross_ret),
                    "net_return": float(net_ret),
                })
                in_pos = False
                n_units = 0.0
                entry_price = entry_date = entry_expected = None
                entry_idx = -1
                entry_size = 0.0

        if (not in_pos and is_tradable and not np.isnan(exp_ret)
                and exp_ret > config.entry_threshold and size > 0.05):
            stake = cash * size
            n_units = (stake * (1 - cost)) / close
            cash -= stake
            entry_price = close
            entry_date = date
            entry_expected = exp_ret
            entry_size = size
            entry_idx = i
            in_pos = True

        equity[i] = (n_units * close if in_pos else 0.0) + cash
        position[i] = 1 if in_pos else 0

    return trades, equity, position


def confidence_from_spread(spread, lookback=120, lo_pct=0.5, hi_pct=1.0):
    """Map quantile spread → position size in [lo_pct, hi_pct].

    Larger spread = more uncertainty = smaller position. Spread is
    normalized by its trailing 250-bar median so the sizing is regime-aware.
    """
    n = len(spread)
    sizes = np.full(n, lo_pct)
    s_series = pd.Series(spread)
    norm = s_series / s_series.rolling(250, min_periods=60).median()
    for t in range(n):
        if np.isnan(norm.iloc[t]):
            sizes[t] = (lo_pct + hi_pct) / 2
            continue
        # Map (norm > 1 => big spread => small size; norm < 1 => small spread => big size)
        # Using inverse mapping: size = clip(hi_pct + (1 - norm)*0.5, lo_pct, hi_pct)
        s = hi_pct - 0.5 * max(norm.iloc[t] - 1.0, 0.0)
        sizes[t] = float(max(lo_pct, min(hi_pct, s)))
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", default="btc_data/features_BTCUSDT_1d.csv")
    parser.add_argument("--out", default="btc_data/backtest_results.json")
    parser.add_argument("--models", default="xgb,lgb,mlp,dcnn,catboost,tabnet",
                        help="base models for stacking")
    parser.add_argument("--no_regime", action="store_true")
    parser.add_argument("--entry_threshold", type=float, default=0.005)
    args = parser.parse_args()

    print("Loading & engineering features...")
    full, feat_cols = prepare_dataset(Path(args.features_csv))

    X = full[feat_cols].to_numpy(dtype=np.float32)
    closes = full["close"].to_numpy(dtype=np.float64)
    dates = full.index.to_numpy()
    rv = full["rv"].to_numpy(dtype=np.float64)
    target_norm5 = full[f"target_norm_{PRIMARY_HORIZON}"].to_numpy(dtype=np.float32)
    target_raw5 = full[f"target_raw_{PRIMARY_HORIZON}"].to_numpy(dtype=np.float32)

    initial_train = int(len(full) * INITIAL_TRAIN_FRAC)
    print(f"  X shape: {X.shape}  initial_train={initial_train}\n")

    config = StrategyConfig(cls_min_prob=0.0, entry_threshold=args.entry_threshold)
    if args.no_regime:
        tradable = pd.Series(True, index=full.index)
    else:
        tradable = build_regime_filter(full, config)
    tradable_arr = tradable.to_numpy()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"=== Step 1: walk-forward fit each base model ({model_names}) ===")
    raw_preds = {}
    sign_preds = {}
    base_results = {}
    for name in model_names:
        if name not in btc_models.REGISTRY:
            print(f"  [skip] {name}")
            continue
        t0 = time.time()
        try:
            raw, _, refits = walk_forward_single(
                lambda nm=name: btc_models.make(nm),
                X, target_norm5, dates,
                initial_train, REFIT_EVERY, PRIMARY_HORIZON,
            )
        except Exception as e:
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            continue
        sc = apply_sign_correction(raw, target_norm5, PRIMARY_HORIZON)
        ic_raw = safe_corr(sc, target_norm5)
        elapsed = time.time() - t0
        raw_preds[name] = raw
        sign_preds[name] = sc
        base_results[name] = {
            "ic_norm": ic_raw,
            "elapsed_s": round(elapsed, 1),
            "refits": len(refits),
        }
        print(f"  {name:12s}  IC_norm_sc={ic_raw:+.4f}  ({elapsed:.1f}s)")

    if not sign_preds:
        print("No base models succeeded.")
        return

    print(f"\n=== Step 2: walk-forward QuantileLGB for sizing ===")
    t0 = time.time()
    q25, q50, q75 = walk_forward_quantile(
        X, target_norm5, dates, initial_train, REFIT_EVERY, PRIMARY_HORIZON
    )
    spread = q75 - q25
    print(f"  done in {time.time()-t0:.1f}s; spread mean={np.nanmean(spread):.3f}")

    print(f"\n=== Step 3: Ridge meta-learner over base models ===")
    t0 = time.time()
    stacked, coef_hist, stack_names = stack_meta_learner(
        sign_preds, target_norm5, PRIMARY_HORIZON,
        lookback_min=80, refit_every=15, alpha=5.0,
    )
    ic_stack_norm = safe_corr(stacked, target_norm5)
    ic_stack_raw = safe_corr(stacked * np.sqrt(PRIMARY_HORIZON) * rv, target_raw5)
    print(f"  IC_norm={ic_stack_norm:+.4f}  IC_raw={ic_stack_raw:+.4f}  "
          f"refits={len(coef_hist)}  ({time.time()-t0:.1f}s)")
    if coef_hist:
        last_coefs = coef_hist[-1]["coefs"]
        print(f"  last fit coefs: " +
              ", ".join(f"{k}={v:+.3f}" for k, v in last_coefs.items()))

    print(f"\n=== Step 4: strategy variant comparison ===")
    print(f"  {'variant':28s} {'IC_norm':>8s} {'IC_raw':>8s} {'HR':>6s} "
          f"{'Sharpe':>7s} {'Total':>8s} {'PF':>5s} {'MaxDD':>7s} {'N':>4s}")

    sizes_full = np.ones(len(X))
    sizes_q = confidence_from_spread(spread, lookback=120, lo_pct=0.4, hi_pct=1.0)

    variants = []
    # 1) Each individual base model (full size)
    for name, sc in sign_preds.items():
        exp_ret = sc * np.sqrt(PRIMARY_HORIZON) * rv
        variants.append((f"{name}", sc, exp_ret, sizes_full))

    # 2) Mean of base models (full size)
    base_arr = np.array(list(sign_preds.values()))
    mean_sc = np.nanmean(base_arr, axis=0)
    mean_sc = np.where(np.isnan(base_arr).all(axis=0), np.nan, mean_sc)
    variants.append(("mean_ensemble", mean_sc, mean_sc * np.sqrt(PRIMARY_HORIZON) * rv, sizes_full))

    # 3) Top-1 by trailing IC (per bar)
    from btc.advanced_pipeline import trailing_ic_matrix, blend_ensemble
    actuals_dict = {k: target_norm5 for k in sign_preds}
    ic_matrix = trailing_ic_matrix(sign_preds, actuals_dict, PRIMARY_HORIZON,
                                   lookback=120, min_obs=40)
    top1 = blend_ensemble(method="top_k", preds_dict=sign_preds,
                          ic_matrix=ic_matrix, top_k=1)
    variants.append(("top1_by_trailing_IC", top1, top1 * np.sqrt(PRIMARY_HORIZON) * rv, sizes_full))

    # 4) Stacked (full size)
    variants.append(("stacked_ridge", stacked, stacked * np.sqrt(PRIMARY_HORIZON) * rv, sizes_full))

    # 5) Stacked + quantile sizing
    variants.append(("stacked_ridge + qsize", stacked, stacked * np.sqrt(PRIMARY_HORIZON) * rv, sizes_q))

    # 6) Mean ensemble + quantile sizing
    variants.append(("mean_ensemble + qsize", mean_sc, mean_sc * np.sqrt(PRIMARY_HORIZON) * rv, sizes_q))

    # 7) Top1 + quantile sizing
    variants.append(("top1 + qsize", top1, top1 * np.sqrt(PRIMARY_HORIZON) * rv, sizes_q))

    variant_results = {}
    for label, sc, exp_ret, sizes in variants:
        ic_norm = safe_corr(sc, target_norm5)
        ic_raw = safe_corr(exp_ret, target_raw5)
        hr, n_hr = hit_rate(exp_ret, target_raw5)
        trades, equity, position = run_strategy_with_size(
            dates, closes, exp_ret, sizes, tradable_arr, config
        )
        metrics = compute_metrics(equity, dates, closes, trades, config.initial_capital)
        variant_results[label] = {
            "ic_norm": ic_norm, "ic_raw": ic_raw,
            "hit_rate": hr, "hit_rate_n": n_hr,
            "metrics": metrics,
            "predictions_norm": sc,
            "expected_ret": exp_ret,
            "sizes": sizes,
            "trades": trades, "equity": equity, "position": position,
        }
        pf = metrics["profit_factor"]
        pf_str = "inf" if pf is None else f"{pf:.2f}"
        print(f"  {label:28s} {ic_norm:+.4f}  {ic_raw:+.4f}  {hr:.3f}  "
              f"{metrics['sharpe']:+.3f}  {metrics['total_return']:+.2%}  "
              f"{pf_str:>5s}  {metrics['max_drawdown']:+.2%}  "
              f"{metrics['n_trades']:>4d}")

    # Pick best by Sharpe (≥15 trades)
    eligible = [(k, v) for k, v in variant_results.items()
                if v["metrics"]["n_trades"] >= 15]
    if not eligible:
        eligible = list(variant_results.items())
    best_label, best_res = max(
        eligible,
        key=lambda kv: (kv[1]["metrics"]["sharpe"], kv[1]["metrics"]["total_return"])
    )
    print(f"\nBest variant (by Sharpe ≥15 trades): {best_label}")

    # Save dashboard JSON
    cm = np.maximum.accumulate(best_res["equity"])
    dd = (best_res["equity"] - cm) / cm
    bh_eq = closes / closes[0] * config.initial_capital
    equity_curve = [
        {"date": pd.Timestamp(d).strftime("%Y-%m-%d"),
         "strategy": float(eq), "buy_hold": float(bh),
         "drawdown": float(dd_), "in_position": int(p), "tradable": int(t)}
        for d, eq, bh, dd_, p, t in zip(
            dates, best_res["equity"], bh_eq, dd, best_res["position"], tradable_arr)
    ]
    predictions = [
        {"date": pd.Timestamp(d).strftime("%Y-%m-%d"),
         "close": float(c),
         "predicted_5d": (None if np.isnan(p) else float(p)),
         "actual_5d": (None if np.isnan(a) else float(a)),
         "predicted_norm": (None if np.isnan(pn) else float(pn)),
         "actual_norm": (None if np.isnan(yn) else float(yn))}
        for d, c, p, a, pn, yn in zip(
            dates, closes, best_res["expected_ret"], target_raw5,
            best_res["predictions_norm"], target_norm5)
    ]

    n_blocked = int((~tradable).sum())
    out = {
        "config": asdict(config),
        "horizon": PRIMARY_HORIZON,
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "ic_norm": best_res["ic_norm"],
        "ic_raw": best_res["ic_raw"],
        "rmse_raw": float(np.sqrt(np.nanmean(
            (best_res["expected_ret"] - target_raw5) ** 2))),
        "refits": next((v["refits"] for v in base_results.values()), 29),
        "best_model": best_label,
        "metrics": best_res["metrics"],
        "model_comparison": [
            {
                "model": k,
                "ic_norm_sign_corrected": v["ic_norm"],
                "ic_raw_sign_corrected": v["ic_raw"],
                "hit_rate": v["hit_rate"],
                "hit_rate_n": v["hit_rate_n"],
                "sharpe": v["metrics"]["sharpe"],
                "total_return": v["metrics"]["total_return"],
                "profit_factor": v["metrics"]["profit_factor"],
                "max_drawdown": v["metrics"]["max_drawdown"],
                "n_trades": v["metrics"]["n_trades"],
                "win_rate": v["metrics"]["win_rate"],
                "elapsed_s": base_results.get(k, {}).get("elapsed_s", 0),
            }
            for k, v in variant_results.items()
        ],
        "stacking_meta": {
            "base_models": list(sign_preds.keys()),
            "n_refits": len(coef_hist),
            "last_coefs": coef_hist[-1] if coef_hist else None,
            "stack_ic_norm": ic_stack_norm,
            "stack_ic_raw": ic_stack_raw,
        },
        "predictions": predictions,
        "trades": best_res["trades"],
        "equity_curve": equity_curve,
        "regime_filter_enabled": not args.no_regime,
        "regime_blocked_bars": n_blocked,
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
