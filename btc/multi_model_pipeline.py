"""Multi-model BTC pipeline — same I/O as advanced_pipeline but pluggable model.

Walks each model from xgbmodel.btc_models through the v3 pipeline:
  - Identical features (133 cols after coverage filter)
  - Identical target (vol-normalized 5d forward log return)
  - Identical training schedule (walk-forward, refit every 30 bars,
    initial train = 50% of valid history, sample_weight = exp decay,
    early stopping on last 60 bars of train)
  - Identical adaptive sign correction (trailing OOS IC, lookback=120)
  - Identical regime gate + strategy

Output: btc_data/multi_model_results.json (consumed by dashboard.build_btc_xgb)

Usage:
    ./venv/Scripts/python -m btc.multi_model_pipeline
    ./venv/Scripts/python -m btc.multi_model_pipeline --models xgb,lgb,mlp
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

from btc.advanced_pipeline import (
    HORIZONS, PRIMARY_HORIZON, INITIAL_TRAIN_FRAC, REFIT_EVERY,
    WEIGHT_HALF_LIFE, ES_VAL_SIZE,
    StrategyConfig, prepare_dataset, build_regime_filter, run_strategy,
    compute_metrics, exp_decay_weights, safe_corr,
)
from xgbmodel import btc_models

warnings.filterwarnings("ignore", category=UserWarning)


def walk_forward_single(model_factory, X, y_norm, dates, initial_train,
                        refit_every, horizon, hl=WEIGHT_HALF_LIFE):
    """Walk-forward predict using a single model class (no horizon ensemble).

    Returns:
        raw_preds: (n,) sign-uncorrected predictions of normalized target
        importances: list of feature_importances_ per refit (or None)
        refit_dates: refit anchor dates
    """
    n = len(X)
    raw_preds = np.full(n, np.nan)
    importances = []
    refit_dates = []
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
            sw_full = exp_decay_weights(te, hl)
            # NOTE: factories work on full X up to te internally
            sw = np.where(valid, sw_full, 0.0).astype(np.float32)
            # Drop rows with nan target by passing via sample_weight=0 trick;
            # but cleaner: pass only-valid via X[valid], y[valid], reconstruct sw
            X_in = X
            y_in = y_norm.copy()
            # Replace NaN targets with 0 (they get weight 0 anyway)
            y_in_filled = np.where(valid, y_in[:te], 0.0).astype(np.float32)
            # Extend y_in to length te with filled values
            y_pass = y_in.copy()
            y_pass[:te] = y_in_filled

            model = model_factory()
            model.fit(X_in, y_pass, end_idx=te, sample_weight=sw,
                      val_size=ES_VAL_SIZE)
            imp = model.feature_importances_
            if imp is not None:
                importances.append(imp.copy())
            else:
                importances.append(None)
            refit_dates.append(pd.Timestamp(dates[t]).strftime("%Y-%m-%d"))
            last_refit = t

        if model is not None:
            try:
                raw_preds[t] = model.predict(X, t)
            except Exception:
                raw_preds[t] = np.nan

    return raw_preds, importances, refit_dates


def trailing_ic(raw_preds, targets, t, h, lookback=120, min_obs=40):
    end = t - h
    if end <= 0:
        return None
    mask = ~np.isnan(raw_preds[:end]) & ~np.isnan(targets[:end])
    if mask.sum() < min_obs:
        return None
    idx = np.where(mask)[0][-lookback:]
    a, b = raw_preds[idx], targets[idx]
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def apply_sign_correction(raw_preds, targets, h, lookback=120, min_obs=40,
                          sign_min_abs=0.04):
    n = len(raw_preds)
    out = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(raw_preds[t]):
            continue
        ic = trailing_ic(raw_preds, targets, t, h, lookback, min_obs)
        if ic is None or abs(ic) < sign_min_abs:
            out[t] = raw_preds[t]
        else:
            out[t] = (-1.0 if ic < 0 else 1.0) * raw_preds[t]
    return out


def hit_rate(predicted, actual):
    """Directional hit rate among non-zero predictions."""
    v = ~np.isnan(predicted) & ~np.isnan(actual)
    p, a = predicted[v], actual[v]
    if len(p) == 0:
        return float("nan"), 0
    nonflat = np.abs(p) > 0
    if nonflat.sum() == 0:
        return float("nan"), 0
    p, a = p[nonflat], a[nonflat]
    hits = (np.sign(p) == np.sign(a)).mean()
    return float(hits), int(len(p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", default="btc_data/features_BTCUSDT_1d.csv")
    parser.add_argument("--out", default="btc_data/backtest_results.json")
    parser.add_argument("--models", default="xgb,lgb,mlp,dcnn,transformer",
                        help="comma-separated model names from xgbmodel.btc_models REGISTRY")
    parser.add_argument("--no_regime", action="store_true")
    parser.add_argument("--cls_min_prob", type=float, default=0.0,
                        help="classifier gate disabled by default for fair comparison")
    parser.add_argument("--entry_threshold", type=float, default=0.005)
    args = parser.parse_args()

    print("Loading & engineering features (v3 same as advanced_pipeline)...")
    full, feat_cols = prepare_dataset(Path(args.features_csv))

    X = full[feat_cols].to_numpy(dtype=np.float32)
    closes = full["close"].to_numpy(dtype=np.float64)
    dates = full.index.to_numpy()
    rv = full["rv"].to_numpy(dtype=np.float64)
    target_norm5 = full[f"target_norm_{PRIMARY_HORIZON}"].to_numpy(dtype=np.float32)
    target_raw5 = full[f"target_raw_{PRIMARY_HORIZON}"].to_numpy(dtype=np.float32)

    initial_train = int(len(full) * INITIAL_TRAIN_FRAC)
    print(f"  X shape: {X.shape}  initial_train={initial_train}  refit_every={REFIT_EVERY}")

    config = StrategyConfig(cls_min_prob=args.cls_min_prob,
                            entry_threshold=args.entry_threshold)
    if args.no_regime:
        tradable = pd.Series(True, index=full.index)
    else:
        tradable = build_regime_filter(full, config)
    tradable_arr = tradable.to_numpy()
    n_blocked = int((~tradable).sum())
    print(f"  regime filter: {n_blocked} bars blocked ({n_blocked / len(full) * 100:.1f}%)")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"\nModels: {model_names}\n")

    results = {}
    for name in model_names:
        if name not in btc_models.REGISTRY:
            print(f"[skip] {name}: unknown model")
            continue
        print(f"=== {name} ===")
        t0 = time.time()
        try:
            raw, imps, refit_dates = walk_forward_single(
                lambda nm=name: btc_models.make(nm),
                X, target_norm5, dates,
                initial_train, REFIT_EVERY, PRIMARY_HORIZON,
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        elapsed = time.time() - t0

        ic_norm_raw = safe_corr(raw, target_norm5)
        sign_corrected = apply_sign_correction(raw, target_norm5, PRIMARY_HORIZON)
        ic_norm_sc = safe_corr(sign_corrected, target_norm5)

        expected_ret = sign_corrected * np.sqrt(PRIMARY_HORIZON) * rv
        ic_raw = safe_corr(expected_ret, target_raw5)
        hr, n_hr = hit_rate(expected_ret, target_raw5)

        # Use a flat probability of 1.0 to bypass the classifier gate
        cls_prob_dummy = np.ones_like(expected_ret)
        trades, equity, position = run_strategy(
            dates, closes, expected_ret, cls_prob_dummy, tradable_arr, config
        )
        metrics = compute_metrics(equity, dates, closes, trades, config.initial_capital)

        # Final-fit importance for tabular models (last refit only)
        final_imp = None
        if imps and imps[-1] is not None:
            final_imp = sorted(
                [{"feature": f, "importance": float(v)}
                 for f, v in zip(feat_cols, imps[-1])],
                key=lambda r: r["importance"], reverse=True
            )

        results[name] = {
            "elapsed_s": round(elapsed, 1),
            "ic_raw_uncorrected": ic_norm_raw,
            "ic_norm_uncorrected": ic_norm_raw,
            "ic_norm_sign_corrected": ic_norm_sc,
            "ic_raw_sign_corrected": ic_raw,
            "hit_rate": hr,
            "hit_rate_n": n_hr,
            "metrics": metrics,
            "feature_importance": final_imp,
            "raw_preds": raw,
            "sign_corrected": sign_corrected,
            "expected_ret": expected_ret,
            "trades": trades,
            "equity": equity,
            "position": position,
            "refits": len(refit_dates),
        }
        print(f"  IC norm (raw)             : {ic_norm_raw:+.4f}")
        print(f"  IC norm (sign corrected)  : {ic_norm_sc:+.4f}")
        print(f"  IC raw  (sign corrected)  : {ic_raw:+.4f}")
        print(f"  Hit rate                  : {hr:.4f} (n={n_hr})")
        print(f"  Total return / Sharpe     : {metrics['total_return']:+.2%} / {metrics['sharpe']:+.3f}")
        print(f"  Max DD / PF / Trades      : {metrics['max_drawdown']:+.2%} / "
              f"{metrics['profit_factor']:.2f} / {metrics['n_trades']}"
              if metrics['profit_factor'] is not None else
              f"  Max DD / PF / Trades      : {metrics['max_drawdown']:+.2%} / inf / {metrics['n_trades']}")
        print(f"  Wall time                 : {elapsed:.1f}s\n")

    # Comparison summary
    print("\n=== Comparison ===")
    print(f"  {'model':12s} {'IC_norm_sc':>11s} {'IC_raw_sc':>10s} {'HR':>6s} "
          f"{'Sharpe':>7s} {'Total':>8s} {'PF':>5s} {'MaxDD':>7s} {'N':>4s} {'sec':>5s}")
    for name, r in results.items():
        pf = r["metrics"]["profit_factor"]
        pf_str = "inf" if pf is None else f"{pf:.2f}"
        print(f"  {name:12s} {r['ic_norm_sign_corrected']:+.4f}      "
              f"{r['ic_raw_sign_corrected']:+.4f}    {r['hit_rate']:.3f}  "
              f"{r['metrics']['sharpe']:+.3f}  {r['metrics']['total_return']:+.2%}  "
              f"{pf_str:>5s}  {r['metrics']['max_drawdown']:+.2%}  "
              f"{r['metrics']['n_trades']:>4d}  {r['elapsed_s']:>5.1f}")

    # Pick best by Sharpe (≥15 trades)
    eligible = [(k, v) for k, v in results.items()
                if v["metrics"]["n_trades"] >= 15]
    if not eligible:
        eligible = list(results.items())
    if not eligible:
        print("\nNo results.")
        return

    best_name, best_res = max(eligible,
        key=lambda kv: (kv[1]["metrics"]["sharpe"], kv[1]["metrics"]["total_return"])
    )
    print(f"\nBest model (by Sharpe, ≥15 trades): {best_name}")

    # Save: comparison + best model details for dashboard reuse
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
            best_res["sign_corrected"], target_norm5)
    ]

    out = {
        "config": asdict(config),
        "horizon": PRIMARY_HORIZON,
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "model_comparison": [
            {
                "model": k,
                "ic_norm_uncorrected": v["ic_norm_uncorrected"],
                "ic_norm_sign_corrected": v["ic_norm_sign_corrected"],
                "ic_raw_sign_corrected": v["ic_raw_sign_corrected"],
                "hit_rate": v["hit_rate"],
                "hit_rate_n": v["hit_rate_n"],
                "sharpe": v["metrics"]["sharpe"],
                "total_return": v["metrics"]["total_return"],
                "profit_factor": v["metrics"]["profit_factor"],
                "max_drawdown": v["metrics"]["max_drawdown"],
                "n_trades": v["metrics"]["n_trades"],
                "win_rate": v["metrics"]["win_rate"],
                "elapsed_s": v["elapsed_s"],
                "refits": v["refits"],
            }
            for k, v in results.items()
        ],
        "best_model": best_name,
        "metrics": best_res["metrics"],
        "ic_raw": best_res["ic_raw_sign_corrected"],
        "ic_norm": best_res["ic_norm_sign_corrected"],
        "rmse_raw": float(np.sqrt(np.nanmean(
            (best_res["expected_ret"] - target_raw5) ** 2))),
        "refits": best_res["refits"],
        "feature_importance": best_res["feature_importance"] or [],
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
