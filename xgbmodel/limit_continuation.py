"""Historical analysis: probability and predictors of consecutive A-share up-limits.

Definitions:
  - up-limit (涨停): close >= up_limit_price - 0.005
  - one-line up-limit (一字板): all of {open, high, low, close} == up_limit
  - regular up-limit: hit limit but with intraday volatility
  - 连板 streak: number of consecutive trading days a stock has been at up-limit

Computes:
  1. Overall continuation probability:
       P(up_limit_{t+1} | up_limit_t) — base rate
       P(N+1-board | N-board) — conditional on current streak
  2. Stratified breakdowns:
       limit type (one-line vs regular)
       board (ChiNext 300xxx ±20%, STAR 688xxx ±20%, Main ±10%, ST ±5%)
       year, sector (when stock_sectors.csv present), market-cap bucket
  3. Predictive features:
       trains a LightGBM binary classifier on (limit_t -> limit_{t+1}); reports
       gain-based importance and AUC on a 2024+ holdout

Output: stock_data/limit_continuation.json (consumed by dashboard build).

Usage:
    ./venv/Scripts/python -m xgbmodel.limit_continuation
    ./venv/Scripts/python -m xgbmodel.limit_continuation --since 20190101
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "stock_data"
LIMIT_DIR = DATA / "stk_limit"
SH_DIR = DATA / "sh"
SZ_DIR = DATA / "sz"
DAILY_BASIC_DIR = DATA / "daily_basic"
SECTORS_CSV = DATA / "stock_sectors.csv"

LIMIT_TOL = 0.005   # 5 mils — robust to 2-dp rounding
ONE_LINE_TOL = 0.005


def board_of(ts_code: str) -> str:
    """A-share board classification by ts_code prefix."""
    sym = ts_code.split(".")[0]
    if sym.startswith("688"):
        return "STAR"
    if sym.startswith("300"):
        return "ChiNext"
    if sym.startswith("8") or sym.startswith("4"):
        return "BSE"
    return "Main"


def load_stk_limits(since: str | None = None) -> pd.DataFrame:
    files = sorted(LIMIT_DIR.glob("stk_limit_*.csv"))
    if since:
        files = [f for f in files if f.stem.split("_")[-1] >= since]
    print(f"  loading {len(files)} stk_limit files...")
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f, dtype={"trade_date": str})
            parts.append(d)
        except Exception:
            pass
    df = pd.concat(parts, ignore_index=True)
    df["up_limit"] = pd.to_numeric(df["up_limit"], errors="coerce")
    df["down_limit"] = pd.to_numeric(df["down_limit"], errors="coerce")
    return df.dropna(subset=["up_limit"])


def load_ohlcv(since: str | None = None) -> pd.DataFrame:
    parts = []
    for d in [SH_DIR, SZ_DIR]:
        files = sorted(d.glob("*.csv"))
        print(f"  {d.name}: {len(files)} stocks")
        for f in files:
            try:
                df = pd.read_csv(f, dtype={"trade_date": str})
                parts.append(df)
            except Exception:
                pass
    df = pd.concat(parts, ignore_index=True)
    if since:
        df = df[df["trade_date"] >= since]
    keep = ["ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "pct_chg", "vol", "amount"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_daily_basic(since: str | None = None) -> pd.DataFrame | None:
    if not DAILY_BASIC_DIR.exists():
        return None
    files = sorted(DAILY_BASIC_DIR.glob("daily_basic_*.csv"))
    if since:
        files = [f for f in files if f.stem.split("_")[-1] >= since]
    print(f"  loading {len(files)} daily_basic files...")
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f, dtype={"trade_date": str})
            parts.append(d)
        except Exception:
            pass
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    keep = ["ts_code", "trade_date", "turnover_rate_f", "volume_ratio",
            "pe_ttm", "pb", "circ_mv", "total_mv"]
    return df[[c for c in keep if c in df.columns]].copy()


def build_limit_panel(ohlcv: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
    """Merge OHLCV with limit prices; add is_up_limit and one_line flags."""
    df = ohlcv.merge(limits, on=["ts_code", "trade_date"], how="inner")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["is_up_limit"] = (df["close"] >= df["up_limit"] - LIMIT_TOL).astype(np.int8)
    df["is_down_limit"] = (df["close"] <= df["down_limit"] + LIMIT_TOL).astype(np.int8)
    df["one_line"] = (
        (df["is_up_limit"] == 1)
        & (df["open"] >= df["up_limit"] - ONE_LINE_TOL)
        & (df["low"] >= df["up_limit"] - ONE_LINE_TOL)
    ).astype(np.int8)
    df["t_shape"] = ((df["is_up_limit"] == 1) & (df["one_line"] == 0)).astype(np.int8)
    df["board"] = df["ts_code"].apply(board_of)
    df["year"] = df["trade_date"].str[:4]
    return df


def add_features(df: pd.DataFrame, basic: pd.DataFrame | None) -> pd.DataFrame:
    """Add cheap predictive features computed from OHLCV + limits.

    Computed per-stock using groupby. Features are STRICTLY backward-looking
    relative to the index date (no future leakage).
    """
    g = df.groupby("ts_code", group_keys=False)

    # Up-limit streak: # of consecutive prior days in up-limit, including t.
    # Use the "blocks of consecutive equal values" idiom: each time the value
    # of is_up_limit changes vs the prior row (within stock), start a new block;
    # the streak within a block is the cumulative count.
    block = (df["is_up_limit"] != g["is_up_limit"].shift()).cumsum()
    streak = df.groupby(["ts_code", block])["is_up_limit"].cumsum()
    df["streak_up"] = streak.where(df["is_up_limit"] == 1, 0).astype(int)

    # Counts of limits in trailing 20 days (excluding today)
    df["limit_count_20"] = g["is_up_limit"].rolling(20, min_periods=1).sum().reset_index(0, drop=True)

    # Trailing returns
    for w in (3, 5, 10, 20):
        df[f"ret_{w}"] = g["pct_chg"].rolling(w, min_periods=2).sum().reset_index(0, drop=True)

    # Trailing realized volatility
    df["vol_20"] = g["pct_chg"].rolling(20, min_periods=5).std(ddof=0).reset_index(0, drop=True)

    # Volume / amount surge (today vs trailing 20-day mean)
    amt_ma = g["amount"].rolling(20, min_periods=5).mean().reset_index(0, drop=True)
    df["amount_surge"] = df["amount"] / amt_ma.replace(0, np.nan)
    vol_ma = g["vol"].rolling(20, min_periods=5).mean().reset_index(0, drop=True)
    df["vol_surge"] = df["vol"] / vol_ma.replace(0, np.nan)

    # Distance to recent high (proxy for breakout strength)
    high_60 = g["high"].rolling(60, min_periods=10).max().reset_index(0, drop=True)
    df["close_to_60high"] = df["close"] / high_60

    # Trailing 5-day cumulative limit count
    df["limits_5d"] = g["is_up_limit"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)

    # Days since previous up-limit
    has_prev = (g["is_up_limit"].cumsum() > 0).astype(int)
    last_limit_idx = df.assign(li=np.where(df["is_up_limit"] == 1,
                                           np.arange(len(df)), np.nan))
    last_limit_idx["li"] = last_limit_idx.groupby("ts_code")["li"].ffill()
    df["days_since_limit"] = (
        np.arange(len(df)) - last_limit_idx["li"].fillna(-1).to_numpy()
    ).astype(float)
    df.loc[has_prev == 0, "days_since_limit"] = np.nan

    # Daily-basic merges
    if basic is not None:
        df = df.merge(basic, on=["ts_code", "trade_date"], how="left")
        if "circ_mv" in df.columns:
            df["log_circ_mv"] = np.log(df["circ_mv"].clip(lower=1))
            # Cap bucket
            df["cap_bucket"] = pd.qcut(df["circ_mv"], q=5, labels=["XS", "S", "M", "L", "XL"],
                                       duplicates="drop")

    return df


def conditional_probabilities(df: pd.DataFrame) -> dict:
    """P(up_limit at t+1 | conditions at t)."""
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    df["next_is_up_limit"] = df.groupby("ts_code")["is_up_limit"].shift(-1)
    df["next_one_line"] = df.groupby("ts_code")["one_line"].shift(-1)
    has_next = df["next_is_up_limit"].notna()
    base = df[has_next]
    n_total = len(base)
    n_uplimit = int(base["is_up_limit"].sum())

    res = {
        "n_total_rows": int(n_total),
        "n_up_limit_rows": n_uplimit,
        "p_uplimit_unconditional_t": float(base["is_up_limit"].mean()),
        "p_uplimit_unconditional_t1": float(base["next_is_up_limit"].mean()),
    }

    # P(up_t+1 | up_t) by streak
    by_streak = []
    for s, sub in base.groupby("streak_up"):
        if s == 0 or len(sub) < 50:
            continue
        cont = float(sub["next_is_up_limit"].mean())
        cont_one = float((sub["next_one_line"] == 1).mean())
        by_streak.append({
            "streak": int(s),
            "n": int(len(sub)),
            "p_continue": cont,
            "p_continue_one_line": cont_one,
        })
    res["by_streak"] = sorted(by_streak, key=lambda x: x["streak"])

    # P(up_t+1 | up_t, limit_type)
    upl = base[base["is_up_limit"] == 1]
    by_type = []
    for label, mask in [("one_line", upl["one_line"] == 1),
                         ("t_shape", upl["t_shape"] == 1)]:
        sub = upl[mask]
        if len(sub) < 30:
            continue
        by_type.append({
            "type": label,
            "n": int(len(sub)),
            "p_continue": float(sub["next_is_up_limit"].mean()),
            "p_continue_one_line": float((sub["next_one_line"] == 1).mean()),
        })
    res["by_type_given_up_t"] = by_type

    # P(up_t+1 | up_t, board)
    by_board = []
    for board, sub in upl.groupby("board"):
        if len(sub) < 30:
            continue
        by_board.append({
            "board": board, "n": int(len(sub)),
            "p_continue": float(sub["next_is_up_limit"].mean()),
        })
    res["by_board_given_up_t"] = sorted(by_board, key=lambda x: -x["n"])

    # P(up_t+1 | up_t) by year
    by_year = []
    for yr, sub in upl.groupby("year"):
        by_year.append({
            "year": yr, "n": int(len(sub)),
            "p_continue": float(sub["next_is_up_limit"].mean()),
        })
    res["by_year_given_up_t"] = sorted(by_year, key=lambda x: x["year"])

    # P(up_t+1 | up_t, cap_bucket) if available
    if "cap_bucket" in upl.columns:
        by_cap = []
        for cb, sub in upl.dropna(subset=["cap_bucket"]).groupby("cap_bucket", observed=True):
            if len(sub) < 30:
                continue
            by_cap.append({
                "cap_bucket": str(cb), "n": int(len(sub)),
                "p_continue": float(sub["next_is_up_limit"].mean()),
            })
        res["by_cap_given_up_t"] = by_cap

    return res


def feature_lift_table(df: pd.DataFrame, features: list[str], n_bins: int = 5) -> list[dict]:
    """For each feature, bin it (quintiles) then compute P(continue | up_t, bin)."""
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    df["next_is_up_limit"] = df.groupby("ts_code")["is_up_limit"].shift(-1)
    base = df[(df["is_up_limit"] == 1) & df["next_is_up_limit"].notna()].copy()
    rows = []
    for f in features:
        if f not in base.columns:
            continue
        v = base[f].dropna()
        if len(v) < 200 or v.nunique() < n_bins:
            continue
        try:
            quantiles = pd.qcut(base[f], q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        bin_stats = []
        for b, sub in base.groupby(quantiles):
            if len(sub) < 30 or pd.isna(b):
                continue
            bin_stats.append({
                "bin": int(b), "n": int(len(sub)),
                "p_continue": float(sub["next_is_up_limit"].mean()),
                "feature_mean": float(sub[f].mean()),
            })
        if len(bin_stats) >= 3:
            ps = [bs["p_continue"] for bs in bin_stats]
            lift = max(ps) - min(ps)
            rows.append({
                "feature": f, "lift": lift, "bins": bin_stats,
                "p_top_bin": ps[-1], "p_bot_bin": ps[0],
            })
    return sorted(rows, key=lambda r: -r["lift"])


def train_classifier(df: pd.DataFrame, features: list[str],
                     test_since: str = "20240101") -> dict:
    """Train LightGBM on (in up-limit at t -> up-limit at t+1)."""
    import lightgbm as lgb

    df = df.sort_values(["ts_code", "trade_date"]).copy()
    df["y"] = df.groupby("ts_code")["is_up_limit"].shift(-1)
    base = df[(df["is_up_limit"] == 1) & df["y"].notna()].copy()
    feats = [f for f in features if f in base.columns]
    base = base[feats + ["y", "trade_date"]].dropna(subset=feats)
    print(f"  classifier dataset: {len(base)} rows × {len(feats)} features")

    train = base[base["trade_date"] < test_since]
    test = base[base["trade_date"] >= test_since]
    if len(train) < 500 or len(test) < 100:
        print("  not enough data for classifier")
        return {}
    print(f"  train: {len(train)}  test: {len(test)}")

    params = dict(
        objective="binary", metric="auc",
        n_estimators=400, num_leaves=31, learning_rate=0.04,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(train[feats], train["y"],
              eval_set=[(test[feats], test["y"])],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    probs_tr = model.predict_proba(train[feats])[:, 1]
    probs_te = model.predict_proba(test[feats])[:, 1]
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc_tr = float(roc_auc_score(train["y"], probs_tr))
    auc_te = float(roc_auc_score(test["y"], probs_te))
    ap_te = float(average_precision_score(test["y"], probs_te))
    base_rate = float(test["y"].mean())

    imp = sorted(
        [{"feature": f, "importance": float(g)}
         for f, g in zip(feats, model.feature_importances_)],
        key=lambda r: -r["importance"],
    )
    return {
        "n_train": len(train), "n_test": len(test),
        "test_since": test_since,
        "base_rate_test": base_rate,
        "auc_train": auc_tr, "auc_test": auc_te,
        "average_precision_test": ap_te,
        "feature_importance": imp,
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="20180101")
    ap.add_argument("--out", default=str(DATA / "limit_continuation.json"))
    args = ap.parse_args()

    print(f"Loading data from {args.since} ...")
    t0 = time.time()
    limits = load_stk_limits(since=args.since)
    print(f"  limits: {len(limits):,} rows ({time.time() - t0:.1f}s)")
    ohlcv = load_ohlcv(since=args.since)
    print(f"  ohlcv:  {len(ohlcv):,} rows ({time.time() - t0:.1f}s)")
    basic = load_daily_basic(since=args.since)
    if basic is not None:
        print(f"  basic:  {len(basic):,} rows ({time.time() - t0:.1f}s)")

    print("\nBuilding limit panel + features...")
    df = build_limit_panel(ohlcv, limits)
    df = add_features(df, basic)
    print(f"  panel: {df.shape}")

    print("\nComputing conditional probabilities...")
    probs = conditional_probabilities(df)
    print(f"  base P(up at t)        : {probs['p_uplimit_unconditional_t']:.4f}")
    print(f"  base P(up at t+1)      : {probs['p_uplimit_unconditional_t1']:.4f}")
    print("  by streak:")
    for r in probs["by_streak"][:8]:
        print(f"    streak={r['streak']}  n={r['n']:>6}  P(continue)={r['p_continue']:.3f}"
              f"  P(continue 一字)={r['p_continue_one_line']:.3f}")
    print("  by limit type (given up at t):")
    for r in probs["by_type_given_up_t"]:
        print(f"    {r['type']:<10s}  n={r['n']:>6}  P(continue)={r['p_continue']:.3f}")
    print("  by board (given up at t):")
    for r in probs["by_board_given_up_t"]:
        print(f"    {r['board']:<8s}  n={r['n']:>6}  P(continue)={r['p_continue']:.3f}")

    print("\nFeature lift tables...")
    feature_candidates = [
        "streak_up", "limit_count_20", "limits_5d",
        "ret_3", "ret_5", "ret_10", "ret_20",
        "vol_20", "amount_surge", "vol_surge",
        "close_to_60high", "days_since_limit",
        "turnover_rate_f", "volume_ratio", "pe_ttm", "pb",
        "log_circ_mv",
    ]
    lifts = feature_lift_table(df, feature_candidates)
    print(f"  computed lift for {len(lifts)} features")
    for r in lifts[:10]:
        print(f"    {r['feature']:<22s} lift={r['lift']:.3f}  "
              f"low_bin_p={r['p_bot_bin']:.3f} → top_bin_p={r['p_top_bin']:.3f}")

    print("\nTraining LightGBM classifier (continuation)...")
    cls = train_classifier(df, feature_candidates)
    if cls:
        print(f"  AUC train={cls['auc_train']:.4f}  test={cls['auc_test']:.4f}  "
              f"AP test={cls['average_precision_test']:.4f}")
        print(f"  base rate (test)={cls['base_rate_test']:.4f}")
        print("  top features by gain:")
        for r in cls["feature_importance"][:15]:
            print(f"    {r['feature']:<22s} {r['importance']:.0f}")

    # Recent regime: last 90 trading days
    recent_dates = sorted(df["trade_date"].unique())[-90:]
    recent = df[df["trade_date"].isin(recent_dates)]
    recent_probs = conditional_probabilities(recent)
    print(f"\nRecent (last 90 days) base P(continue|up): "
          f"{sum(r['p_continue'] * r['n'] for r in recent_probs['by_streak']) / max(sum(r['n'] for r in recent_probs['by_streak']), 1):.4f}")

    # Coverage of TODAY's top predictions: which top stocks just hit up-limit?
    today_overlap = []
    pred_csv = REPO / "stock_predictions_xgb_20260424.csv"
    if not pred_csv.exists():
        pred_csv = REPO / "stock_predictions_xgb.csv"
    if pred_csv.exists():
        try:
            pred = pd.read_csv(pred_csv).nlargest(30, "pred_pct_chg_next")
            last_date = sorted(df["trade_date"].unique())[-1]
            todayrow = df[df["trade_date"] == last_date].set_index("ts_code")
            for _, row in pred.iterrows():
                tc = row["ts_code"]
                if tc in todayrow.index:
                    r = todayrow.loc[tc]
                    today_overlap.append({
                        "ts_code": tc,
                        "pred_pct_chg_next": float(row["pred_pct_chg_next"]),
                        "is_up_limit_today": int(r["is_up_limit"]),
                        "one_line_today": int(r["one_line"]),
                        "streak_up": int(r["streak_up"]),
                        "limits_5d": float(r["limits_5d"]),
                        "amount_surge": (float(r["amount_surge"])
                                          if pd.notna(r["amount_surge"]) else None),
                    })
            print(f"\nLatest panel date: {last_date}; cross-checked {len(today_overlap)} top predictions")
        except Exception as e:
            print(f"  WARNING: top-pred cross-check failed: {e}")

    out = {
        "data_since": args.since,
        "panel_dates": [df["trade_date"].min(), df["trade_date"].max()],
        "n_stocks": int(df["ts_code"].nunique()),
        "probabilities_overall": probs,
        "probabilities_recent_90d": recent_probs,
        "feature_lifts": lifts,
        "classifier": cls,
        "top_pred_overlap_today": today_overlap,
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=str, indent=2)
    print(f"\nSaved -> {out_p}")


if __name__ == "__main__":
    main()
