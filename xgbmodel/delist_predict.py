"""A-share delisting risk prediction.

Pipeline
========

1. Snapshot dataset
   For each (ts_code, snapshot_date) pair where the stock has ≥ 365 days
   of history at the snapshot AND we can observe at least 365 days of
   forward outcome:
     y = 1  if stock delists in (snapshot_date, snapshot_date + 365]
     y = 0  otherwise

   Snapshots are quarterly (Jan/Apr/Jul/Oct) from 2019 through the most
   recent eligible quarter. Survival snapshots also cover delisted stocks
   *before* the 365-day window (those rows are y=0, which is fine — it
   teaches the model what 'safe' regimes look like).

2. Features (~30 cols, computed STRICTLY from data ≤ snapshot_date)
   Price / volume:    log_close, ret_30/90/180/365, drawdown_1y/3y,
                      dist_from_1y_low, realized_vol_60, avg_range_60,
                      log_avg_amount_60/365, amount_trend,
                      zero_vol_days_90, low_vol_days_90
   Daily basic:       log_circ_mv, pb, pe_ttm_clipped, turnover_60d
   ST history:        is_st, is_starred_st, days_in_st_total,
                      n_st_episodes, days_since_last_st_end
   Metadata:          days_listed, board (Main/ChiNext/STAR/BSE)

3. Training: LightGBM binary, scale_pos_weight = neg/pos.
   Walk-forward split by snapshot year:
     train  ≤ 2022
     val    = 2023
     test   = 2024–latest

4. Score every currently-listed stock at the most-recent quarter snapshot.

Output -> stock_data/delist_predictions.json (consumed by dashboard).

Usage:
    ./venv/Scripts/python -m xgbmodel.delist_predict
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "stock_data"
SH_DIR = DATA / "sh"
SZ_DIR = DATA / "sz"
DAILY_BASIC_DIR = DATA / "daily_basic"
SECTORS_CSV = DATA / "stock_sectors.csv"
STATUS_CSV = DATA / "stock_basic_status.csv"
ST_CSV = DATA / "st_history.csv"

SNAPSHOT_QUARTERS = ["0101", "0401", "0701", "1001"]
EMPTY_ST = pd.DataFrame(columns=["ts_code", "st_kind", "start_date", "end_date"])
SNAPSHOT_YEARS = list(range(2019, 2027))  # extend as data accumulates
LOOK_FORWARD_DAYS = 365
MIN_HISTORY_DAYS = 200       # need at least 200 trading days of history
TODAY_INT = int(date.today().strftime("%Y%m%d"))


def board_of(ts_code: str) -> str:
    sym = ts_code.split(".")[0]
    if sym.startswith("688"):
        return "STAR"
    if sym.startswith("300"):
        return "ChiNext"
    if sym.startswith("8") or sym.startswith("4"):
        return "BSE"
    return "Main"


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_status() -> pd.DataFrame:
    df = pd.read_csv(STATUS_CSV, dtype={"delist_date": str})
    df["delist_date"] = df["delist_date"].fillna("99999999")
    return df


def load_st() -> pd.DataFrame:
    df = pd.read_csv(ST_CSV, dtype={"start_date": str, "end_date": str})
    df["start_date"] = df["start_date"].fillna("99999999")
    df["end_date"] = df["end_date"].fillna("99999999")
    return df


def find_ohlcv_path(ts_code: str) -> Path | None:
    sym = ts_code.split(".")[0]
    if ts_code.endswith(".SH"):
        p = SH_DIR / f"{sym}.csv"
    elif ts_code.endswith(".SZ"):
        p = SZ_DIR / f"{sym}.csv"
    else:
        p = SH_DIR / f"{sym}.csv"
    return p if p.exists() else None


def load_ohlcv(ts_code: str) -> pd.DataFrame | None:
    p = find_ohlcv_path(ts_code)
    if p is None:
        # Try delisted_ohlcv if present
        for d in ["delisted_sh", "delisted_sz", "delisted"]:
            cand = DATA / d / f"{ts_code.split('.')[0]}.csv"
            if cand.exists():
                p = cand
                break
    if p is None:
        return None
    try:
        df = pd.read_csv(p, dtype={"trade_date": str})
        for c in ["open", "high", "low", "close", "vol", "amount", "pct_chg"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("trade_date").reset_index(drop=True)
        return df
    except Exception:
        return None


def load_daily_basic_panel(ts_codes: list[str], min_date: str) -> pd.DataFrame:
    """Load daily_basic for given codes after min_date. Returns long DF."""
    parts = []
    code_set = set(ts_codes)
    files = sorted(DAILY_BASIC_DIR.glob("daily_basic_*.csv"))
    files = [f for f in files if f.stem.split("_")[-1] >= min_date]
    print(f"  loading {len(files)} daily_basic files (since {min_date})...")
    for f in files:
        try:
            d = pd.read_csv(f, dtype={"trade_date": str})
            d = d[d["ts_code"].isin(code_set)]
            keep = ["ts_code", "trade_date", "turnover_rate_f", "pb",
                    "pe_ttm", "circ_mv"]
            parts.append(d[[c for c in keep if c in d.columns]])
        except Exception:
            pass
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    for c in ["turnover_rate_f", "pb", "pe_ttm", "circ_mv"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─── Feature engineering at a single snapshot ─────────────────────────────────

def compute_features(ohlcv: pd.DataFrame, snap: str,
                     daily_basic: pd.DataFrame | None,
                     st_episodes: pd.DataFrame,
                     list_date: str | None,
                     board: str) -> dict | None:
    """Return feature dict for this stock at this snapshot, or None if not computable."""
    past = ohlcv[ohlcv["trade_date"] <= snap]
    if len(past) < MIN_HISTORY_DAYS:
        return None

    close = past["close"].to_numpy(dtype=np.float64)
    high = past["high"].to_numpy(dtype=np.float64)
    low = past["low"].to_numpy(dtype=np.float64)
    vol = past["vol"].to_numpy(dtype=np.float64)
    amount = past["amount"].to_numpy(dtype=np.float64)

    last_close = float(close[-1]) if len(close) else np.nan
    if not np.isfinite(last_close) or last_close <= 0:
        return None

    def cum_ret(n):
        if len(close) <= n:
            return np.nan
        return float(close[-1] / close[-n - 1] - 1)

    def realized_vol(n):
        if len(close) <= n:
            return np.nan
        rets = np.diff(np.log(close[-n - 1:]))
        return float(np.std(rets, ddof=0) * np.sqrt(252))

    def avg_range(n):
        n = min(n, len(high))
        h = high[-n:]; l = low[-n:]; c = close[-n:]
        return float(np.mean((h - l) / np.where(c > 0, c, np.nan)))

    def log_avg_amount(n):
        n = min(n, len(amount))
        m = np.nanmean(amount[-n:])
        return float(np.log(max(m, 1)))

    def zero_vol_count(n):
        n = min(n, len(vol))
        return int(np.sum(vol[-n:] == 0))

    def low_vol_count(n):
        n = min(n, len(vol))
        v = vol[-n:]
        med = np.nanmedian(v)
        if not np.isfinite(med) or med == 0:
            return 0
        return int(np.sum(v < 0.1 * med))

    high_1y = float(np.nanmax(close[-min(252, len(close)):]))
    high_3y = float(np.nanmax(close[-min(750, len(close)):]))
    low_1y = float(np.nanmin(close[-min(252, len(close)):]))

    feat = {
        # log_close DROPPED: banks (low absolute price) get false-positive flag
        "ret_30":               cum_ret(30),
        "ret_90":               cum_ret(90),
        "ret_180":              cum_ret(180),
        "ret_365":              cum_ret(min(365, len(close) - 1)),
        "drawdown_1y":          float(last_close / high_1y - 1) if high_1y > 0 else np.nan,
        "drawdown_3y":          float(last_close / high_3y - 1) if high_3y > 0 else np.nan,
        "dist_from_1y_low":     float(last_close / low_1y - 1) if low_1y > 0 else np.nan,
        "realized_vol_60":      realized_vol(60),
        "avg_range_60":         avg_range(60),
        "log_avg_amount_60":    log_avg_amount(60),
        "log_avg_amount_365":   log_avg_amount(min(365, len(amount))),
        "amount_trend":         (log_avg_amount(60) - log_avg_amount(min(365, len(amount)))),
        "zero_vol_days_90":     zero_vol_count(90),
        "low_vol_days_90":      low_vol_count(90),
    }

    # Daily basic (latest at-or-before snapshot)
    if daily_basic is not None and len(daily_basic):
        db = daily_basic[daily_basic["trade_date"] <= snap]
        if len(db):
            row = db.iloc[-1]
            cm = float(row.get("circ_mv", np.nan))
            feat["log_circ_mv"] = float(np.log(max(cm, 1))) if cm > 0 else np.nan
            pb = float(row.get("pb", np.nan))
            feat["pb"] = pb if 0 < pb < 100 else np.nan
            pe = float(row.get("pe_ttm", np.nan))
            feat["pe_ttm"] = max(min(pe, 200), -200) if np.isfinite(pe) else np.nan
            # 60d turnover average from db
            db60 = db.tail(60)
            tr = pd.to_numeric(db60.get("turnover_rate_f"), errors="coerce")
            feat["turnover_60d"] = float(tr.mean()) if len(tr) else np.nan
        else:
            feat["log_circ_mv"] = feat["pb"] = feat["pe_ttm"] = feat["turnover_60d"] = np.nan
    else:
        feat["log_circ_mv"] = feat["pb"] = feat["pe_ttm"] = feat["turnover_60d"] = np.nan

    # ST history features
    is_st = 0; is_starred = 0
    days_in_st = 0; n_episodes = 0; last_st_end = None
    for _, row in st_episodes.iterrows():
        s, e = row["start_date"], row["end_date"]
        if s > snap:
            continue
        n_episodes += 1
        eff_end = e if e <= snap else snap
        # Convert to dates and diff
        try:
            ds = datetime.strptime(s, "%Y%m%d")
            de = datetime.strptime(eff_end, "%Y%m%d")
            days_in_st += max((de - ds).days, 0)
        except Exception:
            pass
        if e == "99999999" or e > snap:
            is_st = 1
            if str(row.get("st_kind", "")).startswith("*"):
                is_starred = 1
        if e <= snap and e != "99999999":
            if last_st_end is None or e > last_st_end:
                last_st_end = e
    days_since_last_st_end = np.nan
    if last_st_end is not None:
        try:
            de = datetime.strptime(last_st_end, "%Y%m%d")
            ds = datetime.strptime(snap, "%Y%m%d")
            days_since_last_st_end = (ds - de).days
        except Exception:
            pass

    feat["is_st"] = is_st
    feat["is_starred_st"] = is_starred
    feat["days_in_st_total"] = days_in_st
    feat["n_st_episodes"] = n_episodes
    feat["days_since_last_st_end"] = days_since_last_st_end

    # days_listed DROPPED: per-stock OHLCV window length is a data artifact,
    # not real listing age (most stocks show ~1812 days = our 5-year data span).

    # Board one-hots
    feat["is_main"] = int(board == "Main")
    feat["is_chinext"] = int(board == "ChiNext")
    feat["is_star"] = int(board == "STAR")
    feat["is_bse"] = int(board == "BSE")

    return feat


# ─── Snapshot grid construction ───────────────────────────────────────────────

def make_snapshots() -> list[str]:
    out = []
    for y in SNAPSHOT_YEARS:
        for q in SNAPSHOT_QUARTERS:
            d = f"{y}{q}"
            if int(d) <= TODAY_INT:
                out.append(d)
    return out


def assign_label(ts_code: str, snap: str, status: pd.DataFrame) -> int | None:
    """y=1 if stock delists in (snap, snap+365], y=0 if alive throughout that
    window AND we can observe at least snap+365 in our data, None if the
    snapshot is too recent to be labeled (used for inference, not training)."""
    snap_int = int(snap)
    fwd = (datetime.strptime(snap, "%Y%m%d") + timedelta(days=LOOK_FORWARD_DAYS))
    fwd_int = int(fwd.strftime("%Y%m%d"))
    if fwd_int > TODAY_INT:
        return None  # not yet labelable
    rec = status[status["ts_code"] == ts_code]
    if rec.empty:
        return None
    rec = rec.iloc[0]
    if rec["list_status"] == "L":
        return 0
    dd = rec["delist_date"]
    if dd == "99999999":
        return 0
    dd_int = int(dd)
    if snap_int < dd_int <= fwd_int:
        return 1
    if dd_int <= snap_int:
        return None  # already delisted at snapshot, drop
    return 0


# ─── Build dataset ────────────────────────────────────────────────────────────

def build_dataset(status: pd.DataFrame, st: pd.DataFrame,
                  inference_snap: str | None = None,
                  max_stocks: int = 0) -> tuple[pd.DataFrame, list[str]]:
    snapshots = make_snapshots()
    print(f"snapshots: {len(snapshots)} ({snapshots[0]} → {snapshots[-1]})")
    if inference_snap and inference_snap not in snapshots:
        snapshots.append(inference_snap)

    # Pre-load daily_basic for ALL codes once (heavy step)
    # Always include all delisted (positives are scarce); cap only the listed.
    all_listed = status[status["list_status"] == "L"]["ts_code"].tolist()
    all_delisted = status[status["list_status"] == "D"]["ts_code"].tolist()
    if max_stocks > 0:
        all_listed = all_listed[:max(max_stocks - len(all_delisted), 100)]
    all_codes = all_listed + all_delisted

    earliest_snap = min(snapshots)
    db_min = (datetime.strptime(earliest_snap, "%Y%m%d") - timedelta(days=400)).strftime("%Y%m%d")
    daily_basic_all = load_daily_basic_panel(all_codes, db_min)
    print(f"  daily_basic rows: {len(daily_basic_all):,}")
    # Pre-bucket by ts_code for O(1) lookup instead of linear scan per stock
    print("  bucketing daily_basic by ts_code...")
    if len(daily_basic_all):
        daily_basic_all = daily_basic_all.sort_values(["ts_code", "trade_date"])
        db_by_code = {tc: g.reset_index(drop=True)
                      for tc, g in daily_basic_all.groupby("ts_code", sort=False)}
    else:
        db_by_code = {}
    del daily_basic_all
    print(f"  buckets: {len(db_by_code)}")
    # Bucket ST history and status too
    st_by_code = {tc: g for tc, g in st.groupby("ts_code", sort=False)}
    status_by_code = {row["ts_code"]: row for _, row in status.iterrows()}

    rows = []
    feature_keys = None

    t0 = time.time()
    for i, ts_code in enumerate(all_codes):
        if i % 250 == 0 and i:
            print(f"  [{i}/{len(all_codes)}] elapsed {time.time()-t0:.0f}s, "
                  f"rows so far {len(rows)}")
        ohlcv = load_ohlcv(ts_code)
        if ohlcv is None or ohlcv.empty:
            continue

        rec = status_by_code.get(ts_code)
        if rec is None:
            continue
        list_date = ohlcv["trade_date"].iloc[0]  # safe proxy if list_date col missing
        delist_date = rec["delist_date"]
        board = board_of(ts_code)

        st_eps = st_by_code.get(ts_code, EMPTY_ST)
        db_stock = db_by_code.get(ts_code)

        # Pre-compute delist_int (once per stock) for the label loop below
        dd_int = int(delist_date) if delist_date != "99999999" else None
        list_status = rec["list_status"]

        for snap in snapshots:
            snap_int = int(snap)
            # Skip snapshots at or after the stock delisted
            if dd_int is not None and snap_int >= dd_int:
                continue
            # Inline label (faster than calling assign_label which scans status)
            fwd = (datetime.strptime(snap, "%Y%m%d") + timedelta(days=LOOK_FORWARD_DAYS))
            fwd_int = int(fwd.strftime("%Y%m%d"))
            is_inference = (inference_snap is not None and snap == inference_snap)
            if fwd_int > TODAY_INT:
                if not is_inference:
                    continue
                label = None
            elif list_status == "L":
                label = 0
            elif dd_int is None:
                label = 0
            elif snap_int < dd_int <= fwd_int:
                label = 1
            else:
                label = 0

            feat = compute_features(ohlcv, snap, db_stock, st_eps, list_date, board)
            if feat is None:
                continue
            if feature_keys is None:
                feature_keys = list(feat.keys())
            row = {"ts_code": ts_code, "snapshot": snap,
                   "name": rec["name"], "y": label, **feat}
            rows.append(row)

    print(f"  built {len(rows)} snapshots in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows), feature_keys or []


# ─── Train ────────────────────────────────────────────────────────────────────

def train_classifier(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    df = df.copy()
    df["snap_year"] = df["snapshot"].str[:4].astype(int)

    train = df[(df["snap_year"] <= 2022) & df["y"].notna()].copy()
    val = df[(df["snap_year"] == 2023) & df["y"].notna()].copy()
    test = df[(df["snap_year"] >= 2024) & df["y"].notna()].copy()
    print(f"\ntrain: {len(train)}  ({int(train.y.sum())} pos)")
    print(f"val  : {len(val)}    ({int(val.y.sum())} pos)")
    print(f"test : {len(test)}   ({int(test.y.sum())} pos)")

    if train.empty or val.empty:
        raise SystemExit("Not enough labeled data to train")

    pos = max(int(train.y.sum()), 1)
    neg = max(int((1 - train.y).sum()), 1)
    spw = neg / pos
    print(f"  scale_pos_weight = {spw:.1f}")

    params = dict(
        objective="binary", metric="auc",
        n_estimators=600, num_leaves=31, learning_rate=0.04,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        min_child_samples=30, scale_pos_weight=spw,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(train[feat_cols], train["y"],
              eval_set=[(val[feat_cols], val["y"])],
              callbacks=[lgb.early_stopping(40, verbose=False)])

    aucs = {}
    for name, sub in [("train", train), ("val", val), ("test", test)]:
        if sub.empty:
            continue
        prob = model.predict_proba(sub[feat_cols])[:, 1]
        aucs[name] = {
            "auc": float(roc_auc_score(sub["y"], prob)),
            "ap": float(average_precision_score(sub["y"], prob)),
            "n": int(len(sub)),
            "n_pos": int(sub["y"].sum()),
            "base_rate": float(sub["y"].mean()),
        }
    print("\nMetrics:")
    for k, v in aucs.items():
        print(f"  {k:6s}  AUC={v['auc']:.4f}  AP={v['ap']:.4f}  "
              f"n={v['n']}  pos={v['n_pos']}  base={v['base_rate']:.4f}")

    importance = sorted(
        [{"feature": f, "importance": float(g)}
         for f, g in zip(feat_cols, model.feature_importances_)],
        key=lambda r: -r["importance"],
    )
    print("\nTop features by gain:")
    for r in importance[:15]:
        print(f"  {r['feature']:<26s} {r['importance']:.0f}")

    return {
        "model": model,
        "metrics": aucs,
        "feature_importance": importance,
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
    }


# ─── Score currently-listed stocks ────────────────────────────────────────────

def score_inference(model_obj, feat_cols, status, st, snap):
    listed = status[status["list_status"] == "L"]["ts_code"].tolist()
    print(f"\nScoring {len(listed)} currently-listed stocks at snapshot {snap}...")
    rows = []
    t0 = time.time()
    db_min = (datetime.strptime(snap, "%Y%m%d") - timedelta(days=400)).strftime("%Y%m%d")
    daily_basic_all = load_daily_basic_panel(listed, db_min)
    print("  bucketing daily_basic by ts_code...")
    db_by_code = ({tc: g.reset_index(drop=True)
                   for tc, g in daily_basic_all.sort_values(["ts_code", "trade_date"])
                                                .groupby("ts_code", sort=False)}
                  if len(daily_basic_all) else {})
    del daily_basic_all
    st_by_code = {tc: g for tc, g in st.groupby("ts_code", sort=False)}
    status_by_code = {row["ts_code"]: row for _, row in status.iterrows()}

    for i, ts_code in enumerate(listed):
        if i % 500 == 0 and i:
            print(f"  [{i}/{len(listed)}] elapsed {time.time()-t0:.0f}s")
        ohlcv = load_ohlcv(ts_code)
        if ohlcv is None or ohlcv.empty:
            continue
        rec = status_by_code.get(ts_code)
        if rec is None:
            continue
        list_date = ohlcv["trade_date"].iloc[0]
        board = board_of(ts_code)
        st_eps = st_by_code.get(ts_code, EMPTY_ST)
        db_stock = db_by_code.get(ts_code)
        feat = compute_features(ohlcv, snap, db_stock, st_eps, list_date, board)
        if feat is None:
            continue
        rows.append({"ts_code": ts_code, "name": rec["name"], **feat})

    df = pd.DataFrame(rows)
    print(f"  features ready for {len(df)} stocks")
    X = df[feat_cols]
    df["p_delist"] = model_obj.predict_proba(X)[:, 1]
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_stocks", type=int, default=0)
    ap.add_argument("--inference_snap", default=None,
                    help="YYYYMMDD; defaults to last completed quarter")
    ap.add_argument("--out", default=str(DATA / "delist_predictions.json"))
    args = ap.parse_args()

    print("Loading metadata...")
    status = load_status()
    st = load_st()
    print(f"  {len(status)} stocks ({(status.list_status=='L').sum()} L, "
          f"{(status.list_status=='D').sum()} D), {len(st)} ST episodes")

    # Default inference snapshot = last quarter that we'd actually use today
    if args.inference_snap is None:
        snaps = make_snapshots()
        # most recent snapshot whose +365 day window starts no later than today
        args.inference_snap = snaps[-1]
    print(f"  inference snapshot: {args.inference_snap}")

    print("\nBuilding training dataset...")
    df, feat_cols = build_dataset(status, st,
                                  inference_snap=args.inference_snap,
                                  max_stocks=args.max_stocks)
    print(f"  total rows: {len(df)}, features: {len(feat_cols)}")

    train_df = df[df["y"].notna()].copy()
    out = train_classifier(train_df, feat_cols)
    model = out["model"]

    pred_df = score_inference(model, feat_cols, status, st, args.inference_snap)
    pred_df = pred_df.sort_values("p_delist", ascending=False).reset_index(drop=True)

    print(f"\nP(delist) distribution among {len(pred_df)} listed stocks:")
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"  q{int(q*100)} = {pred_df['p_delist'].quantile(q):.4f}")
    print(f"  >0.50 count: {(pred_df['p_delist'] > 0.5).sum()}")
    print(f"  >0.30 count: {(pred_df['p_delist'] > 0.3).sum()}")
    print(f"  >0.10 count: {(pred_df['p_delist'] > 0.1).sum()}")

    # Yearly delist counts (for context)
    status_d = status[status["list_status"] == "D"].copy()
    status_d["year"] = status_d["delist_date"].astype(str).str[:4]
    by_year = status_d["year"].value_counts().sort_index().to_dict()

    payload = {
        "inference_snapshot": args.inference_snap,
        "feature_columns": feat_cols,
        "feature_importance": out["feature_importance"],
        "metrics": out["metrics"],
        "best_iteration": out["best_iteration"],
        "n_listed": int((status.list_status == "L").sum()),
        "n_delisted_total": int((status.list_status == "D").sum()),
        "delistings_by_year": by_year,
        "snapshot_grid": make_snapshots(),
        "predictions": [
            {
                "ts_code": r["ts_code"],
                "name": r["name"],
                "p_delist": float(r["p_delist"]),
                **{f: (None if (isinstance(r.get(f), float) and np.isnan(r[f]))
                       else (float(r[f]) if isinstance(r[f], (int, float, np.floating, np.integer))
                             else r[f]))
                   for f in feat_cols},
            }
            for _, r in pred_df.iterrows()
        ],
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str, separators=(",", ":"))
    print(f"\nSaved -> {out_p}  ({out_p.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
