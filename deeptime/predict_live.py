"""
Live inference: predict excess returns for T+1..T+5 for all stocks.

Uses the most recent seq_len days of price/feature data as input, then
predicts excess returns (pct_chg - CSI300_pct_chg) for the next 5 trading days.

Usage:
    python -m deeptime.predict_live                          # all stocks in sh/ sz/
    python -m deeptime.predict_live --max_stocks 1000        # subset
    python -m deeptime.predict_live --top 30                 # show top/bottom stocks

Output:
    plots/deeptime_results/live_predictions_<date>.csv
"""

import argparse
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from deeptime.config import (
    get_config, FORWARD_WINDOWS, NUM_HORIZONS, get_horizon_name,
    DT_FEATURE_COLUMNS, _DT_FUTURE_FEAT_IDX, _DT_OBS_PAST_FEAT_IDX,
    NUM_DT_OBSERVED_PAST, NUM_DT_KNOWN_FUTURE,
    SEQUENCE_LENGTH, MAX_FORWARD_WINDOW,
)
from deeptime.model import create_deeptime_model

from dl.data_processing import (
    load_sector_data, load_daily_basic_data, load_market_context_data,
    load_index_membership_data, load_stk_limit_data, load_moneyflow_data,
    calculate_technical_features, calculate_date_features,
    apply_cs_normalization, compute_daily_cs_stats,
    merge_daily_basic, merge_market_context, merge_index_membership,
    merge_stk_limit, merge_moneyflow,
    get_chinese_holidays_for_year,
)
from deeptime.data_processing import (
    load_fina_indicator_data, forward_fill_fundamentals,
    load_block_trade_data, _pregroup_block_trade, merge_block_trade_features,
    compute_extended_moneyflow, compute_price_limit_ratios,
)


# ─── Trading calendar ─────────────────────────────────────────────────────────

def get_next_trading_days(last_date: pd.Timestamp, n: int = 5) -> list:
    """Return the next n trading days after last_date (skip weekends + CN holidays)."""
    holidays = set()
    for yr in [last_date.year, last_date.year + 1]:
        for h in get_chinese_holidays_for_year(yr):
            holidays.add(h[0].date())

    days = []
    d = last_date + timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5 and d.date() not in holidays:
            days.append(d)
        d += timedelta(days=1)
    return days


# ─── Feature engineering for a single stock ───────────────────────────────────

def build_live_sequence(
    ts_code:         str,
    filepath:        str,
    sector_id:       int,
    industry_id:     int,
    sub_ind_id:      int,
    size_id:         int,
    daily_basic_dict: dict,
    daily_cs_stats:  dict,
    market_context:  pd.DataFrame,
    index_membership: dict,
    stk_limit_dict:  dict,
    moneyflow_dict:  dict,
    fina_dict:       dict,
    block_dict:      dict,
    cs_tech_stats:   dict,
    future_dates:    list,
    seq_len:         int = 30,
    area_dict:       dict = None,
    board_dict:      dict = None,
    ipo_age_dict:    dict = None,
) -> dict | None:
    """
    Build the obs_seq and future_inputs tensors for the most recent window.
    Returns None if insufficient data.
    """
    try:
        df = pd.read_csv(filepath)
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df = df.sort_values('trade_date').reset_index(drop=True)

        bare = str(ts_code).split('.')[0]

        # Feature engineering (same pipeline as training)
        stock_basic = daily_basic_dict.get(bare, pd.DataFrame())
        df = merge_daily_basic(df, stock_basic, ts_code=None)
        if daily_cs_stats:
            df = apply_cs_normalization(df, daily_cs_stats)
        if market_context is not None:
            df = merge_market_context(df, market_context)
        else:
            from dl.config import MARKET_CONTEXT_FEATURES
            for c in MARKET_CONTEXT_FEATURES: df[c] = 0.0
        if index_membership is not None:
            df = merge_index_membership(df, index_membership, ts_code)
        else:
            from dl.config import INDEX_MEMBERSHIP_FEATURES
            for c in INDEX_MEMBERSHIP_FEATURES: df[c] = 0.0

        df = merge_stk_limit(df, stk_limit_dict.get(bare), None)
        df = compute_price_limit_ratios(df, stk_limit_dict.get(bare))
        df = merge_moneyflow(df, moneyflow_dict.get(bare), None)
        df = compute_extended_moneyflow(df, moneyflow_dict.get(bare))
        df = calculate_technical_features(df)
        if cs_tech_stats:
            from deeptime.config import DT_CS_NORMALIZE_TECH_FEATURES
            df = apply_cs_normalization(df, cs_tech_stats, DT_CS_NORMALIZE_TECH_FEATURES)

        df = forward_fill_fundamentals(df, fina_dict.get(bare))
        df = merge_block_trade_features(df, block_dict.get(bare))

        for col in DT_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        df = df.dropna(subset=DT_FEATURE_COLUMNS)
        if len(df) < seq_len:
            return None

        # Use the last seq_len rows as the input window
        window = df.tail(seq_len).reset_index(drop=True)
        last_date = window['trade_date'].iloc[-1]

        features = window[DT_FEATURE_COLUMNS].values.astype('float32')
        obs_idx  = np.array(_DT_OBS_PAST_FEAT_IDX, dtype=np.intp)
        obs_seq  = features[:, obs_idx]   # (seq_len, n_past)

        # ── Build known-future features for next 5 trading days ──────────────
        # Create a synthetic future DataFrame for calendar/price-limit features
        future_df = pd.DataFrame({'trade_date': future_dates})
        future_df = calculate_date_features(future_df)

        # Price limits for future dates: use last available stk_limit or ±10% default
        last_limit = stk_limit_dict.get(bare)
        if last_limit is not None and len(last_limit) > 0:
            last_row = last_limit.sort_values('trade_date').iloc[-1]
            pre_close = float(window['close'].iloc[-1])
            up_lim = float(last_row.get('up_limit', pre_close * 1.1))
            dn_lim = float(last_row.get('down_limit', pre_close * 0.9))
            up_ratio = np.clip(up_lim / (pre_close + 1e-8) - 1.0, 0.04, 0.22)
            dn_ratio = np.clip(dn_lim / (pre_close + 1e-8) - 1.0, -0.22, -0.04)
        else:
            up_ratio, dn_ratio = 0.10, -0.10

        future_df['up_limit_ratio']   = up_ratio
        future_df['down_limit_ratio'] = dn_ratio

        # Ensure all known-future feature columns exist
        from deeptime.config import DT_KNOWN_FUTURE_COLUMNS
        for col in DT_KNOWN_FUTURE_COLUMNS:
            if col not in future_df.columns:
                future_df[col] = 0.0

        fut_idx = np.array(_DT_FUTURE_FEAT_IDX, dtype=np.intp)
        # Reconstruct: DT_KNOWN_FUTURE_COLUMNS are a subset of DT_FEATURE_COLUMNS
        # For future_inputs, we need (max_fw, n_future)
        future_arr = future_df[DT_KNOWN_FUTURE_COLUMNS].values[:MAX_FORWARD_WINDOW].astype('float32')

        # Look up new static IDs from enriched sector data
        area_id  = area_dict.get(ts_code, area_dict.get(bare, 0))     if area_dict  else 0
        board_id = board_dict.get(ts_code, board_dict.get(bare, 0))   if board_dict else 0
        ipo_id   = ipo_age_dict.get(ts_code, ipo_age_dict.get(bare, 6)) if ipo_age_dict else 6

        return {
            'ts_code':     ts_code,
            'last_date':   last_date,
            'obs_seq':     obs_seq,
            'future_inputs': future_arr,
            'sector_id':   sector_id,
            'industry_id': industry_id,
            'sub_ind_id':  sub_ind_id,
            'size_id':     size_id,
            'area_id':     area_id,
            'board_id':    board_id,
            'ipo_age_id':  ipo_id,
        }

    except Exception as e:
        return None


# ─── Batch inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(sequences: list, model, device: str, batch_size: int = 256):
    """Run model on a list of sequence dicts, return predictions array (N, 5)."""
    model.eval()
    all_preds = []

    for b_start in range(0, len(sequences), batch_size):
        batch = sequences[b_start:b_start + batch_size]

        obs    = torch.tensor(np.stack([s['obs_seq']      for s in batch]), dtype=torch.float32).to(device)
        future = torch.tensor(np.stack([s['future_inputs'] for s in batch]), dtype=torch.float32).to(device)
        def _ids(key, default=0):
            return torch.tensor([s.get(key, default) for s in batch], dtype=torch.long).to(device)
        sec   = _ids('sector_id');  ind  = _ids('industry_id')
        sub   = _ids('sub_ind_id'); sz   = _ids('size_id')
        area  = _ids('area_id');    board = _ids('board_id')
        ipo   = _ids('ipo_age_id', default=6)

        with torch.autocast(device.split(':')[0], torch.float16, enabled=(device != 'cpu')):
            preds = model(obs, future, sec, ind, sub, sz, area, board, ipo)

        all_preds.append(preds.float().cpu().numpy())

    return np.concatenate(all_preds, axis=0)


# ─── Inference-time normalization (matches normalize_cache) ───────────────────

def _apply_live_scaler(sequences: list, cache_dir: str) -> list:
    """
    Apply the same Tier-1/Tier-2 normalization that was fitted on the training
    cache to live inference sequences, ensuring training/inference consistency.

    Loads feature_scaler.npz saved by normalize_cache().  If not found (e.g.
    older cache), logs a warning and returns sequences unchanged.
    """
    scaler_path = os.path.join(cache_dir, 'feature_scaler.npz')
    if not os.path.exists(scaler_path):
        print("  [warn] No feature_scaler.npz in cache — using unnormalized features")
        return sequences

    s = np.load(scaler_path)
    clip_lo    = s['clip_lo'].astype('float64')
    clip_hi    = s['clip_hi'].astype('float64')
    tier1_idx  = s['tier1_idx'].astype(int)
    tier1_mean = s['tier1_mean'].astype('float64')
    tier1_std  = s['tier1_std'].astype('float64')

    n_past = len(clip_lo)
    for seq in sequences:
        obs = seq['obs_seq'][:, :n_past].astype('float64')   # (T, n_past)
        # Tier 2: global ±5σ safety clip
        np.clip(obs, clip_lo[np.newaxis, :], clip_hi[np.newaxis, :], out=obs)
        # Tier 1: standardize fina + market MACD/MTM features
        for fi in tier1_idx:
            if fi < obs.shape[1] and tier1_std[fi] > 1e-8:
                obs[:, fi] = (obs[:, fi] - tier1_mean[fi]) / tier1_std[fi]
        seq['obs_seq'] = obs.astype('float32')

    print(f"  Applied feature scaler ({len(tier1_idx)} Tier-1 features standardized, "
          f"all {n_past} features clipped)")
    return sequences


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='deeptime live prediction')
    p.add_argument('--max_stocks', type=int, default=0, help='0 = all stocks')
    p.add_argument('--top',        type=int, default=20, help='Show top/bottom N predictions')
    p.add_argument('--batch_size', type=int, default=256)
    # Model architecture (must match training checkpoint)
    p.add_argument('--hidden',      type=int, default=None,
                   help='TFT hidden dim (default from config: 128)')
    p.add_argument('--heads',       type=int, default=None,
                   help='Attention heads (default from config: 4)')
    p.add_argument('--lstm_layers', type=int, default=None,
                   help='LSTM layers (default from config: 2)')
    p.add_argument('--dropout',     type=float, default=None,
                   help='Dropout rate (default: 0.15)')
    p.add_argument('--seed',        type=int, default=42,
                   help='Random seed for stock sampling')
    args = p.parse_args()

    # Build config with CLI overrides
    overrides = {}
    if args.hidden      is not None: overrides['tft_hidden']      = args.hidden
    if args.heads       is not None: overrides['tft_heads']       = args.heads
    if args.lstm_layers is not None: overrides['tft_lstm_layers'] = args.lstm_layers
    if args.dropout     is not None: overrides['tft_dropout']     = args.dropout

    config   = get_config(**overrides)
    data_dir = config['data_dir']
    device   = config['device']
    seq_len  = config['sequence_length']

    # ── Load model ─────────────────────────────────────────────────────────
    model = create_deeptime_model(config).to(device).eval()
    ckpt_path = config.get('model_save_path',
                           os.path.join(config['data_dir'], 'deeptime_model.pth'))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    print(f"Model: epoch {ckpt['epoch']}, val IC={ckpt['val_ic']:.4f}")

    # ── Stock file list ─────────────────────────────────────────────────────
    stock_files = []
    for subdir in ['sh', 'sz']:
        d = os.path.join(data_dir, subdir)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.csv'):
                    ts_code = f.replace('.csv', '')
                    stock_files.append((ts_code, os.path.join(d, f)))

    if args.max_stocks > 0 and args.max_stocks < len(stock_files):
        # Use seeded random sampling (same as main.py training)
        import random
        rng = random.Random(args.seed)
        stock_files = rng.sample(stock_files, args.max_stocks)
        print(f"Sampled {len(stock_files)} stocks (seed={args.seed})")
    else:
        print(f"Predicting {len(stock_files)} stocks...")

    # ── Load auxiliary data ─────────────────────────────────────────────────
    sector_data    = load_sector_data(data_dir)
    daily_basic    = load_daily_basic_data(data_dir)
    market_context = load_market_context_data(data_dir)
    index_membership = load_index_membership_data(data_dir)
    stk_limit      = load_stk_limit_data(data_dir)
    moneyflow      = load_moneyflow_data(data_dir)

    ts_codes = [ts for ts, _ in stock_files]
    fina_data = load_fina_indicator_data(data_dir, ts_codes)
    block_trade_daily = load_block_trade_data(data_dir)
    block_by_stock    = _pregroup_block_trade(block_trade_daily)
    del block_trade_daily

    # Pre-group
    import gc
    daily_basic_dict = {}
    if len(daily_basic) > 0:
        for key, grp in daily_basic.groupby('ts_code'):
            daily_basic_dict[str(key).split('.')[0]] = grp.drop(columns=['ts_code'], errors='ignore').reset_index(drop=True)
        del daily_basic; gc.collect()

    daily_cs_stats = compute_daily_cs_stats(daily_basic_dict)

    stk_limit_dict = {}
    if len(stk_limit) > 0:
        for key, grp in stk_limit.groupby('ts_code'):
            stk_limit_dict[str(key).split('.')[0]] = grp.reset_index(drop=True)
        del stk_limit; gc.collect()

    moneyflow_dict = {}
    if len(moneyflow) > 0:
        for key, grp in moneyflow.groupby('ts_code'):
            moneyflow_dict[str(key).split('.')[0]] = grp.reset_index(drop=True)
        del moneyflow; gc.collect()

    # Sector / industry encodings
    sector_to_id   = {s: i for i, s in enumerate(sector_data['sector'].unique())} if len(sector_data) else {}
    sector_to_id['Unknown'] = len(sector_to_id)
    industry_to_id = {}
    sub_ind_to_id  = {}
    if len(sector_data) and 'industry' in sector_data.columns:
        for i, ind in enumerate(sector_data['industry'].dropna().unique()):
            industry_to_id[str(ind)] = i
    industry_to_id['Unknown'] = len(industry_to_id)
    if len(sector_data) and 'sub_industry' in sector_data.columns:
        for i, sub in enumerate(sector_data['sub_industry'].dropna().unique()):
            sub_ind_to_id[str(sub)] = i
    sub_ind_to_id['Unknown'] = len(sub_ind_to_id)

    # Build int-ID encodings for area and board BEFORE the per-row loop
    # (area_dict/board_dict must store int IDs, not strings, for torch.tensor)
    def _enum(series):
        vals = sorted(set(str(v) for v in series.dropna().unique() if str(v) != 'Unknown'))
        m = {v: i for i, v in enumerate(vals)}
        m['Unknown'] = len(m)
        return m

    area_to_id_enc  = _enum(sector_data['area'])   if 'area'   in sector_data.columns else {'Unknown': 0}
    board_to_id_enc = _enum(sector_data['market']) if 'market' in sector_data.columns else {'Unknown': 0}

    sector_dict = {}; industry_dict = {}; sub_ind_dict = {}
    area_dict = {}; board_dict = {}; ipo_age_dict = {}
    for _, row in sector_data.iterrows():
        code = str(row['ts_code']); bare = code.split('.')[0]
        for d in [code, bare]:
            sector_dict[d]   = str(row.get('sector',   'Unknown'))
            industry_dict[d] = str(row.get('industry', 'Unknown'))
            sub_ind_dict[d]  = 'Unknown'
            # Store int IDs directly (not string names) so torch.tensor works
            area_dict[d]  = area_to_id_enc.get(str(row.get('area',   'Unknown')), 0)
            board_dict[d] = board_to_id_enc.get(str(row.get('market', 'Unknown')), 0)
            # IPO age bucket (0-5)
            try:
                ld = pd.to_datetime(str(row.get('list_date', '')), errors='coerce')
                if pd.isna(ld):
                    ipo_age_dict[d] = 6
                else:
                    yrs = (pd.Timestamp.today() - ld).days / 365.25
                    ipo_age_dict[d] = (0 if yrs<1 else 1 if yrs<2 else 2 if yrs<3
                                       else 3 if yrs<5 else 4 if yrs<10 else 5)
            except Exception:
                ipo_age_dict[d] = 6

    # CS tech stats (lightweight first pass on first 200 stocks)
    from dl.data_processing import compute_cross_section_tech_stats
    print("Computing CS tech stats (fast pass on subset)...")
    cs_tech_stats = compute_cross_section_tech_stats(
        stock_files[:min(200, len(stock_files))], min_data_points=100
    )

    # ── Determine next 5 trading days ──────────────────────────────────────
    # Use the last date from first stock as reference
    sample_df = pd.read_csv(stock_files[0][1])
    last_date = pd.to_datetime(sample_df['trade_date'].astype(str)).max()
    future_dates = get_next_trading_days(last_date, n=MAX_FORWARD_WINDOW)
    print(f"Last data date: {last_date.date()}")
    print(f"Predicting for: {[str(d.date()) for d in future_dates]}")

    # ── Build sequences ─────────────────────────────────────────────────────
    sequences = []
    skipped   = 0
    print(f"\nBuilding sequences...")

    for i, (ts_code, filepath) in enumerate(stock_files):
        bare = str(ts_code).split('.')[0]
        sector   = sector_dict.get(ts_code, 'Unknown')
        sec_id   = sector_to_id.get(sector, sector_to_id['Unknown'])
        ind_str  = industry_dict.get(ts_code, 'Unknown')
        ind_id   = industry_to_id.get(ind_str, industry_to_id['Unknown'])
        sub_str  = sub_ind_dict.get(ts_code, 'Unknown')
        sub_id   = sub_ind_to_id.get(sub_str, sub_ind_to_id['Unknown'])
        size_id  = 10  # default; exact decile requires cross-section

        result = build_live_sequence(
            ts_code, filepath,
            sec_id, ind_id, sub_id, size_id,
            daily_basic_dict, daily_cs_stats,
            market_context, index_membership,
            stk_limit_dict, moneyflow_dict,
            fina_data, block_by_stock,
            cs_tech_stats, future_dates, seq_len,
            area_dict=area_dict,
            board_dict=board_dict,
            ipo_age_dict=ipo_age_dict,
        )
        if result is None:
            skipped += 1
        else:
            sequences.append(result)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(stock_files)} stocks built ({skipped} skipped)")

    print(f"\n{len(sequences)} sequences built, {skipped} skipped")

    # ── Apply same normalization as training (load scaler from cache) ─────────
    sequences = _apply_live_scaler(sequences, config['cache_dir'])

    # ── Run inference ────────────────────────────────────────────────────────
    print("Running inference...")
    preds = run_inference(sequences, model, device, args.batch_size)

    # ── Build output DataFrame ───────────────────────────────────────────────
    # ── Compute bull probabilities via cross-sectional sigmoid normalization ──
    # P(outperform CSI300) for each horizon: normalize predictions across all
    # stocks on the same date, then apply sigmoid.
    # sigmoid(z) where z = (pred - median) / std gives calibrated rank-based
    # probability: top stock → ~90%, median stock → 50%, bottom → ~10%.
    bull_probs = np.zeros_like(preds)
    for h in range(NUM_HORIZONS):
        col = preds[:, h]
        mu  = np.nanmean(col)
        sig = np.nanstd(col) + 1e-6
        z   = (col - mu) / sig
        bull_probs[:, h] = 1.0 / (1.0 + np.exp(-z))   # sigmoid

    rows = []
    for i, seq in enumerate(sequences):
        row = {
            'ts_code':   seq['ts_code'],
            'last_date': str(seq['last_date'].date()),
        }
        for h in range(NUM_HORIZONS):
            hn = get_horizon_name(h)
            target_date = str(future_dates[h].date())
            row[f'pred_{hn}']      = round(float(preds[i, h]), 4)
            row[f'prob_{hn}']      = round(float(bull_probs[i, h]), 4)
            row[f'pred_date_{hn}'] = target_date
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('pred_day1', ascending=False).reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir  = os.path.join(_ROOT, 'plots', 'deeptime_results')
    os.makedirs(out_dir, exist_ok=True)
    date_str = str(last_date.date()).replace('-', '')
    out_path = os.path.join(out_dir, f'live_predictions_{date_str}.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} predictions to {out_path}")

    # ── Print top/bottom ─────────────────────────────────────────────────────
    if args.top > 0:
        pred_cols = [f'pred_{get_horizon_name(h)}' for h in range(NUM_HORIZONS)]
        prob_cols = [f'prob_{get_horizon_name(h)}' for h in range(NUM_HORIZONS)]
        show_cols = ['ts_code', 'last_date'] + pred_cols + prob_cols

        print(f"\n{'='*60}")
        print(f"Top {args.top} stocks by Day+1 predicted excess return (prob = P(outperform CSI300)):")
        print(df.head(args.top)[show_cols].to_string(index=False))

        print(f"\nBottom {args.top} stocks by Day+1 predicted excess return:")
        print(df.tail(args.top)[show_cols].to_string(index=False))

    return df


if __name__ == '__main__':
    main()
