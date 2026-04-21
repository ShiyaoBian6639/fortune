"""
Prediction functions for trained models.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import NUM_CLASSES, NUM_HORIZONS, FEATURE_COLUMNS, CHANGE_BUCKETS, get_class_names, get_horizon_name
from .models import TransformerClassifier, TemperatureScaler
from .data_processing import (
    calculate_technical_features, load_sector_data,
    load_daily_basic_data, merge_daily_basic,
    load_market_context_data, load_index_membership_data,
    load_stk_limit_data, load_moneyflow_data,
    merge_market_context, merge_index_membership,
    merge_stk_limit, merge_moneyflow,
    get_stock_files,
    compute_daily_cs_stats, apply_cs_normalization,
)


def predict_specific_stocks(
    stock_codes: List[str],
    model_path: str,
    data_dir: str,
    sector_data: pd.DataFrame = None,
    device: str = 'cpu'
) -> Dict[str, Dict]:
    """
    Make predictions for specific stocks using a trained model.

    Args:
        stock_codes: List of stock codes (e.g., ['001270', '300788'])
        model_path: Path to the saved model checkpoint
        data_dir: Directory containing stock data
        sector_data: DataFrame with sector information (loaded if None)
        device: Device to run predictions on

    Returns:
        Dictionary mapping stock codes to prediction results
    """
    print("\n" + "=" * 60)
    print("Predicting for Specific Stocks")
    print("=" * 60)

    # Load model checkpoint
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return {}

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_config = checkpoint['config']

    # Reconstruct feature scaler
    feat_scaler = StandardScaler()
    feat_scaler.mean_  = checkpoint['scaler_mean']
    feat_scaler.scale_ = checkpoint['scaler_scale']

    input_dim = len(feat_scaler.mean_)

    # Load temperature scaler (calibration) if saved in checkpoint
    num_horizons = saved_config.get('num_horizons', NUM_HORIZONS)
    temp_scaler = TemperatureScaler(num_horizons).to(device)
    if 'temperatures' in checkpoint:
        temp_scaler.temperatures = torch.nn.Parameter(
            checkpoint['temperatures'].to(device)
        )

    # Load sector data if not provided
    if sector_data is None:
        sector_data = load_sector_data(data_dir)

    # Load only recent fundamental data — prediction needs ~300 days of history
    _recent = 300
    daily_basic = load_daily_basic_data(data_dir, last_n_files=_recent)

    # Pre-compute daily cross-section PE/PB stats so that apply_cs_normalization()
    # can be called per-stock — mirrors what prepare_dataset_to_disk does at train time.
    _daily_basic_dict = {}
    if len(daily_basic) > 0:
        for _key, _grp in daily_basic.groupby('ts_code'):
            _bare = str(_key).split('.')[0]
            _daily_basic_dict[_bare] = _grp.drop(columns=['ts_code'], errors='ignore').reset_index(drop=True)
    daily_cs_stats = compute_daily_cs_stats(_daily_basic_dict) if _daily_basic_dict else {}

    # Load market context and index membership (same sources as training)
    market_context   = load_market_context_data(data_dir)
    index_membership = load_index_membership_data(data_dir)

    # Load stk_limit and moneyflow (zeros if not downloaded yet)
    stk_limit_df = load_stk_limit_data(data_dir, last_n_files=_recent)
    stk_limit = stk_limit_df if len(stk_limit_df) > 0 else None
    moneyflow_df = load_moneyflow_data(data_dir, last_n_files=_recent)
    moneyflow = moneyflow_df if len(moneyflow_df) > 0 else None

    # Get number of sectors / industries
    num_sectors = len(sector_data['sector'].unique()) if len(sector_data) > 0 else 0

    # Create sector encoding
    if len(sector_data) > 0:
        sector_to_id = {sector: i for i, sector in enumerate(sector_data['sector'].unique())}
    else:
        sector_to_id = {}
    sector_to_id['Unknown'] = len(sector_to_id)

    # Create industry encoding
    if len(sector_data) > 0 and 'industry' in sector_data.columns:
        industry_to_id = {ind: i for i, ind in enumerate(sector_data['industry'].dropna().unique())}
    else:
        industry_to_id = {}
    industry_to_id['Unknown'] = len(industry_to_id)
    num_industries = max(0, len(industry_to_id) - 1)

    # Create model and load weights
    from .config import NUM_RELATIVE_CLASSES
    _model_type = saved_config.get('model_type', 'transformer')
    if _model_type == 'tft':
        from .models import create_tft_model
        model = create_tft_model(saved_config, num_sectors, num_industries).to(device)
    else:
        model = TransformerClassifier(
            input_dim             = input_dim,
            num_classes           = NUM_CLASSES,
            num_horizons          = num_horizons,
            d_model               = saved_config['d_model'],
            nhead                 = saved_config['nhead'],
            num_layers            = saved_config['num_layers'],
            dim_feedforward       = saved_config['dim_feedforward'],
            dropout               = saved_config['dropout'],
            num_sectors           = num_sectors,
            use_sector            = (num_sectors > 0),
            num_industries        = num_industries,
            use_relative_head     = saved_config.get('use_relative_head', False),
            num_relative_classes  = NUM_RELATIVE_CLASSES,
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = {}
    class_names     = get_class_names()
    sequence_length = saved_config['sequence_length']

    # 7-class schema: bear=0-2, neutral=3, bull=4-6  (boundaries from CHANGE_BUCKETS)
    BEAR_CLASSES = list(range(0, 3))
    BULL_CLASSES = list(range(4, 7))

    for stock_code in stock_codes:
        print(f"\n{'-' * 50}")
        print(f"Stock: {stock_code}")
        print(f"{'-' * 50}")

        # Find stock file
        stock_path = None
        for market in ['sz', 'sh']:
            possible_path = os.path.join(data_dir, market, f"{stock_code}.csv")
            if os.path.exists(possible_path):
                stock_path = possible_path
                break

        if stock_path is None:
            print(f"  Stock data not found for {stock_code}")
            results[stock_code] = {'error': 'Stock data not found'}
            continue

        # Load and process stock data
        try:
            df = pd.read_csv(stock_path)
            df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
            df = df.sort_values('trade_date').reset_index(drop=True)
        except Exception as e:
            print(f"  Error loading stock data: {e}")
            results[stock_code] = {'error': str(e)}
            continue

        ts_code = f"{stock_code}.SZ" if 'sz' in stock_path else f"{stock_code}.SH"

        df = merge_daily_basic(df, daily_basic, ts_code)
        if daily_cs_stats:
            df = apply_cs_normalization(df, daily_cs_stats)
        df = merge_market_context(df, market_context)
        df = merge_index_membership(df, index_membership, stock_code)
        df = merge_stk_limit(df, stk_limit, ts_code)
        df = merge_moneyflow(df, moneyflow, ts_code)
        df = calculate_technical_features(df)

        sector_info  = sector_data[sector_data['ts_code'] == ts_code] if len(sector_data) > 0 else pd.DataFrame()
        sector       = sector_info['sector'].values[0]   if len(sector_info) > 0 else 'Unknown'
        industry     = sector_info['industry'].values[0] if (len(sector_info) > 0 and 'industry' in sector_info.columns) else 'Unknown'
        sector_id    = sector_to_id.get(sector,   sector_to_id['Unknown'])
        industry_id  = industry_to_id.get(industry, industry_to_id['Unknown'])

        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        df = df.dropna(subset=FEATURE_COLUMNS)

        if len(df) < sequence_length + 1:
            print(f"  Insufficient data (need {sequence_length + 1} days, got {len(df)})")
            results[stock_code] = {'error': 'Insufficient data'}
            continue

        latest_close = df['close'].iloc[-1]
        latest_date  = df['trade_date'].iloc[-1]

        seq = df[FEATURE_COLUMNS].values[-sequence_length:]
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        seq_norm = feat_scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(1, sequence_length, -1)
        seq_norm = np.nan_to_num(seq_norm, nan=0.0, posinf=0.0, neginf=0.0)
        seq_norm = np.clip(seq_norm, -5.0, 5.0)

        seq_t = torch.FloatTensor(seq_norm).to(device)
        sec_t = torch.LongTensor([sector_id]).to(device)
        ind_t = torch.LongTensor([industry_id]).to(device)

        # For TFT, build future_inputs from the last row's known-future features.
        # In prediction mode the "future" days aren't in df, so we replicate the
        # last available row's calendar features as a best-effort approximation.
        fut_t = None
        if getattr(model, '_is_tft', False):
            from .config import _FUTURE_FEAT_IDX, FORWARD_WINDOWS as _fw
            _max_fw = max(_fw)
            last_row = df[FEATURE_COLUMNS].values[-1]
            fut_feat = np.stack([last_row[_FUTURE_FEAT_IDX]] * _max_fw, axis=0)  # (_max_fw, 27)
            fut_t = torch.FloatTensor(fut_feat[np.newaxis]).to(device)            # (1, _max_fw, 27)

        with torch.no_grad():
            if fut_t is not None:
                out = model(seq_t, fut_t, sec_t, ind_t)
            else:
                out = model(seq_t, sec_t, ind_t)
            logits = out[0] if isinstance(out, tuple) else out   # (1, H, C)
            logits = temp_scaler(logits.float())
            probs  = torch.softmax(logits, dim=2).cpu().numpy()[0]   # (H, C)

        # per-horizon probabilities
        horizon_results = {}
        for h in range(num_horizons):
            hname = get_horizon_name(h)
            p = probs[h]                             # (C,)
            pred_cls = int(np.argmax(p))
            horizon_results[hname] = {
                'predicted_class': pred_cls,
                'predicted_label': class_names[pred_cls],
                'confidence': float(p[pred_cls]),
                'bull_prob':  float(p[BULL_CLASSES].sum()),
                'bear_prob':  float(p[BEAR_CLASSES].sum()),
                'probs':      {class_names[i]: float(p[i]) for i in range(NUM_CLASSES)},
            }

        # Average probabilities across horizons
        avg_probs = probs.mean(axis=0)              # (C,)
        avg_pred  = int(np.argmax(avg_probs))
        avg_result = {
            'predicted_class': avg_pred,
            'predicted_label': class_names[avg_pred],
            'confidence':      float(avg_probs[avg_pred]),
            'bull_prob':       float(avg_probs[BULL_CLASSES].sum()),
            'bear_prob':       float(avg_probs[BEAR_CLASSES].sum()),
            'probs':           {class_names[i]: float(avg_probs[i]) for i in range(NUM_CLASSES)},
        }

        results[stock_code] = {
            'latest_date':  latest_date.strftime('%Y-%m-%d'),
            'latest_close': float(latest_close),
            'sector':       sector,
            'horizons':     horizon_results,
            'average':      avg_result,
        }

        # ── Print results ─────────────────────────────────────────────────────
        print(f"  Latest Date:  {latest_date.strftime('%Y-%m-%d')}")
        print(f"  Latest Close: {latest_close:.2f}")
        print(f"  Sector:       {sector}")

        for h in range(num_horizons):
            hname = get_horizon_name(h)
            hr    = horizon_results[hname]
            print(f"\n  {hname.upper()}:")
            print(f"    Predicted: {hr['predicted_label']}  (conf {hr['confidence']*100:.1f}%)")
            print(f"    Bull (>+1%): {hr['bull_prob']*100:.1f}%  "
                  f"Bear (<-1%): {hr['bear_prob']*100:.1f}%  "
                  f"Neutral: {(1-hr['bull_prob']-hr['bear_prob'])*100:.1f}%")
            top3 = sorted(hr['probs'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Top-3: " + "  ".join(f"{cn}:{p*100:.1f}%" for cn, p in top3))

        print(f"\n  AVERAGE ACROSS HORIZONS:")
        print(f"    Predicted: {avg_result['predicted_label']}  (conf {avg_result['confidence']*100:.1f}%)")
        print(f"    Bull (>+1%): {avg_result['bull_prob']*100:.1f}%  "
              f"Bear (<-1%): {avg_result['bear_prob']*100:.1f}%  "
              f"Neutral: {(1-avg_result['bull_prob']-avg_result['bear_prob'])*100:.1f}%")

    return results


def _load_checkpoint(model_path: str, device: str, data_dir: str):
    """
    Load checkpoint and reconstruct model + scalers.

    Returns (model, feat_scaler, config, sector_data, temp_scaler).
    temp_scaler is always returned; temperatures default to 1.0 if not saved.
    """
    checkpoint   = torch.load(model_path, map_location=device, weights_only=False)
    saved_config = checkpoint['config']

    feat_scaler = StandardScaler()
    feat_scaler.mean_  = checkpoint['scaler_mean']
    feat_scaler.scale_ = checkpoint['scaler_scale']

    num_horizons = saved_config.get('num_horizons', NUM_HORIZONS)
    temp_scaler  = TemperatureScaler(num_horizons).to(device)
    if 'temperatures' in checkpoint:
        temp_scaler.temperatures = torch.nn.Parameter(
            checkpoint['temperatures'].to(device)
        )

    sector_data    = load_sector_data(data_dir)
    num_sectors    = len(sector_data['sector'].unique()) if len(sector_data) > 0 else 0
    num_industries = (
        len(sector_data['industry'].dropna().unique())
        if len(sector_data) > 0 and 'industry' in sector_data.columns
        else 0
    )

    from .config import NUM_RELATIVE_CLASSES
    model = TransformerClassifier(
        input_dim             = len(feat_scaler.mean_),
        num_classes           = NUM_CLASSES,
        num_horizons          = num_horizons,
        d_model               = saved_config['d_model'],
        nhead                 = saved_config['nhead'],
        num_layers            = saved_config['num_layers'],
        dim_feedforward       = saved_config['dim_feedforward'],
        dropout               = saved_config['dropout'],
        num_sectors           = num_sectors,
        use_sector            = (num_sectors > 0),
        num_industries        = num_industries,
        use_relative_head     = saved_config.get('use_relative_head', False),
        num_relative_classes  = NUM_RELATIVE_CLASSES,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, feat_scaler, saved_config, sector_data, temp_scaler


def predict_all_stocks(
    model_path:      str,
    data_dir:        str,
    output_csv:      Optional[str] = None,
    batch_size:      int           = 512,
    device:          str           = 'cuda',
    top_n:           int           = 20,
) -> pd.DataFrame:
    """
    Predict next-day price change for every stock in data_dir.

    Strategy: collect the latest `sequence_length`-day window from each stock
    CSV, stack all windows into one array, then run a single batched inference
    pass on the GPU — far faster than calling the model once per stock.

    Args:
        model_path: Path to the .pth checkpoint saved by training.
        data_dir:   Root data directory (contains sh/, sz/ sub-dirs).
        output_csv: If given, save the full results DataFrame here.
        batch_size: Inference batch size (512 fits any GPU).
        device:     'cuda' or 'cpu'.
        top_n:      Print this many top/bottom predictions to the console.

    Returns:
        DataFrame sorted by expected_return descending, with columns:
        stock_code, ts_code, market, sector, latest_date, latest_close,
        predicted_label, confidence, expected_return, bull_prob, bear_prob,
        prob_<class> …
    """
    import time
    from .numba_optimizations import warmup as numba_warmup
    print("\n" + "=" * 60)
    print("Batch Prediction — All Stocks")
    print("=" * 60)

    # Warm up numba JIT once so per-stock calls don't pay compilation cost
    print("Warming up numba JIT...")
    numba_warmup()

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return pd.DataFrame()

    model, scaler, saved_config, sector_data, temp_scaler = _load_checkpoint(model_path, device, data_dir)
    sequence_length = saved_config['sequence_length']
    num_horizons    = saved_config.get('num_horizons', NUM_HORIZONS)
    class_names     = get_class_names()

    sector_to_id = {s: i for i, s in enumerate(sector_data['sector'].unique())} \
                   if len(sector_data) > 0 else {}
    sector_to_id['Unknown'] = len(sector_to_id)

    industry_to_id = (
        {ind: i for i, ind in enumerate(sector_data['industry'].dropna().unique())}
        if len(sector_data) > 0 and 'industry' in sector_data.columns
        else {}
    )
    industry_to_id['Unknown'] = len(industry_to_id)

    industry_by_ts: dict = {}
    if len(sector_data) > 0 and 'industry' in sector_data.columns:
        industry_by_ts = dict(zip(sector_data['ts_code'], sector_data['industry']))

    # How many rows of history to read per stock CSV.
    # Files are sorted newest-first, so nrows=HISTORY gives the most recent
    # HISTORY trading days without reading the full 9-year history.
    # 250 rows (~1 year) is enough for all indicators to stabilise:
    #   EMA-26 (~130 rows), OBV rolling-std-20, RSI-14, ADX-14, patterns-20.
    HISTORY = 250

    # ── Load fundamental data once ───────────────────────────────────────────
    # For prediction we only need the most recent ~300 trading days (~14 months).
    # HISTORY rows cover the longest indicator window (EMA-26 ≈ 130 rows to converge)
    # plus the 30-day model window. Loading 300 files instead of 2247 cuts startup
    # from ~75s to ~10s with no loss of prediction accuracy.
    RECENT = HISTORY + 50  # small buffer beyond HISTORY
    daily_basic      = load_daily_basic_data(data_dir, last_n_files=RECENT)
    market_context   = load_market_context_data(data_dir)
    index_membership = load_index_membership_data(data_dir)
    stk_limit_df     = load_stk_limit_data(data_dir, last_n_files=RECENT)
    moneyflow_df     = load_moneyflow_data(data_dir, last_n_files=RECENT)

    # Pre-group daily_basic by ts_code for O(1) per-stock lookup.
    daily_basic_by_ts: dict = {}
    _daily_cs_stats_all: dict = {}
    if daily_basic is not None and not daily_basic.empty:
        daily_basic_by_ts = {
            code: grp.reset_index(drop=True)
            for code, grp in daily_basic.groupby('ts_code', sort=False)
        }
        # Compute CS stats from the grouped dict (no extra file I/O)
        _daily_basic_bare = {
            str(code).split('.')[0]: grp
            for code, grp in daily_basic_by_ts.items()
        }
        _daily_cs_stats_all = compute_daily_cs_stats(_daily_basic_bare)
        del daily_basic

    # Pre-group stk_limit by ts_code for O(1) per-stock lookup.
    stk_limit_by_ts: dict = {}
    if len(stk_limit_df) > 0:
        for code, grp in stk_limit_df.groupby('ts_code', sort=False):
            stk_limit_by_ts[code] = grp.reset_index(drop=True)
        del stk_limit_df

    # Pre-group moneyflow by ts_code for O(1) per-stock lookup.
    moneyflow_by_ts: dict = {}
    if len(moneyflow_df) > 0:
        for code, grp in moneyflow_df.groupby('ts_code', sort=False):
            moneyflow_by_ts[code] = grp.reset_index(drop=True)
        del moneyflow_df

    # Pre-index sector data by ts_code for O(1) lookup.
    sector_by_ts: dict = {}
    if len(sector_data) > 0:
        sector_by_ts = dict(zip(sector_data['ts_code'], sector_data['sector']))

    # ── Gather all stock files ────────────────────────────────────────────────
    all_files = []
    for market in ['sh', 'sz']:
        for stock_code, fp in get_stock_files(data_dir, market, max_stocks=None):
            all_files.append((market, stock_code, fp))

    print(f"Found {len(all_files)} stock files")

    # ── Per-stock feature cache ───────────────────────────────────────────────
    # Cache raw (pre-scaler) feature arrays keyed by ts_code.
    # CSVs are sorted newest-first, so we peek only the first row to get the
    # latest trade_date.  If it matches the cached date we skip the expensive
    # calculate_technical_features() call entirely (~60 ms/stock → <1 ms/stock).
    # Cache is invalidated automatically when new data arrives (date changes).
    # We store raw features so the cache remains valid across model retrains.
    _expected_shape = (sequence_length, len(FEATURE_COLUMNS))
    cache_dir = os.path.join(data_dir, 'pred_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # ── Extract latest window per stock ──────────────────────────────────────
    seqs, sector_ids, industry_ids, meta_rows = [], [], [], []
    skipped = cache_hits = cache_misses = 0

    t0 = time.perf_counter()
    for i, (market, stock_code, fp) in enumerate(all_files):
        if (i + 1) % 500 == 0:
            elapsed_so_far = time.perf_counter() - t0
            print(f"  Processing {i+1}/{len(all_files)}... "
                  f"hits={cache_hits}, misses={cache_misses}, "
                  f"elapsed={elapsed_so_far:.1f}s")

        ts_code = f"{stock_code}.{'SH' if market == 'sh' else 'SZ'}"
        ts_safe = ts_code.replace('.', '_')
        cache_file = os.path.join(cache_dir, f"{ts_safe}.npz")

        # ── Fast peek: read only the first data row to get latest_date ───────
        try:
            peek = pd.read_csv(fp, nrows=1, usecols=['trade_date'],
                               dtype={'trade_date': str})
            latest_date_key = peek['trade_date'].iloc[0]  # e.g. "20260414"
        except Exception:
            skipped += 1
            continue

        # ── Try cache ─────────────────────────────────────────────────────────
        features_raw = None
        cached_close  = None
        cached_sector = None
        cached_date   = None  # "YYYY-MM-DD" for output

        if os.path.exists(cache_file):
            try:
                cached = np.load(cache_file, allow_pickle=True)
                if (str(cached['latest_date']) == latest_date_key
                        and cached['features_raw'].shape == _expected_shape
                        and 'cs_normalized' in cached  # v2 cache flag
                        and bool(cached['cs_normalized'])):
                    features_raw  = cached['features_raw']
                    cached_close  = float(cached['latest_close'])
                    cached_sector = str(cached['sector'])
                    cached_date   = str(cached['latest_date_fmt'])
                    cache_hits   += 1
            except Exception:
                pass  # corrupted cache — will recompute

        # ── Cache miss: full processing pipeline ─────────────────────────────
        if features_raw is None:
            try:
                df = pd.read_csv(fp, nrows=HISTORY, dtype={'trade_date': str})
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                stk_basic = daily_basic_by_ts.get(ts_code)
                df = merge_daily_basic(df, stk_basic,
                                       None if stk_basic is not None else ts_code)
                if _daily_cs_stats_all:
                    df = apply_cs_normalization(df, _daily_cs_stats_all)
                df = merge_market_context(df, market_context)
                df = merge_index_membership(df, index_membership, stock_code)
                df = merge_stk_limit(df, stk_limit_by_ts.get(ts_code), ts_code)
                df = merge_moneyflow(df, moneyflow_by_ts.get(ts_code), ts_code)
                df = calculate_technical_features(df)
                for col in FEATURE_COLUMNS:
                    if col not in df.columns:
                        df[col] = 0.0
                df = df.dropna(subset=FEATURE_COLUMNS)
            except Exception:
                skipped += 1
                continue

            if len(df) < sequence_length:
                skipped += 1
                continue

            features_raw  = df[FEATURE_COLUMNS].values[-sequence_length:]
            features_raw  = np.nan_to_num(features_raw,
                                          nan=0.0, posinf=0.0, neginf=0.0)
            cached_close  = float(df['close'].iloc[-1])
            cached_sector = sector_by_ts.get(ts_code, 'Unknown')
            cached_date   = df['trade_date'].iloc[-1].strftime('%Y-%m-%d')
            cache_misses += 1

            # Persist to cache for next run (cs_normalized=True marks v2 format)
            try:
                np.savez_compressed(
                    cache_file,
                    features_raw    = features_raw,
                    latest_date     = np.array(latest_date_key),
                    latest_date_fmt = np.array(cached_date),
                    latest_close    = np.array(cached_close),
                    sector          = np.array(cached_sector),
                    cs_normalized   = np.array(True),
                )
            except Exception:
                pass  # non-fatal: cache write failure just means no speedup next run

        # Prefer live sector/industry data over cached (in case mapping changed)
        sector_name  = sector_by_ts.get(ts_code, cached_sector or 'Unknown')
        industry_name = industry_by_ts.get(ts_code, 'Unknown')
        sector_id    = sector_to_id.get(sector_name,   sector_to_id['Unknown'])
        industry_id  = industry_to_id.get(industry_name, industry_to_id['Unknown'])

        # Apply scaler (always live — model may have been retrained)
        features = scaler.transform(features_raw)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, -5.0, 5.0)

        seqs.append(features)
        sector_ids.append(sector_id)
        industry_ids.append(industry_id)
        meta_rows.append({
            'stock_code':   stock_code,
            'ts_code':      ts_code,
            'market':       market.upper(),
            'sector':       sector_name,
            'latest_date':  cached_date,
            'latest_close': cached_close,
        })

    elapsed = time.perf_counter() - t0
    print(f"Feature extraction: {elapsed:.1f}s | valid={len(seqs)}, "
          f"skipped={skipped}, cache_hits={cache_hits}, cache_misses={cache_misses}")

    if not seqs:
        print("No valid sequences — check data_dir.")
        return pd.DataFrame()

    # ── Batched GPU inference ─────────────────────────────────────────────────
    seq_arr = np.array(seqs, dtype=np.float32)          # (N, T, F)
    sec_arr = np.array(sector_ids, dtype=np.int64)       # (N,)
    ind_arr = np.array(industry_ids, dtype=np.int64)     # (N,)

    all_probs_list = []
    use_amp        = device.startswith('cuda')

    print(f"\nRunning inference on {len(seqs)} stocks (batch_size={batch_size})...")
    t1 = time.perf_counter()

    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            end   = min(start + batch_size, len(seqs))
            seq_t = torch.from_numpy(seq_arr[start:end]).to(device, non_blocking=True)
            sec_t = torch.from_numpy(sec_arr[start:end]).to(device, non_blocking=True)
            ind_t = torch.from_numpy(ind_arr[start:end]).to(device, non_blocking=True)

            with torch.autocast(device_type=device.split(':')[0],
                                dtype=torch.float16, enabled=use_amp):
                out    = model(seq_t, sec_t, ind_t)
                logits = out[0] if isinstance(out, tuple) else out # (B, H, C)

            lg    = temp_scaler(logits.float())
            probs = torch.softmax(lg, dim=-1).cpu().numpy()       # (B, H, C) or (B, C)
            all_probs_list.append(probs)

    all_probs = np.concatenate(all_probs_list, axis=0)
    elapsed   = time.perf_counter() - t1
    print(f"Inference done: {elapsed:.2f}s for {len(seqs)} stocks")

    # Normalise to (N, H, C) regardless of single/multi-horizon
    if all_probs.ndim == 2:                                        # (N, C) single horizon
        all_probs = all_probs[:, np.newaxis, :]                    # → (N, 1, C)

    # ── Build results DataFrame ───────────────────────────────────────────────
    # 7-class schema: bear=0-2 (<-1%), neutral=3 (-1%–1%), bull=4-6 (>+1%)
    BEAR_CLASSES = list(range(0, 3))
    BULL_CLASSES = list(range(4, NUM_CLASSES))

    # Expected return: weighted average of bucket midpoints.
    # Infinite endpoints are capped at ±15 (outer buckets are ±10%).
    _CAP = 15.0
    MIDPOINTS = np.array([
        (max(lo, -_CAP) + min(hi, _CAP)) / 2.0
        for lo, hi, _ in CHANGE_BUCKETS
    ])

    # Average probabilities across horizons for aggregate metrics
    avg_probs     = all_probs.mean(axis=1)                        # (N, C)
    predicted_cls = avg_probs.argmax(axis=1)
    expected_return = (avg_probs * MIDPOINTS).sum(axis=1)

    rows = []
    for i, meta in enumerate(meta_rows):
        row = dict(meta)
        row['predicted_label']  = class_names[predicted_cls[i]]
        row['confidence']       = float(avg_probs[i, predicted_cls[i]])
        row['expected_return']  = float(expected_return[i])
        row['bull_prob'] = float(avg_probs[i, BULL_CLASSES].sum())
        row['bear_prob'] = float(avg_probs[i, BEAR_CLASSES].sum())
        for j, cn in enumerate(class_names):
            row[f'prob_{cn}'] = float(avg_probs[i, j])
        # Per-horizon columns: prob_<class>_dayN, bull_dayN, bear_dayN
        for h in range(num_horizons):
            hname = get_horizon_name(h)
            hp    = all_probs[i, h]                                 # (C,)
            row[f'bull_{hname}'] = float(hp[BULL_CLASSES].sum())
            row[f'bear_{hname}'] = float(hp[BEAR_CLASSES].sum())
            row[f'pred_{hname}'] = class_names[int(hp.argmax())]
        rows.append(row)

    df_out = pd.DataFrame(rows).sort_values('expected_return', ascending=False)
    df_out = df_out.reset_index(drop=True)

    # ── Console summary ───────────────────────────────────────────────────────
    horizon_bull_cols = [f'bull_{get_horizon_name(h)}' for h in range(num_horizons)]
    cols = (['stock_code', 'sector', 'latest_date', 'latest_close',
             'predicted_label', 'confidence', 'expected_return', 'bull_prob']
            + horizon_bull_cols)
    # Keep only columns that exist (guard against single-horizon models)
    cols = [c for c in cols if c in df_out.columns]

    print(f"\n{'─'*60}")
    print(f"TOP {top_n} BULLISH PREDICTIONS")
    print(f"{'─'*60}")
    print(df_out.head(top_n)[cols].to_string(index=False))

    print(f"\n{'─'*60}")
    print(f"TOP {top_n} BEARISH PREDICTIONS")
    print(f"{'─'*60}")
    print(df_out.tail(top_n)[cols].to_string(index=False))

    if output_csv:
        df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\nFull results saved → {output_csv}")

    return df_out
