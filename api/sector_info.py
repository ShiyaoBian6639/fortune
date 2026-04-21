"""
Download enriched sector / static stock attributes from Tushare Pro.

Produces stock_data/stock_sectors.csv with columns:
    ts_code, symbol, name, area, market, list_date, is_hs,
    sw_l1_code, sw_l1_name, sw_l2_code, sw_l2_name

Usage:
    python -m api.sector_info               # download and save
    from api.sector_info import run; run()  # scripted
"""

import time
from pathlib import Path

import pandas as pd
import tushare as ts

TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR      = Path('./stock_data')
OUTPUT_FILE   = DATA_DIR / 'stock_sectors.csv'
CALL_INTERVAL = 0.25   # conservative rate for classify / member calls
API_TIMEOUT   = 120    # seconds


def init_tushare():
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)
    try:
        pro._DataApi__timeout = API_TIMEOUT
    except AttributeError:
        pass
    return pro


def _retry(fn, *args, label='call', max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait = 10 * (2 ** attempt)
            if attempt < max_retries - 1:
                print(f"  [{label}] attempt {attempt+1}/{max_retries}: {type(e).__name__}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"{label} failed: {e}") from e


def get_stock_basic(pro) -> pd.DataFrame:
    """Fetch ts_code, symbol, name, area, market, list_date, is_hs for all A-shares."""
    fields = 'ts_code,symbol,name,area,market,list_date,is_hs'
    df_L = _retry(pro.stock_basic, exchange='', list_status='L', fields=fields, label='stock_basic(L)')
    df_P = _retry(pro.stock_basic, exchange='', list_status='P', fields=fields, label='stock_basic(P)')
    df_D = _retry(pro.stock_basic, exchange='', list_status='D', fields=fields, label='stock_basic(D)')
    df   = pd.concat([df_L, df_P, df_D], ignore_index=True)
    df['exchange'] = df['ts_code'].str.split('.').str[1]
    df = df[df['exchange'].isin(['SH', 'SZ'])].copy()
    df['list_date'] = pd.to_datetime(df['list_date'].astype(str), errors='coerce')
    print(f"  stock_basic: {len(df)} stocks (SH + SZ)")
    return df


def get_sw_classification(pro) -> pd.DataFrame:
    """
    Fetch Shenwan (申万) Level-1 and Level-2 industry classification.

    Returns DataFrame with columns: ts_code, sw_l1_code, sw_l1_name, sw_l2_code, sw_l2_name
    """
    # ── Get SW L1 and L2 index codes ─────────────────────────────────────────
    print("  Fetching SW2021 Level-1 industry list...")
    l1_df = _retry(pro.index_classify, level='L1', src='SW2021', label='index_classify(L1)')
    time.sleep(CALL_INTERVAL)

    print("  Fetching SW2021 Level-2 industry list...")
    l2_df = _retry(pro.index_classify, level='L2', src='SW2021', label='index_classify(L2)')
    time.sleep(CALL_INTERVAL)

    print(f"  SW L1: {len(l1_df)} sectors  |  SW L2: {len(l2_df)} sub-industries")

    # ── For each L1, get member stocks ────────────────────────────────────────
    l1_members = []
    for _, row in l1_df.iterrows():
        code = row['index_code']
        name = row['industry_name']
        try:
            members = _retry(pro.index_member, index_code=code, label=f'index_member(L1/{code})')
            if members is not None and not members.empty:
                members = members[members['is_new'] == 'Y'].copy()
                members['sw_l1_code'] = code
                members['sw_l1_name'] = name
                l1_members.append(members[['con_code', 'sw_l1_code', 'sw_l1_name']])
        except Exception as e:
            print(f"    [warn] L1 {code} ({name}): {e}")
        time.sleep(CALL_INTERVAL)

    # ── For each L2, get member stocks ────────────────────────────────────────
    l2_members = []
    for _, row in l2_df.iterrows():
        code = row['index_code']
        name = row['industry_name']
        try:
            members = _retry(pro.index_member, index_code=code, label=f'index_member(L2/{code})')
            if members is not None and not members.empty:
                members = members[members['is_new'] == 'Y'].copy()
                members['sw_l2_code'] = code
                members['sw_l2_name'] = name
                l2_members.append(members[['con_code', 'sw_l2_code', 'sw_l2_name']])
        except Exception as e:
            print(f"    [warn] L2 {code} ({name}): {e}")
        time.sleep(CALL_INTERVAL)

    # ── Combine ───────────────────────────────────────────────────────────────
    if not l1_members:
        print("  [warn] No L1 membership data returned")
        return pd.DataFrame(columns=['ts_code', 'sw_l1_code', 'sw_l1_name',
                                     'sw_l2_code', 'sw_l2_name'])

    l1_map = (pd.concat(l1_members, ignore_index=True)
              .rename(columns={'con_code': 'ts_code'})
              .drop_duplicates('ts_code'))

    if l2_members:
        l2_map = (pd.concat(l2_members, ignore_index=True)
                  .rename(columns={'con_code': 'ts_code'})
                  .drop_duplicates('ts_code'))
        sw_map = l1_map.merge(l2_map, on='ts_code', how='left')
    else:
        sw_map = l1_map.copy()
        sw_map['sw_l2_code'] = ''
        sw_map['sw_l2_name'] = ''

    print(f"  SW classification mapped: {len(sw_map)} stocks")
    return sw_map


def build_sector_csv(pro) -> pd.DataFrame:
    """Build the enriched stock_sectors.csv and save to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Building enriched sector data...")
    basic  = get_stock_basic(pro)
    sw_map = get_sw_classification(pro)

    # Merge
    df = basic.merge(sw_map, on='ts_code', how='left')

    # Fill missing SW classification with 'Unknown'
    for col in ('sw_l1_code', 'sw_l1_name', 'sw_l2_code', 'sw_l2_name'):
        df[col] = df[col].fillna('Unknown')

    # Normalise market name
    df['market'] = df['market'].fillna('主板').str.strip()

    # Add derived columns used by downstream code
    # 'sector' = coarse 9-category mapping (backward-compat with existing pipelines)
    sector_map = {
        '金融':     'Finance',
        '地产':     'Real Estate',
        '消费':     'Consumer',
        '医药':     'Healthcare',
        '材料':     'Materials',
        '能源':     'Energy',
        '工业':     'Industrials',
        '科技':     'Technology',
        '信息技术': 'Technology',
        '通信':     'Technology',
        '公用事业': 'Utilities',
        '交通运输': 'Industrials',
    }
    def _to_sector(sw_l1_name):
        for kw, sector in sector_map.items():
            if kw in str(sw_l1_name):
                return sector
        return 'Other'

    df['sector']   = df['sw_l1_name'].apply(_to_sector)
    df['industry'] = df['sw_l2_name']   # finer-grained, used by deeptime

    # Select and reorder final columns
    cols = ['ts_code', 'symbol', 'name', 'area', 'market', 'list_date',
            'is_hs', 'sw_l1_code', 'sw_l1_name', 'sw_l2_code', 'sw_l2_name',
            'sector', 'industry']
    df = df[cols].sort_values('ts_code').reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} stocks to {OUTPUT_FILE}")

    # Summary
    print(f"  SW L1 sectors: {df['sw_l1_name'].nunique()} unique")
    print(f"  SW L2 sub-industries: {df['sw_l2_name'].nunique()} unique")
    print(f"  Regions (area): {df['area'].nunique()} unique")
    print(f"  Market types: {df['market'].value_counts().to_dict()}")
    return df


def run():
    """Download and save enriched sector/static information."""
    pro = init_tushare()
    build_sector_csv(pro)


if __name__ == '__main__':
    run()
