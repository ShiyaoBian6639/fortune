"""
Extended Feature Engineering for DeepTime Model

Integrates new data sources:
1. Financial Statements (income, balancesheet, cashflow) - quarterly fundamentals
2. Financial Extras (forecast, express, dividend) - event-driven signals
3. Limit/Dragon-Tiger Data - market sentiment and momentum
4. THS Index Data - sector/concept membership and momentum
5. Chip Distribution - holder structure analysis

Features are categorized into:
- Static variates (GAT): time-invariant stock characteristics
- Historical time series (TFT): daily/quarterly evolving features
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Static Variates (GAT) ────────────────────────────────────────────────────

# These features represent time-invariant (or slowly changing) stock characteristics
# used in Graph Attention Networks for inter-stock relationship modeling.

STATIC_FINANCIAL_FEATURES = [
    # From balancesheet - capital structure
    'debt_to_equity',        # 总负债/股东权益
    'current_ratio_static',  # 流动资产/流动负债
    'asset_turnover',        # 营收/总资产
    # From income - profitability structure
    'gross_margin_static',   # 毛利率
    'operating_margin',      # 营业利润率
    'net_margin_static',     # 净利率
    # From dividend - dividend characteristics
    'has_dividend',          # 是否有分红 (0/1)
    'avg_div_yield',         # 平均股息率
    'div_consistency',       # 分红一致性 (连续分红年数/总年数)
]

STATIC_MEMBERSHIP_FEATURES = [
    # From THS index membership
    'n_ths_concepts',        # 所属同花顺概念数量
    'n_ths_industries',      # 所属同花顺行业数量
    'is_hot_concept',        # 是否属于热门概念 (龙头股)
]


def compute_static_financial_features(
    ts_code: str,
    data_dir: str,
) -> Dict[str, float]:
    """
    Compute static financial features from financial statements.

    Returns dict of feature_name -> value (use as embedding or continuous input).
    """
    result = {k: 0.0 for k in STATIC_FINANCIAL_FEATURES}
    bare = str(ts_code).split('.')[0]
    suffix = '_SH' if ts_code.endswith('.SH') or bare.startswith('6') else '_SZ'

    # Load balancesheet
    bs_path = Path(data_dir) / 'fina_statements' / 'balancesheet' / f"{bare}{suffix}.csv"
    if bs_path.exists():
        try:
            bs = pd.read_csv(bs_path)
            if len(bs) >= 2:
                # Use median of recent reports to reduce noise
                recent = bs.sort_values('end_date', ascending=False).head(8)
                total_liab = recent['total_liab'].median() if 'total_liab' in recent.columns else 0
                total_equity = recent['total_hldr_eqy_exc_min_int'].median() if 'total_hldr_eqy_exc_min_int' in recent.columns else 0
                cur_assets = recent['total_cur_assets'].median() if 'total_cur_assets' in recent.columns else 0
                cur_liab = recent['total_cur_liab'].median() if 'total_cur_liab' in recent.columns else 0
                total_assets = recent['total_assets'].median() if 'total_assets' in recent.columns else 0

                if total_equity > 0:
                    result['debt_to_equity'] = np.clip(total_liab / total_equity, 0, 10)
                if cur_liab > 0:
                    result['current_ratio_static'] = np.clip(cur_assets / cur_liab, 0, 5)
        except Exception:
            pass

    # Load income statement
    inc_path = Path(data_dir) / 'fina_statements' / 'income' / f"{bare}{suffix}.csv"
    if inc_path.exists():
        try:
            inc = pd.read_csv(inc_path)
            if len(inc) >= 2:
                recent = inc.sort_values('end_date', ascending=False).head(8)
                revenue = recent['revenue'].median() if 'revenue' in recent.columns else 0
                oper_cost = recent['oper_cost'].median() if 'oper_cost' in recent.columns else 0
                oper_profit = recent['operate_profit'].median() if 'operate_profit' in recent.columns else 0
                net_income = recent['n_income'].median() if 'n_income' in recent.columns else 0

                if revenue > 0:
                    result['gross_margin_static'] = np.clip((revenue - oper_cost) / revenue, -1, 1)
                    result['operating_margin'] = np.clip(oper_profit / revenue, -1, 1)
                    result['net_margin_static'] = np.clip(net_income / revenue, -1, 1)

                # Load total_assets from balancesheet for turnover
                if bs_path.exists():
                    bs = pd.read_csv(bs_path)
                    if 'total_assets' in bs.columns:
                        total_assets = bs.sort_values('end_date', ascending=False)['total_assets'].iloc[0]
                        if total_assets > 0:
                            result['asset_turnover'] = np.clip(revenue / total_assets, 0, 5)
        except Exception:
            pass

    # Load dividend data
    div_path = Path(data_dir) / 'fina_extras' / 'dividend' / f"{bare}{suffix}.csv"
    if div_path.exists():
        try:
            div = pd.read_csv(div_path)
            if len(div) >= 1:
                result['has_dividend'] = 1.0

                # Count unique dividend years
                if 'end_date' in div.columns:
                    div['year'] = div['end_date'].astype(str).str[:4].astype(int)
                    total_years = div['year'].max() - div['year'].min() + 1 if len(div) > 1 else 1
                    div_years = div['year'].nunique()
                    result['div_consistency'] = div_years / max(total_years, 1)

                # Average dividend yield approximation
                if 'cash_div' in div.columns:
                    avg_cash_div = div['cash_div'].mean()
                    result['avg_div_yield'] = np.clip(avg_cash_div / 100, 0, 0.2)  # Normalize
        except Exception:
            pass

    return result


def compute_static_membership_features(
    ts_code: str,
    data_dir: str,
) -> Dict[str, float]:
    """
    Compute static THS concept/industry membership features.
    """
    result = {k: 0.0 for k in STATIC_MEMBERSHIP_FEATURES}
    bare = str(ts_code).split('.')[0]

    ths_member_dir = Path(data_dir) / 'ths_index' / 'ths_member'
    if not ths_member_dir.exists():
        return result

    n_concepts = 0
    n_industries = 0

    # Scan all THS member files to find this stock
    try:
        for fp in ths_member_dir.glob('*.csv'):
            df = pd.read_csv(fp, usecols=['code'] if 'code' in pd.read_csv(fp, nrows=0).columns else ['ts_code'])
            col = 'code' if 'code' in df.columns else 'ts_code'

            # Check if stock is a member
            codes = df[col].astype(str).str.split('.').str[0].tolist()
            if bare in codes:
                # Determine if concept (N) or industry (I) based on index code
                index_code = fp.stem
                if 'N' in index_code or '8' in index_code[:3]:  # Concept indices often start with 8
                    n_concepts += 1
                else:
                    n_industries += 1
    except Exception:
        pass

    result['n_ths_concepts'] = min(n_concepts, 20)  # Cap to prevent extreme values
    result['n_ths_industries'] = min(n_industries, 10)
    result['is_hot_concept'] = 1.0 if n_concepts >= 5 else 0.0

    return result


# ─── Historical Time Series (TFT) ─────────────────────────────────────────────

# Daily/event-driven features for the Temporal Fusion Transformer

FORECAST_FEATURES = [
    'has_forecast',        # 是否有业绩预告
    'forecast_direction',  # 预告方向: -1=预减, 0=不确定, 1=预增
    'forecast_magnitude',  # 预告变化幅度 (p_change_min + p_change_max) / 2
]

EXPRESS_FEATURES = [
    'has_express',         # 是否有业绩快报
    'express_growth',      # 业绩快报营收同比增长
    'express_profit_yoy',  # 业绩快报利润同比增长
]

LIMIT_FEATURES = [
    'is_limit_up',         # 涨停 (from limit_list_d)
    'is_limit_down',       # 跌停
    'limit_times',         # 连板次数
    'on_dragon_tiger',     # 是否上龙虎榜
    'dragon_tiger_net',    # 龙虎榜净买入/总成交
]

CHIP_FEATURES = [
    'winner_rate',         # 获利比例
    'cost_concentration',  # 成本集中度 (cost_85pct - cost_15pct) / weight_avg
    'chip_divergence',     # 筹码分歧度
]

THS_MOMENTUM_FEATURES = [
    'ths_concept_momentum',  # 所属概念近期涨幅
    'ths_industry_momentum', # 所属行业近期涨幅
]


def load_forecast_data(
    data_dir: str,
    ts_codes: List[str],
) -> Dict[str, pd.DataFrame]:
    """Load earnings forecast data for all stocks."""
    result = {}
    forecast_dir = Path(data_dir) / 'fina_extras' / 'forecast'
    if not forecast_dir.exists():
        return result

    for ts_code in ts_codes:
        bare = str(ts_code).split('.')[0]
        for suffix in ['_SZ', '_SH', '']:
            fp = forecast_dir / f"{bare}{suffix}.csv"
            if fp.exists():
                try:
                    df = pd.read_csv(fp)
                    df['ann_date'] = pd.to_datetime(df['ann_date'].astype(str), errors='coerce')
                    result[bare] = df.dropna(subset=['ann_date']).sort_values('ann_date')
                except Exception:
                    pass
                break

    return result


def load_express_data(
    data_dir: str,
    ts_codes: List[str],
) -> Dict[str, pd.DataFrame]:
    """Load express earnings data for all stocks."""
    result = {}
    express_dir = Path(data_dir) / 'fina_extras' / 'express'
    if not express_dir.exists():
        return result

    for ts_code in ts_codes:
        bare = str(ts_code).split('.')[0]
        for suffix in ['_SZ', '_SH', '']:
            fp = express_dir / f"{bare}{suffix}.csv"
            if fp.exists():
                try:
                    df = pd.read_csv(fp)
                    df['ann_date'] = pd.to_datetime(df['ann_date'].astype(str), errors='coerce')
                    result[bare] = df.dropna(subset=['ann_date']).sort_values('ann_date')
                except Exception:
                    pass
                break

    return result


def load_limit_data_by_stock(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load limit list data aggregated by stock."""
    result = {}
    limit_dir = Path(data_dir) / 'limit_data' / 'limit_list_d_by_stock'
    if not limit_dir.exists():
        return result

    for fp in limit_dir.glob('*.csv'):
        bare = fp.stem
        try:
            df = pd.read_csv(fp)
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), errors='coerce')
                result[bare] = df.sort_values('trade_date')
        except Exception:
            pass

    return result


def load_dragon_tiger_by_stock(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load dragon-tiger list data aggregated by stock."""
    result = {}
    dt_dir = Path(data_dir) / 'limit_data' / 'top_list_by_stock'
    if not dt_dir.exists():
        return result

    for fp in dt_dir.glob('*.csv'):
        bare = fp.stem
        try:
            df = pd.read_csv(fp)
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), errors='coerce')
                result[bare] = df.sort_values('trade_date')
        except Exception:
            pass

    return result


def load_chip_perf_by_stock(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load chip distribution performance data by stock."""
    result = {}
    chip_dir = Path(data_dir) / 'chip_data' / 'cyq_perf'
    if not chip_dir.exists():
        return result

    for fp in chip_dir.glob('[!_]*.csv'):
        bare = fp.stem.replace('_SH', '').replace('_SZ', '')
        try:
            df = pd.read_csv(fp)
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), errors='coerce')
                result[bare] = df.sort_values('trade_date')
        except Exception:
            pass

    return result


def merge_forecast_features(
    stock_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge forecast features into stock dataframe."""
    # Initialize all columns at once to avoid fragmentation
    stock_df = stock_df.assign(**{col: 0.0 for col in FORECAST_FEATURES})

    if forecast_df is None or len(forecast_df) == 0:
        return stock_df

    if not pd.api.types.is_datetime64_any_dtype(stock_df['trade_date']):
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'].astype(str))

    # For each trading day, find the most recent forecast before that day
    fc_sorted = forecast_df.sort_values('ann_date').reset_index(drop=True)
    ann_dates = fc_sorted['ann_date'].values
    trade_dates = stock_df['trade_date'].values

    # Find last forecast index for each trading day
    idx = np.searchsorted(ann_dates, trade_dates, side='right') - 1

    has_forecast = np.zeros(len(stock_df), dtype='float32')
    forecast_direction = np.zeros(len(stock_df), dtype='float32')
    forecast_magnitude = np.zeros(len(stock_df), dtype='float32')

    for i, fi in enumerate(idx):
        if fi >= 0:
            row = fc_sorted.iloc[fi]
            # Only use forecast if it's within 90 days
            days_since = (trade_dates[i] - ann_dates[fi]).astype('timedelta64[D]').astype(int)
            if days_since <= 90:
                has_forecast[i] = 1.0

                # Direction from 'type' column
                fc_type = str(row.get('type', '')).lower()
                if any(k in fc_type for k in ['预增', '略增', '续盈', '扭亏']):
                    forecast_direction[i] = 1.0
                elif any(k in fc_type for k in ['预减', '略减', '续亏', '首亏']):
                    forecast_direction[i] = -1.0

                # Magnitude
                p_min = float(row.get('p_change_min', 0) or 0)
                p_max = float(row.get('p_change_max', 0) or 0)
                forecast_magnitude[i] = np.clip((p_min + p_max) / 200, -1, 1)  # Normalize to [-1, 1]

    stock_df['has_forecast'] = has_forecast
    stock_df['forecast_direction'] = forecast_direction
    stock_df['forecast_magnitude'] = forecast_magnitude

    return stock_df


def merge_express_features(
    stock_df: pd.DataFrame,
    express_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge express earnings features into stock dataframe."""
    # Initialize all columns at once to avoid fragmentation
    stock_df = stock_df.assign(**{col: 0.0 for col in EXPRESS_FEATURES})

    if express_df is None or len(express_df) == 0:
        return stock_df

    if not pd.api.types.is_datetime64_any_dtype(stock_df['trade_date']):
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'].astype(str))

    exp_sorted = express_df.sort_values('ann_date').reset_index(drop=True)
    ann_dates = exp_sorted['ann_date'].values
    trade_dates = stock_df['trade_date'].values

    idx = np.searchsorted(ann_dates, trade_dates, side='right') - 1

    has_express = np.zeros(len(stock_df), dtype='float32')
    express_growth = np.zeros(len(stock_df), dtype='float32')
    express_profit = np.zeros(len(stock_df), dtype='float32')

    for i, ei in enumerate(idx):
        if ei >= 0:
            row = exp_sorted.iloc[ei]
            days_since = (trade_dates[i] - ann_dates[ei]).astype('timedelta64[D]').astype(int)
            if days_since <= 60:
                has_express[i] = 1.0
                yoy_sales = float(row.get('yoy_sales', 0) or 0)
                yoy_np = float(row.get('yoy_net_profit', 0) or 0)
                express_growth[i] = np.clip(yoy_sales / 100, -1, 1)
                express_profit[i] = np.clip(yoy_np / 100, -1, 1)

    stock_df['has_express'] = has_express
    stock_df['express_growth'] = express_growth
    stock_df['express_profit_yoy'] = express_profit

    return stock_df


def merge_limit_features(
    stock_df: pd.DataFrame,
    limit_df: Optional[pd.DataFrame],
    dragon_tiger_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge limit and dragon-tiger features into stock dataframe."""
    # .copy() defragments the DataFrame (accumulated from many prior merges)
    # before adding new columns, eliminating the PerformanceWarning.
    stock_df = stock_df.copy()
    stock_df = stock_df.assign(**{col: 0.0 for col in LIMIT_FEATURES})

    if not pd.api.types.is_datetime64_any_dtype(stock_df['trade_date']):
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'].astype(str))

    # Limit data
    if limit_df is not None and len(limit_df) > 0:
        if not pd.api.types.is_datetime64_any_dtype(limit_df['trade_date']):
            limit_df['trade_date'] = pd.to_datetime(limit_df['trade_date'].astype(str))

        merged = stock_df[['trade_date']].merge(
            limit_df[['trade_date', 'limit', 'limit_times']].drop_duplicates('trade_date'),
            on='trade_date', how='left'
        )
        stock_df['is_limit_up'] = (merged['limit'] == 'U').astype('float32').values
        stock_df['is_limit_down'] = (merged['limit'] == 'D').astype('float32').values
        stock_df['limit_times'] = merged['limit_times'].fillna(0).clip(0, 10).astype('float32').values

    # Dragon-tiger data
    if dragon_tiger_df is not None and len(dragon_tiger_df) > 0:
        if not pd.api.types.is_datetime64_any_dtype(dragon_tiger_df['trade_date']):
            dragon_tiger_df['trade_date'] = pd.to_datetime(dragon_tiger_df['trade_date'].astype(str))

        dt_dates = set(dragon_tiger_df['trade_date'].dt.strftime('%Y%m%d'))
        stock_df['on_dragon_tiger'] = stock_df['trade_date'].dt.strftime('%Y%m%d').isin(dt_dates).astype('float32')

        if 'net_amount' in dragon_tiger_df.columns and 'amount' in dragon_tiger_df.columns:
            dt_agg = dragon_tiger_df.groupby('trade_date').agg({
                'net_amount': 'sum',
                'amount': 'sum'
            }).reset_index()
            dt_agg['net_ratio'] = dt_agg['net_amount'] / (dt_agg['amount'] + 1e-8)
            merged = stock_df[['trade_date']].merge(
                dt_agg[['trade_date', 'net_ratio']], on='trade_date', how='left'
            )
            stock_df['dragon_tiger_net'] = merged['net_ratio'].fillna(0).clip(-1, 1).astype('float32').values

    return stock_df


def merge_chip_features(
    stock_df: pd.DataFrame,
    chip_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge chip distribution features into stock dataframe."""
    stock_df = stock_df.copy()   # defragment before adding columns
    stock_df = stock_df.assign(**{col: 0.0 for col in CHIP_FEATURES})

    if chip_df is None or len(chip_df) == 0:
        return stock_df

    if not pd.api.types.is_datetime64_any_dtype(stock_df['trade_date']):
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'].astype(str))

    if not pd.api.types.is_datetime64_any_dtype(chip_df['trade_date']):
        chip_df['trade_date'] = pd.to_datetime(chip_df['trade_date'].astype(str))

    chip_cols = ['trade_date', 'winner_rate', 'cost_15pct', 'cost_85pct', 'weight_avg']
    available = [c for c in chip_cols if c in chip_df.columns]
    if 'trade_date' not in available:
        return stock_df

    merged = stock_df[['trade_date']].merge(
        chip_df[available].drop_duplicates('trade_date'),
        on='trade_date', how='left'
    )

    if 'winner_rate' in merged.columns:
        stock_df['winner_rate'] = merged['winner_rate'].fillna(50).clip(0, 100).div(100).astype('float32').values

    if all(c in merged.columns for c in ['cost_15pct', 'cost_85pct', 'weight_avg']):
        spread = merged['cost_85pct'] - merged['cost_15pct']
        avg = merged['weight_avg'].replace(0, np.nan)
        stock_df['cost_concentration'] = (spread / avg).fillna(0).clip(0, 2).astype('float32').values

    return stock_df


# ─── Feature Column Lists ─────────────────────────────────────────────────────

# All new features to be added to the model

EXTENDED_STATIC_FEATURES = STATIC_FINANCIAL_FEATURES + STATIC_MEMBERSHIP_FEATURES

EXTENDED_TIME_SERIES_FEATURES = (
    FORECAST_FEATURES +
    EXPRESS_FEATURES +
    LIMIT_FEATURES +
    CHIP_FEATURES
)

# Total new features
ALL_EXTENDED_FEATURES = EXTENDED_STATIC_FEATURES + EXTENDED_TIME_SERIES_FEATURES
