"""
Curated catalog mapping every XGBoost model feature → category, Chinese
meaning, and data source. Used by dashboard.build_combined to render the
feature-engineering section.

Every feature in stock_data/models/xgb_pct_chg.features.json must have an
entry here, otherwise build_combined will warn.
"""

from __future__ import annotations

# ─── Group display order + descriptions ───────────────────────────────────────
GROUPS = [
    ('price_action',  '价格行为',         '日内价格形状、收益、动量、波动率'),
    ('volume',        '成交量与换手',     '成交量、换手率、放量缩量'),
    ('technical',     '技术指标',         'RSI / MACD / KDJ / 布林带等经典指标'),
    ('limit_move',    '涨跌停信号',       '涨停/跌停触发、连板、与涨跌停价格距离 (A股特有)'),
    ('pattern',       '形态识别',         'W底 / M顶 等图形识别 (numba 加速检测)'),
    ('moneyflow',     '资金流向',         '小/中/大/特大单买卖差额，主力净流入'),
    ('block_trade',   '大宗交易',         '当日大宗交易笔数、成交占比'),
    ('valuation',     '估值/基本面',      '日级估值因子、市值、季报基本面 (forward-fill)'),
    ('macro',         '宏观/市场环境',    '主流指数 (沪深300/中证500/创业板等) + 全球指数 (滞后1日)'),
    ('csi300_ta',     '沪深300技术形态',  '指数技术因子: BIAS/CCI/DMI/KDJ/RSI/MFI/WR/MACD/PSY/VR'),
    ('cross_section', '横截面特征',       '当日股票横截面排名 / demean / 市场广度 / 离散度'),
    ('sector',        '行业',             '申万一级行业 (label-encoded)'),
    ('calendar',      '日历/时序',        '星期、月份、季度、月初月末等'),
]
GROUP_NAMES = {g[0] for g in GROUPS}

# ─── Per-feature catalog ──────────────────────────────────────────────────────
# (group, meaning_zh, source)
PRICE_DAILY = 'tushare 日线 OHLCV  (stock_data/sh, stock_data/sz)'
DAILY_BASIC = 'tushare daily_basic (stock_data/daily_basic/daily_basic_YYYYMMDD.csv)'
FINA_IND    = 'tushare fina_indicator 季频基本面 (stock_data/fina_indicator/, forward-fill)'
MONEYFLOW   = 'tushare moneyflow 小/中/大/特大单 (stock_data/moneyflow/moneyflow_YYYYMMDD.csv)'
BLOCK_TRADE = 'tushare block_trade 大宗交易 (stock_data/block_trade/block_trade_YYYYMMDD.csv)'
STK_LIMIT   = 'tushare stk_limit 涨跌停价格 (stock_data/stk_limit/stk_limit_YYYYMMDD.csv)'
IDX_DAILY   = 'tushare index_dailybasic (stock_data/index/index_dailybasic/<CODE>.csv)'
IDX_GLOBAL  = '全球指数 (stock_data/index/index_global/<CODE>.csv, 滞后1日防止前视)'
IDX_FACT    = 'tushare idx_factor_pro 沪深300因子 (stock_data/index/idx_factor_pro/000300_SH.csv)'
SECTORS     = '申万行业表 (stock_sectors.csv)'
DERIVED     = '由价格序列即时衍生'
CS_DERIVED  = '横截面计算 (每日按全市场样本)'

CATALOG: dict[str, tuple[str, str, str]] = {
    # ─── Raw OHLCV ──────────────────────────────────────────────────────────
    'open':            ('price_action', '开盘价', PRICE_DAILY),
    'high':            ('price_action', '最高价', PRICE_DAILY),
    'low':             ('price_action', '最低价', PRICE_DAILY),
    'close':           ('price_action', '收盘价', PRICE_DAILY),
    'pre_close':       ('price_action', '昨日收盘价', PRICE_DAILY),
    'change':          ('price_action', '涨跌额 (close − pre_close)', PRICE_DAILY),
    'pct_chg':         ('price_action', '当日涨跌幅 % (close / pre_close − 1)', PRICE_DAILY),
    'vol':             ('volume',       '成交量 (手)', PRICE_DAILY),
    'amount':          ('volume',       '成交额 (千元)', PRICE_DAILY),

    # ─── Intraday shape ─────────────────────────────────────────────────────
    'oc_ratio':        ('price_action', '实体比 (close − open) / pre_close', DERIVED),
    'hl_ratio':        ('price_action', '振幅比 (high − low) / pre_close', DERIVED),
    'uppershadow':     ('price_action', '上影线 / pre_close', DERIVED),
    'lowershadow':     ('price_action', '下影线 / pre_close', DERIVED),
    'overnight_gap':   ('price_action', '隔夜跳空 (open − pre_close) / pre_close', DERIVED),
    'log_ret':         ('price_action', '对数收益 ln(close/pre_close)', DERIVED),

    # ─── Return lags ────────────────────────────────────────────────────────
    'ret_lag_1':  ('price_action', '前1日 pct_chg', DERIVED),
    'ret_lag_2':  ('price_action', '前2日 pct_chg', DERIVED),
    'ret_lag_3':  ('price_action', '前3日 pct_chg', DERIVED),
    'ret_lag_5':  ('price_action', '前5日 pct_chg', DERIVED),
    'ret_lag_10': ('price_action', '前10日 pct_chg', DERIVED),

    # ─── Momentum ───────────────────────────────────────────────────────────
    'momentum_5':  ('price_action', '5日累计对数收益', DERIVED),
    'momentum_10': ('price_action', '10日累计对数收益', DERIVED),
    'momentum_20': ('price_action', '20日累计对数收益', DERIVED),

    # ─── MA ratios + slope ──────────────────────────────────────────────────
    'close_ma_5_ratio':  ('price_action', '收盘价 / MA5',  DERIVED),
    'ma_5_slope':        ('price_action', 'MA5 斜率',     DERIVED),
    'close_ma_10_ratio': ('price_action', '收盘价 / MA10', DERIVED),
    'ma_10_slope':       ('price_action', 'MA10 斜率',    DERIVED),
    'close_ma_20_ratio': ('price_action', '收盘价 / MA20', DERIVED),
    'ma_20_slope':       ('price_action', 'MA20 斜率',    DERIVED),
    'close_ma_60_ratio': ('price_action', '收盘价 / MA60', DERIVED),
    'ma_60_slope':       ('price_action', 'MA60 斜率',    DERIVED),

    # ─── Volatility ────────────────────────────────────────────────────────
    'vol_pct_5':    ('price_action', '5日 pct_chg 标准差',  DERIVED),
    'vol_pct_10':   ('price_action', '10日 pct_chg 标准差', DERIVED),
    'vol_pct_20':   ('price_action', '20日 pct_chg 标准差', DERIVED),
    'parkinson_5':  ('price_action', '5日 Parkinson 波动率 (基于 high/low 对数极差)',  DERIVED),
    'parkinson_10': ('price_action', '10日 Parkinson 波动率', DERIVED),
    'parkinson_20': ('price_action', '20日 Parkinson 波动率', DERIVED),

    # ─── Distance from rolling extrema ─────────────────────────────────────
    'dist_from_high_20': ('price_action', '收盘价相对20日最高价的距离 (越接近1越靠近顶)',  DERIVED),
    'dist_from_low_20':  ('price_action', '收盘价相对20日最低价的距离', DERIVED),
    'dist_from_high_60': ('price_action', '收盘价相对60日最高价的距离', DERIVED),
    'dist_from_low_60':  ('price_action', '收盘价相对60日最低价的距离', DERIVED),

    # ─── Volume / amount ratios ────────────────────────────────────────────
    'vol_ratio_5':    ('volume', '当日成交量 / 5日均量',  DERIVED),
    'vol_ratio_20':   ('volume', '当日成交量 / 20日均量', DERIVED),
    'amt_ratio_5':    ('volume', '当日成交额 / 5日均额',  DERIVED),
    'amt_ratio_20':   ('volume', '当日成交额 / 20日均额', DERIVED),
    'vol_pct_chg':    ('volume', '成交量同比变化 (clip ±10)', DERIVED),
    'amount_pct_chg': ('volume', '成交额同比变化 (clip ±10)', DERIVED),

    # ─── Technical indicators ──────────────────────────────────────────────
    'rsi_14':       ('technical', '14日相对强弱指标 RSI',          DERIVED),
    'macd':         ('technical', 'MACD DIF (12-26 EMA 差)',         DERIVED),
    'macd_signal':  ('technical', 'MACD DEA (9日 EMA of DIF)',       DERIVED),
    'macd_hist':    ('technical', 'MACD 柱状 (DIF − DEA)',           DERIVED),
    'bbpct_20':     ('technical', '布林带百分位 %B (20日)',          DERIVED),
    'atr_14_pct':   ('technical', '14日 ATR / close',                DERIVED),
    'obv_flow_ma5':  ('technical', '5日 OBV 差分均值 (无累积漂移)',  DERIVED),
    'obv_flow_ma20': ('technical', '20日 OBV 差分均值',              DERIVED),

    # ─── Limit moves (A-share) ─────────────────────────────────────────────
    'hit_up_limit':       ('limit_move', '当日是否涨停 (1/0)', DERIVED),
    'hit_down_limit':     ('limit_move', '当日是否跌停 (1/0)', DERIVED),
    'limit_up_streak':    ('limit_move', '连续涨停天数', DERIVED),
    'limit_down_streak':  ('limit_move', '连续跌停天数', DERIVED),
    'limit_up_count_20':  ('limit_move', '过去20日涨停次数', DERIVED),
    'limit_down_count_20':('limit_move', '过去20日跌停次数', DERIVED),
    'up_limit_ratio':     ('limit_move', '收盘价距离涨停价的距离', STK_LIMIT),
    'down_limit_ratio':   ('limit_move', '收盘价距离跌停价的距离', STK_LIMIT),

    # ─── Patterns ──────────────────────────────────────────────────────────
    'w_bottom_10': ('pattern', '10日 W 底形态强度 (0/0.5/1)', DERIVED),
    'w_bottom_20': ('pattern', '20日 W 底形态强度', DERIVED),
    'm_top_10':    ('pattern', '10日 M 顶形态强度', DERIVED),
    'm_top_20':    ('pattern', '20日 M 顶形态强度', DERIVED),

    # ─── Money flow ────────────────────────────────────────────────────────
    'net_sm_amount_ratio':  ('moneyflow', '小单净额 / 当日成交额',    MONEYFLOW),
    'net_md_amount_ratio':  ('moneyflow', '中单净额 / 当日成交额',    MONEYFLOW),
    'net_lg_amount_ratio':  ('moneyflow', '大单净额 / 当日成交额',    MONEYFLOW),
    'net_elg_amount_ratio': ('moneyflow', '特大单净额 / 当日成交额',  MONEYFLOW),
    'net_mf_amount_ratio':  ('moneyflow', '主力净流入 / 当日成交额',  MONEYFLOW),

    # ─── Block trades ──────────────────────────────────────────────────────
    'block_count':        ('block_trade', '当日大宗交易笔数',                BLOCK_TRADE),
    'block_vol_ratio':    ('block_trade', '大宗成交量 / 当日总成交量',       BLOCK_TRADE),
    'block_amount_ratio': ('block_trade', '大宗成交额 / 当日总成交额',       BLOCK_TRADE),

    # ─── Valuation / fundamentals ──────────────────────────────────────────
    'turnover_rate':       ('valuation', '换手率 % (基于总股本)',     DAILY_BASIC),
    'turnover_rate_f':     ('valuation', '换手率 % (基于自由流通股)', DAILY_BASIC),
    'volume_ratio':        ('valuation', '量比 (相对前5日均量)',      DAILY_BASIC),
    'pe_ttm':              ('valuation', 'TTM 市盈率',                DAILY_BASIC),
    'pb':                  ('valuation', '市净率',                    DAILY_BASIC),
    'ps_ttm':              ('valuation', 'TTM 市销率',                DAILY_BASIC),
    'dv_ttm':              ('valuation', 'TTM 股息率',                DAILY_BASIC),
    'log_total_mv':        ('valuation', 'log(1+总市值) (规模因子)',  DAILY_BASIC),
    'log_circ_mv':         ('valuation', 'log(1+流通市值)',           DAILY_BASIC),
    'roe':                 ('valuation', '净资产收益率 (季报)',       FINA_IND),
    'roa':                 ('valuation', '总资产收益率',              FINA_IND),
    'grossprofit_margin':  ('valuation', '毛利率',                    FINA_IND),
    'netprofit_margin':    ('valuation', '净利率',                    FINA_IND),
    'current_ratio':       ('valuation', '流动比率',                  FINA_IND),
    'quick_ratio':         ('valuation', '速动比率',                  FINA_IND),
    'debt_to_assets':      ('valuation', '资产负债率',                FINA_IND),
    'assets_yoy':          ('valuation', '总资产同比增速',            FINA_IND),
    'equity_yoy':          ('valuation', '净资产同比增速',            FINA_IND),
    'op_yoy':              ('valuation', '营业利润同比',              FINA_IND),
    'ebt_yoy':             ('valuation', '利润总额同比',              FINA_IND),
    'eps':                 ('valuation', '每股收益',                  FINA_IND),
    'has_fina_data':       ('valuation', '是否有最新财报披露 (1/0)',  FINA_IND),

    # ─── Macro / market context (CSI300, CSI500, SSE50, SSE, SZSE, ChiNext) ─
    'csi300_turnover':   ('macro', '沪深300换手率',  IDX_DAILY),
    'csi300_pe_ttm':     ('macro', '沪深300 PE-TTM', IDX_DAILY),
    'csi300_pb':         ('macro', '沪深300 PB',     IDX_DAILY),
    'csi500_turnover':   ('macro', '中证500换手率',  IDX_DAILY),
    'csi500_pe_ttm':     ('macro', '中证500 PE-TTM', IDX_DAILY),
    'csi500_pb':         ('macro', '中证500 PB',     IDX_DAILY),
    'csi500_pct_chg':    ('macro', '中证500涨跌幅',  IDX_DAILY),
    'sse50_turnover':    ('macro', '上证50换手率',   IDX_DAILY),
    'sse50_pe_ttm':      ('macro', '上证50 PE-TTM',  IDX_DAILY),
    'sse50_pb':          ('macro', '上证50 PB',      IDX_DAILY),
    'sse50_pct_chg':     ('macro', '上证50涨跌幅',   IDX_DAILY),
    'sse_turnover':      ('macro', '上证综指换手率', IDX_DAILY),
    'sse_pe_ttm':        ('macro', '上证综指 PE-TTM',IDX_DAILY),
    'sse_pb':            ('macro', '上证综指 PB',    IDX_DAILY),
    'sse_pct_chg':       ('macro', '上证综指涨跌幅', IDX_DAILY),
    'szse_turnover':     ('macro', '深证成指换手率', IDX_DAILY),
    'szse_pe_ttm':       ('macro', '深证成指 PE-TTM',IDX_DAILY),
    'szse_pb':           ('macro', '深证成指 PB',    IDX_DAILY),
    'chinext_turnover':  ('macro', '创业板指换手率', IDX_DAILY),
    'chinext_pe_ttm':    ('macro', '创业板指 PE-TTM',IDX_DAILY),
    'chinext_pb':        ('macro', '创业板指 PB',    IDX_DAILY),

    # ─── Global indices (lagged 1 day) ─────────────────────────────────────
    'spx_pct_chg_lag1':   ('macro', '标普500 涨跌幅 (滞后1日)',  IDX_GLOBAL),
    'spx_vol_ratio_lag1': ('macro', '标普500 量比 (滞后1日)',    IDX_GLOBAL),
    'dji_pct_chg_lag1':   ('macro', '道琼斯 涨跌幅 (滞后1日)',   IDX_GLOBAL),
    'dji_vol_ratio_lag1': ('macro', '道琼斯 量比 (滞后1日)',     IDX_GLOBAL),
    'ixic_pct_chg_lag1':  ('macro', '纳斯达克 涨跌幅 (滞后1日)', IDX_GLOBAL),
    'ixic_vol_ratio_lag1':('macro', '纳斯达克 量比 (滞后1日)',   IDX_GLOBAL),
    'hsi_pct_chg_lag1':   ('macro', '恒生指数 涨跌幅 (滞后1日)', IDX_GLOBAL),
    'hsi_vol_ratio_lag1': ('macro', '恒生指数 量比 (滞后1日)',   IDX_GLOBAL),
    'n225_pct_chg_lag1':  ('macro', '日经225 涨跌幅 (滞后1日)',  IDX_GLOBAL),
    'n225_vol_ratio_lag1':('macro', '日经225 量比 (滞后1日)',    IDX_GLOBAL),
    'ftse_pct_chg_lag1':  ('macro', '富时100 涨跌幅 (滞后1日)',  IDX_GLOBAL),
    'ftse_vol_ratio_lag1':('macro', '富时100 量比 (滞后1日)',    IDX_GLOBAL),

    # ─── CSI300 idx_factor_pro TA factors ─────────────────────────────────
    'idx_bias1':    ('csi300_ta', '沪深300 BIAS1 (短期偏离均线)', IDX_FACT),
    'idx_bias2':    ('csi300_ta', '沪深300 BIAS2 (中期偏离)',     IDX_FACT),
    'idx_bias3':    ('csi300_ta', '沪深300 BIAS3 (长期偏离)',     IDX_FACT),
    'idx_cci':      ('csi300_ta', '沪深300 CCI 顺势指标',         IDX_FACT),
    'idx_dmi_adx':  ('csi300_ta', '沪深300 DMI ADX 趋势强度',     IDX_FACT),
    'idx_dmi_pdi':  ('csi300_ta', '沪深300 DMI +DI 上行动能',     IDX_FACT),
    'idx_dmi_mdi':  ('csi300_ta', '沪深300 DMI −DI 下行动能',     IDX_FACT),
    'idx_kdj_k':    ('csi300_ta', '沪深300 KDJ K值',              IDX_FACT),
    'idx_kdj_d':    ('csi300_ta', '沪深300 KDJ D值',              IDX_FACT),
    'idx_rsi_6':    ('csi300_ta', '沪深300 RSI(6)',               IDX_FACT),
    'idx_rsi_12':   ('csi300_ta', '沪深300 RSI(12)',              IDX_FACT),
    'idx_rsi_24':   ('csi300_ta', '沪深300 RSI(24)',              IDX_FACT),
    'idx_mfi':      ('csi300_ta', '沪深300 MFI 资金流量指标',     IDX_FACT),
    'idx_wr':       ('csi300_ta', '沪深300 Williams %R',          IDX_FACT),
    'idx_macd_dif': ('csi300_ta', '沪深300 MACD DIF',             IDX_FACT),
    'idx_macd_dea': ('csi300_ta', '沪深300 MACD DEA',             IDX_FACT),
    'idx_psy':      ('csi300_ta', '沪深300 PSY 心理线',           IDX_FACT),
    'idx_vr':       ('csi300_ta', '沪深300 VR 容量比率',          IDX_FACT),

    # ─── Cross-sectional ───────────────────────────────────────────────────
    'cs_rank_pct_chg':            ('cross_section', '当日 pct_chg 在全市场的分位数 (0–1)', CS_DERIVED),
    'cs_rank_turnover_rate_f':    ('cross_section', '换手率分位数',           CS_DERIVED),
    'cs_rank_vol_ratio_20':       ('cross_section', '20日量比分位数',         CS_DERIVED),
    'cs_rank_amt_ratio_20':       ('cross_section', '20日额比分位数',         CS_DERIVED),
    'cs_rank_rsi_14':             ('cross_section', 'RSI(14)分位数',          CS_DERIVED),
    'cs_rank_momentum_20':        ('cross_section', '20日动量分位数',         CS_DERIVED),
    'cs_rank_vol_pct_20':         ('cross_section', '20日波动率分位数',       CS_DERIVED),
    'cs_rank_dist_from_high_20':  ('cross_section', '距20日高点分位数',       CS_DERIVED),
    'cs_rank_net_mf_amount_ratio':('cross_section', '主力净流入分位数',       CS_DERIVED),
    'cs_rank_up_limit_ratio':     ('cross_section', '距涨停距离分位数',       CS_DERIVED),
    'cs_rank_overnight_gap':      ('cross_section', '隔夜跳空分位数',         CS_DERIVED),
    'cs_rank_log_ret':            ('cross_section', '对数收益分位数',         CS_DERIVED),
    'cs_demean_pct_chg':          ('cross_section', '当日 pct_chg − 全市场均值 (去市场)', CS_DERIVED),
    'cs_demean_turnover_rate_f':  ('cross_section', '换手率 − 全市场均值',                  CS_DERIVED),
    'cs_demean_rsi_14':           ('cross_section', 'RSI(14) − 全市场均值',                 CS_DERIVED),
    'cs_demean_momentum_20':      ('cross_section', '20日动量 − 全市场均值',                CS_DERIVED),
    'cs_market_breadth':          ('cross_section', '市场宽度 (当日上涨股票占比)',          CS_DERIVED),
    'cs_daily_dispersion':        ('cross_section', '当日全市场涨跌幅离散度 (横截面 std)',  CS_DERIVED),

    # ─── Sector ─────────────────────────────────────────────────────────────
    'sector_id': ('sector', '申万一级行业 (label-encoded; 0 表示未知)', SECTORS),

    # ─── Calendar ───────────────────────────────────────────────────────────
    'dow':            ('calendar', '星期几 (0–6)',           DERIVED),
    'month':          ('calendar', '月份 (1–12)',            DERIVED),
    'day_of_month':   ('calendar', '当月第几日 (1–31)',      DERIVED),
    'quarter':        ('calendar', '季度 (1–4)',             DERIVED),
    'is_month_end':   ('calendar', '是否月末 (1/0)',         DERIVED),
    'is_month_start': ('calendar', '是否月初 (1/0)',         DERIVED),
    'is_quarter_end': ('calendar', '是否季末 (1/0)',         DERIVED),
}


def lookup(name: str) -> dict:
    if name in CATALOG:
        g, m, s = CATALOG[name]
        return {'name': name, 'group': g, 'meaning': m, 'source': s}
    return {'name': name, 'group': 'unknown', 'meaning': '(无说明)', 'source': '(未知)'}


def coverage(features: list) -> tuple[int, list]:
    missing = [f for f in features if f not in CATALOG]
    return len(features) - len(missing), missing
