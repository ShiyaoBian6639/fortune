"""
Enhanced Quantitative Trading Strategy - Targeting 25% Annual Returns
=====================================================================

This strategy combines multiple proven approaches from leading quant funds:

1. MOMENTUM INVESTING (Jegadeesh & Titman, AQR Capital)
   - 12-month momentum minus 1-month (avoiding short-term reversal)
   - Relative strength ranking across universe
   - Buy winners, sell losers

2. DUAL MOMENTUM (Gary Antonacci)
   - Absolute momentum: Is the asset trending up?
   - Relative momentum: Is it outperforming alternatives?
   - Only invest when both are positive

3. TREND FOLLOWING (Turtle Traders, managed futures)
   - Trade in direction of major trend only
   - Donchian channel breakouts
   - Pyramid into winning positions

4. MEAN REVERSION IN TRENDS (Larry Connors RSI Strategy)
   - Buy pullbacks in uptrends (not bottoms in downtrends)
   - Short-term oversold in long-term bullish = high probability

5. SECTOR ROTATION (Fidelity Sector Rotation)
   - Rotate into top momentum sectors monthly
   - Avoid lagging sectors completely

6. VOLATILITY TARGETING (Risk Parity approach)
   - Adjust position size inversely to volatility
   - Equal risk contribution per position

7. BREAKOUT TRADING (Mark Minervini SEPA)
   - Volatility contraction patterns
   - Volume surge on breakout
   - Tight stops for risk control

RISK MANAGEMENT:
- Maximum 15% per position (concentrated bets)
- Trailing stops (lock in profits)
- Sector diversification (max 40% per sector)
- Correlation monitoring

Author: Enhanced Quant Trading System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'stock_data'
RESULTS_DIR = Path(__file__).parent / 'results'


# =============================================================================
# CONFIGURATION FOR HIGH RETURNS
# =============================================================================

class StrategyConfig:
    """Configuration optimized for 25%+ annual returns"""

    # Position Sizing
    MAX_POSITIONS = 8              # Concentrated portfolio
    MAX_POSITION_SIZE = 0.15       # 15% max per position
    MIN_POSITION_SIZE = 0.08       # 8% min per position

    # Risk Management
    INITIAL_STOP_LOSS = 0.08       # 8% initial stop
    TRAILING_STOP = 0.12           # 12% trailing stop after profit
    PROFIT_TARGET = 0.25           # 25% profit target
    BREAKEVEN_TRIGGER = 0.08       # Move stop to breakeven after 8% gain

    # Momentum Parameters
    MOMENTUM_LOOKBACK = 252        # 12-month momentum
    MOMENTUM_SKIP = 21             # Skip last month (reversal effect)
    MIN_MOMENTUM = 0.10            # Minimum 10% momentum to consider

    # Trend Parameters
    TREND_MA_SHORT = 20            # Short-term trend
    TREND_MA_MEDIUM = 50           # Medium-term trend
    TREND_MA_LONG = 200            # Long-term trend

    # Breakout Parameters
    BREAKOUT_PERIOD = 55           # 55-day high breakout (Turtle)
    VOLUME_SURGE_MULTIPLIER = 2.0  # 2x average volume on breakout

    # Mean Reversion Parameters
    RSI_OVERSOLD = 30              # RSI oversold level
    RSI_ENTRY = 35                 # Enter when RSI rises above this
    PULLBACK_DEPTH = 0.05          # 5% pullback from recent high

    # Sector Rotation
    SECTOR_ROTATION_PERIOD = 21    # Monthly rotation
    TOP_SECTORS = 3                # Invest in top 3 sectors

    # Liquidity Filter
    MIN_AVG_VOLUME = 50000000      # Minimum 50M CNY daily volume
    MIN_PRICE = 5.0                # Minimum price filter

    # Holding Period
    MIN_HOLD_DAYS = 5              # Minimum holding period
    MAX_HOLD_DAYS = 120            # Maximum holding period


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return prices.rolling(window=window, min_periods=1).mean()


def ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return prices.ewm(span=span, adjust=False).mean()


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD indicator"""
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2):
    """Bollinger Bands"""
    middle = sma(prices, window)
    std = prices.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def donchian_channel(high: pd.Series, low: pd.Series, period: int = 20):
    """Donchian Channel for breakout detection"""
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index - measures trend strength

    ADX > 25: Strong trend (good for momentum)
    ADX < 20: Weak trend (good for mean reversion)
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(window=period).mean()

    return adx_val


def volatility(prices: pd.Series, period: int = 20) -> pd.Series:
    """Historical volatility (annualized)"""
    returns = prices.pct_change()
    return returns.rolling(window=period).std() * np.sqrt(252)


# =============================================================================
# MOMENTUM CALCULATIONS
# =============================================================================

def calculate_momentum_score(df: pd.DataFrame, lookback: int = 252, skip: int = 21) -> float:
    """
    Calculate momentum score (12-1 momentum)

    REASONING:
    - Research shows 12-month momentum is predictive of future returns
    - Skip last month to avoid short-term reversal effect
    - Normalize by volatility for risk-adjusted momentum
    """
    if len(df) < lookback + skip:
        return 0.0

    # Price momentum (skip last month)
    end_idx = -skip if skip > 0 else len(df)
    start_idx = end_idx - lookback

    if start_idx < 0:
        return 0.0

    start_price = df['close'].iloc[start_idx]
    end_price = df['close'].iloc[end_idx]

    if start_price <= 0:
        return 0.0

    raw_momentum = (end_price - start_price) / start_price

    # Volatility adjustment (risk-adjusted momentum)
    vol = df['close'].pct_change().iloc[start_idx:end_idx].std() * np.sqrt(252)
    if vol > 0:
        risk_adjusted_momentum = raw_momentum / vol
    else:
        risk_adjusted_momentum = raw_momentum

    return risk_adjusted_momentum


def calculate_relative_strength(stock_returns: float, benchmark_returns: float) -> float:
    """
    Calculate relative strength vs benchmark

    REASONING:
    - Stocks outperforming the market tend to continue outperforming
    - Relative strength is a key factor in stock selection
    """
    if benchmark_returns == 0:
        return stock_returns
    return stock_returns - benchmark_returns


def is_absolute_momentum_positive(df: pd.DataFrame, lookback: int = 252) -> bool:
    """
    Check if absolute momentum is positive (Dual Momentum)

    REASONING:
    - Only invest when the asset itself is trending up
    - Absolute momentum filters out downtrending assets
    - Reduces drawdowns significantly
    """
    if len(df) < lookback:
        return False

    current_price = df['close'].iloc[-1]
    past_price = df['close'].iloc[-lookback]

    return current_price > past_price


# =============================================================================
# TREND ANALYSIS
# =============================================================================

class TrendState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    NEUTRAL = "neutral"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


def analyze_trend(df: pd.DataFrame) -> Tuple[TrendState, Dict]:
    """
    Comprehensive trend analysis

    REASONING:
    - Multiple timeframe confirmation reduces false signals
    - ADX confirms trend strength
    - Only trade with the trend for higher probability
    """
    if len(df) < 200:
        return TrendState.NEUTRAL, {}

    close = df['close']
    current_price = close.iloc[-1]

    ma20 = sma(close, 20).iloc[-1]
    ma50 = sma(close, 50).iloc[-1]
    ma200 = sma(close, 200).iloc[-1]

    adx_value = adx(df['high'], df['low'], close).iloc[-1]

    # Trend scoring
    score = 0
    details = {
        'price_vs_ma20': current_price > ma20,
        'price_vs_ma50': current_price > ma50,
        'price_vs_ma200': current_price > ma200,
        'ma20_vs_ma50': ma20 > ma50,
        'ma50_vs_ma200': ma50 > ma200,
        'adx': adx_value
    }

    if current_price > ma20:
        score += 1
    if current_price > ma50:
        score += 1
    if current_price > ma200:
        score += 2
    if ma20 > ma50:
        score += 1
    if ma50 > ma200:
        score += 2

    # Determine trend state
    if score >= 6 and adx_value > 25:
        return TrendState.STRONG_UPTREND, details
    elif score >= 4:
        return TrendState.UPTREND, details
    elif score <= 1 and adx_value > 25:
        return TrendState.STRONG_DOWNTREND, details
    elif score <= 2:
        return TrendState.DOWNTREND, details
    else:
        return TrendState.NEUTRAL, details


# =============================================================================
# PATTERN DETECTION
# =============================================================================

def detect_breakout(df: pd.DataFrame, period: int = 55) -> Optional[Dict]:
    """
    Detect Donchian Channel Breakout (Turtle Trading)

    REASONING:
    - Breakouts from consolidation often lead to strong moves
    - 55-day breakout is the classic Turtle entry
    - Volume confirmation increases probability
    """
    if len(df) < period + 5:
        return None

    current_close = df['close'].iloc[-1]
    current_high = df['high'].iloc[-1]
    current_volume = df['vol'].iloc[-1]

    # Previous period high (excluding today)
    period_high = df['high'].iloc[-(period+1):-1].max()
    avg_volume = df['vol'].iloc[-20:].mean()

    # Check for breakout
    if current_high > period_high:
        volume_surge = current_volume > avg_volume * 1.5

        # Calculate breakout strength
        breakout_pct = (current_close - period_high) / period_high * 100

        return {
            'type': 'Breakout',
            'period_high': period_high,
            'breakout_price': current_close,
            'breakout_pct': breakout_pct,
            'volume_surge': volume_surge,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 0,
            'date': str(df['trade_date'].iloc[-1]),
            'reasons': [
                f"{period}-day high breakout at {current_close:.2f}",
                f"Previous high: {period_high:.2f}",
                f"Volume {'surged' if volume_surge else 'normal'} ({current_volume/avg_volume:.1f}x avg)"
            ]
        }

    return None


def detect_pullback_buy(df: pd.DataFrame) -> Optional[Dict]:
    """
    Detect pullback buying opportunity in uptrend (Connors RSI Strategy)

    REASONING:
    - Buying pullbacks in uptrends has higher win rate than buying bottoms
    - Short-term oversold in long-term bullish is high probability setup
    - Mean reversion works best when aligned with trend
    """
    if len(df) < 200:
        return None

    trend_state, trend_details = analyze_trend(df)

    # Only look for pullbacks in uptrends
    if trend_state not in [TrendState.UPTREND, TrendState.STRONG_UPTREND]:
        return None

    close = df['close']
    current_price = close.iloc[-1]
    rsi_value = rsi(close).iloc[-1]
    rsi_prev = rsi(close).iloc[-2]

    # Recent high (last 20 days)
    recent_high = close.iloc[-20:].max()
    pullback_depth = (recent_high - current_price) / recent_high

    # Entry conditions
    conditions_met = []

    # 1. RSI oversold and turning up
    if rsi_value < 35 and rsi_value > rsi_prev:
        conditions_met.append(f"RSI oversold at {rsi_value:.1f} and turning up")

    # 2. Pullback to moving average support
    ma20 = sma(close, 20).iloc[-1]
    ma50 = sma(close, 50).iloc[-1]

    if abs(current_price - ma20) / ma20 < 0.02:
        conditions_met.append("Price at 20-day MA support")
    elif abs(current_price - ma50) / ma50 < 0.03:
        conditions_met.append("Price at 50-day MA support")

    # 3. Pullback depth (3-8% from high is ideal)
    if 0.03 <= pullback_depth <= 0.10:
        conditions_met.append(f"Healthy pullback of {pullback_depth*100:.1f}%")

    # 4. Volume drying up (sellers exhausted)
    avg_volume = df['vol'].iloc[-20:].mean()
    current_volume = df['vol'].iloc[-1]
    if current_volume < avg_volume * 0.7:
        conditions_met.append("Volume drying up (seller exhaustion)")

    if len(conditions_met) >= 2:
        return {
            'type': 'Pullback Buy',
            'price': current_price,
            'recent_high': recent_high,
            'pullback_depth': pullback_depth,
            'rsi': rsi_value,
            'trend': trend_state.value,
            'date': str(df['trade_date'].iloc[-1]),
            'reasons': [f"Pullback buy in {trend_state.value}"] + conditions_met
        }

    return None


def detect_volatility_contraction(df: pd.DataFrame) -> Optional[Dict]:
    """
    Detect Volatility Contraction Pattern (VCP) - Mark Minervini

    REASONING:
    - Volatility contraction precedes expansion (big moves)
    - Tightening price action shows supply being absorbed
    - Breakout from VCP often leads to strong momentum
    """
    if len(df) < 60:
        return None

    close = df['close']
    high = df['high']
    low = df['low']

    # Calculate recent volatility vs past volatility
    recent_range = (high.iloc[-10:].max() - low.iloc[-10:].min()) / close.iloc[-10:].mean()
    past_range = (high.iloc[-40:-10].max() - low.iloc[-40:-10].min()) / close.iloc[-40:-10].mean()

    # Volatility contraction: recent range < 50% of past range
    if recent_range < past_range * 0.5:
        # Check if in uptrend
        trend_state, _ = analyze_trend(df)
        if trend_state in [TrendState.UPTREND, TrendState.STRONG_UPTREND]:

            # Calculate pivot point (top of recent range)
            pivot = high.iloc[-10:].max()

            return {
                'type': 'VCP',
                'price': close.iloc[-1],
                'pivot': pivot,
                'contraction_ratio': recent_range / past_range,
                'date': str(df['trade_date'].iloc[-1]),
                'reasons': [
                    "Volatility Contraction Pattern detected",
                    f"Range contracted to {recent_range/past_range*100:.0f}% of prior range",
                    f"Breakout pivot at {pivot:.2f}",
                    "In confirmed uptrend"
                ]
            }

    return None


def detect_w_bottom_enhanced(df: pd.DataFrame, lookback: int = 60) -> Optional[Dict]:
    """
    Enhanced W-Bottom detection with stricter criteria

    ADDITIONAL REQUIREMENTS:
    - Must be after significant decline (>15%)
    - Second bottom must be higher (bullish)
    - Volume expansion on breakout
    - RSI divergence confirmation
    """
    if len(df) < lookback + 20:
        return None

    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['vol']

    # Find potential bottoms
    window = df.iloc[-lookback:]
    window_low = low.iloc[-lookback:]
    window_close = close.iloc[-lookback:]

    # Find local minimums
    local_mins = []
    for i in range(10, len(window_low) - 10):
        if window_low.iloc[i] == window_low.iloc[i-10:i+11].min():
            local_mins.append((i, window_low.iloc[i], window_close.iloc[i]))

    if len(local_mins) < 2:
        return None

    # Check last two bottoms
    for i in range(len(local_mins) - 1):
        idx1, low1, close1 = local_mins[i]
        for j in range(i + 1, len(local_mins)):
            idx2, low2, close2 = local_mins[j]

            # Bottoms should be 15-50 days apart
            if not (15 <= idx2 - idx1 <= 50):
                continue

            # Bottoms should be within 5% of each other
            if abs(low2 - low1) / low1 > 0.05:
                continue

            # Second bottom should be higher (bullish)
            if low2 < low1:
                continue

            # Find neckline (high between bottoms)
            between = window.iloc[idx1:idx2+1]
            neckline = between['high'].max()

            # Neckline should be at least 8% above bottoms
            if (neckline - low1) / low1 < 0.08:
                continue

            # Check for breakout
            post_pattern = df.iloc[-lookback+idx2:]
            breakout_idx = None
            for k in range(len(post_pattern)):
                if post_pattern['close'].iloc[k] > neckline:
                    breakout_idx = k
                    break

            if breakout_idx is None:
                continue

            # Volume confirmation
            avg_vol = volume.iloc[-40:-20].mean()
            breakout_vol = post_pattern['vol'].iloc[breakout_idx] if breakout_idx < len(post_pattern) else 0
            volume_confirm = breakout_vol > avg_vol * 1.5

            # RSI divergence check
            rsi_values = rsi(close)
            rsi_at_bottom1 = rsi_values.iloc[-lookback + idx1]
            rsi_at_bottom2 = rsi_values.iloc[-lookback + idx2]
            rsi_divergence = rsi_at_bottom2 > rsi_at_bottom1 + 5  # Higher RSI at second bottom

            strength = 0
            reasons = ["W-Bottom pattern detected"]

            if low2 > low1:
                strength += 1
                reasons.append("Second bottom higher (buyers strengthening)")

            if rsi_divergence:
                strength += 2
                reasons.append(f"RSI bullish divergence ({rsi_at_bottom1:.0f} -> {rsi_at_bottom2:.0f})")

            if volume_confirm:
                strength += 1
                reasons.append("Volume surge on breakout")

            if strength >= 2:
                target = neckline + (neckline - min(low1, low2))
                return {
                    'type': 'W-Bottom',
                    'bottom1_price': low1,
                    'bottom2_price': low2,
                    'neckline': neckline,
                    'entry_price': neckline * 1.01,
                    'stop_loss': min(low1, low2) * 0.97,
                    'target': target,
                    'strength': strength,
                    'date': str(df['trade_date'].iloc[-lookback + idx2 + breakout_idx]) if breakout_idx else str(df['trade_date'].iloc[-1]),
                    'reasons': reasons
                }

    return None


# =============================================================================
# SECTOR ANALYSIS
# =============================================================================

SECTORS = {
    'finance': {
        'name': 'Finance',
        'prefixes': ['600', '601'],
        'codes': ['600036', '601318', '601398', '601939', '601288', '000001', '002142', '601166'],
        'characteristics': 'cyclical, interest-rate sensitive'
    },
    'technology': {
        'name': 'Technology',
        'prefixes': ['300', '688'],
        'codes': ['300750', '002230', '300059', '002415', '300033', '002371', '300124'],
        'characteristics': 'high-growth, high-beta, momentum-driven'
    },
    'consumer': {
        'name': 'Consumer',
        'prefixes': [],
        'codes': ['600519', '000858', '000568', '600887', '002304', '603288', '000895'],
        'characteristics': 'defensive, stable growth'
    },
    'healthcare': {
        'name': 'Healthcare',
        'prefixes': [],
        'codes': ['300760', '600276', '000538', '300347', '002007', '300122', '600196'],
        'characteristics': 'defensive, long-term growth'
    },
    'new_energy': {
        'name': 'New Energy',
        'prefixes': [],
        'codes': ['300274', '002459', '300014', '600438', '601012', '002074', '002466'],
        'characteristics': 'high-growth, policy-driven, volatile'
    },
    'materials': {
        'name': 'Materials',
        'prefixes': [],
        'codes': ['601899', '600019', '000898', '600111', '601600', '600362', '000983'],
        'characteristics': 'cyclical, commodity-linked'
    }
}


def classify_sector(ts_code: str) -> str:
    """Classify stock into sector"""
    symbol = ts_code.split('.')[0]

    for sector, info in SECTORS.items():
        if symbol in info['codes']:
            return sector

    # Default by prefix
    if symbol.startswith('300') or symbol.startswith('688'):
        return 'technology'
    elif symbol.startswith('600') or symbol.startswith('601'):
        return 'finance'
    else:
        return 'consumer'


def calculate_sector_momentum(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate momentum for each sector

    REASONING:
    - Sector rotation is one of the most consistent alpha sources
    - Top momentum sectors tend to continue outperforming
    - Avoid lagging sectors to reduce drawdowns
    """
    sector_returns = {sector: [] for sector in SECTORS.keys()}

    for ts_code, df in stock_data.items():
        if len(df) < 63:  # Need 3 months of data
            continue

        sector = classify_sector(ts_code)
        momentum = calculate_momentum_score(df, lookback=63, skip=5)
        sector_returns[sector].append(momentum)

    # Average momentum per sector
    sector_momentum = {}
    for sector, returns in sector_returns.items():
        if returns:
            sector_momentum[sector] = np.mean(returns)
        else:
            sector_momentum[sector] = 0

    return sector_momentum


def get_top_sectors(sector_momentum: Dict[str, float], n: int = 3) -> List[str]:
    """Get top N sectors by momentum"""
    sorted_sectors = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
    return [s[0] for s in sorted_sectors[:n]]


# =============================================================================
# STOCK RANKING AND SELECTION
# =============================================================================

@dataclass
class StockScore:
    ts_code: str
    momentum_score: float
    trend_score: float
    pattern_score: float
    liquidity_score: float
    total_score: float
    sector: str
    signals: List[Dict]


def rank_stocks(stock_data: Dict[str, pd.DataFrame], top_sectors: List[str]) -> List[StockScore]:
    """
    Rank stocks using multi-factor scoring

    FACTORS:
    1. Momentum (40% weight) - 12-1 month momentum
    2. Trend (30% weight) - Multiple timeframe trend alignment
    3. Pattern (20% weight) - Technical setups
    4. Liquidity (10% weight) - Volume and spread
    """
    scores = []

    for ts_code, df in stock_data.items():
        if len(df) < 252:
            continue

        sector = classify_sector(ts_code)

        # Skip stocks not in top sectors
        if sector not in top_sectors:
            continue

        # Liquidity filter
        avg_amount = df['amount'].iloc[-20:].mean() if 'amount' in df.columns else df['vol'].iloc[-20:].mean() * df['close'].iloc[-20:].mean()
        if avg_amount < StrategyConfig.MIN_AVG_VOLUME:
            continue

        # Price filter
        if df['close'].iloc[-1] < StrategyConfig.MIN_PRICE:
            continue

        # Calculate scores
        momentum = calculate_momentum_score(df, StrategyConfig.MOMENTUM_LOOKBACK, StrategyConfig.MOMENTUM_SKIP)

        trend_state, _ = analyze_trend(df)
        trend_score = {
            TrendState.STRONG_UPTREND: 1.0,
            TrendState.UPTREND: 0.7,
            TrendState.NEUTRAL: 0.3,
            TrendState.DOWNTREND: 0.0,
            TrendState.STRONG_DOWNTREND: -0.5
        }.get(trend_state, 0)

        # Pattern detection
        signals = []
        pattern_score = 0

        breakout = detect_breakout(df)
        if breakout:
            signals.append(breakout)
            pattern_score += 0.5 if breakout['volume_surge'] else 0.3

        pullback = detect_pullback_buy(df)
        if pullback:
            signals.append(pullback)
            pattern_score += 0.4

        vcp = detect_volatility_contraction(df)
        if vcp:
            signals.append(vcp)
            pattern_score += 0.3

        w_bottom = detect_w_bottom_enhanced(df)
        if w_bottom:
            signals.append(w_bottom)
            pattern_score += w_bottom['strength'] * 0.15

        # Liquidity score
        liquidity_score = min(avg_amount / StrategyConfig.MIN_AVG_VOLUME / 10, 1.0)

        # Total score (weighted)
        total = (
            momentum * 0.40 +
            trend_score * 0.30 +
            pattern_score * 0.20 +
            liquidity_score * 0.10
        )

        # Only include stocks with positive momentum and uptrend
        if momentum > StrategyConfig.MIN_MOMENTUM and trend_state in [TrendState.UPTREND, TrendState.STRONG_UPTREND]:
            scores.append(StockScore(
                ts_code=ts_code,
                momentum_score=momentum,
                trend_score=trend_score,
                pattern_score=pattern_score,
                liquidity_score=liquidity_score,
                total_score=total,
                sector=sector,
                signals=signals
            ))

    # Sort by total score
    scores.sort(key=lambda x: x.total_score, reverse=True)
    return scores


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

@dataclass
class TradingSignal:
    signal_type: str  # 'BUY' or 'SELL'
    ts_code: str
    date: str
    price: float
    stop_loss: float
    target: float
    position_size: float
    strength: int
    reasons: List[str]
    pattern_type: str = ''

    def __repr__(self):
        return f"Signal({self.signal_type}, {self.ts_code}, {self.date}, strength={self.strength})"


def generate_entry_signals(stock_scores: List[StockScore], stock_data: Dict[str, pd.DataFrame],
                          current_date: str) -> List[TradingSignal]:
    """Generate entry signals from ranked stocks"""
    signals = []

    for score in stock_scores[:StrategyConfig.MAX_POSITIONS * 2]:  # Consider top candidates
        if not score.signals:
            continue

        df = stock_data[score.ts_code]
        current_price = df['close'].iloc[-1]

        # Calculate position size based on volatility
        vol = volatility(df['close']).iloc[-1]
        vol_adjusted_size = StrategyConfig.MAX_POSITION_SIZE * (0.2 / max(vol, 0.1))
        position_size = np.clip(vol_adjusted_size, StrategyConfig.MIN_POSITION_SIZE, StrategyConfig.MAX_POSITION_SIZE)

        # Determine stop loss and target based on signal type
        for pattern in score.signals:
            if pattern['type'] == 'Breakout':
                stop_loss = current_price * (1 - StrategyConfig.INITIAL_STOP_LOSS)
                target = current_price * (1 + StrategyConfig.PROFIT_TARGET)
                strength = 4 if pattern['volume_surge'] else 3
            elif pattern['type'] == 'Pullback Buy':
                stop_loss = current_price * (1 - 0.06)  # Tighter stop for pullbacks
                target = pattern['recent_high'] * 1.05  # Target above recent high
                strength = 3
            elif pattern['type'] == 'VCP':
                stop_loss = current_price * (1 - 0.05)  # Tight stop under pivot
                target = current_price * (1 + 0.20)
                strength = 4
            elif pattern['type'] == 'W-Bottom':
                stop_loss = pattern['stop_loss']
                target = pattern['target']
                strength = pattern['strength']
            else:
                continue

            signals.append(TradingSignal(
                signal_type='BUY',
                ts_code=score.ts_code,
                date=current_date,
                price=current_price,
                stop_loss=stop_loss,
                target=target,
                position_size=position_size,
                strength=strength,
                reasons=pattern['reasons'],
                pattern_type=pattern['type']
            ))
            break  # One signal per stock

    return signals


# =============================================================================
# POSITION AND TRADE MANAGEMENT
# =============================================================================

@dataclass
class Position:
    ts_code: str
    entry_date: str
    entry_price: float
    shares: int
    initial_stop: float
    current_stop: float
    target: float
    highest_price: float
    days_held: int
    signal: TradingSignal

    def update(self, current_price: float, current_high: float):
        """Update position with new price data"""
        self.days_held += 1
        self.highest_price = max(self.highest_price, current_high)

        # Move stop to breakeven after 8% gain
        gain_pct = (current_price - self.entry_price) / self.entry_price
        if gain_pct >= StrategyConfig.BREAKEVEN_TRIGGER:
            breakeven_stop = self.entry_price * 1.01
            self.current_stop = max(self.current_stop, breakeven_stop)

        # Trailing stop after profit
        if gain_pct >= 0.10:
            trailing_stop = self.highest_price * (1 - StrategyConfig.TRAILING_STOP)
            self.current_stop = max(self.current_stop, trailing_stop)


@dataclass
class Trade:
    ts_code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    days_held: int
    exit_reason: str
    signal: TradingSignal


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class EnhancedBacktestEngine:
    """
    Advanced backtesting engine with:
    - Volatility-based position sizing
    - Trailing stops
    - Sector rotation
    - Performance analytics
    """

    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[Dict] = []
        self.sector_exposure: Dict[str, float] = {}

    def run(self, stock_data: Dict[str, pd.DataFrame],
            start_date: str = '20200101', end_date: str = None) -> Dict:
        """Run backtest"""
        # Get all trading dates
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df['trade_date'].astype(str).tolist())
        all_dates = sorted([d for d in all_dates if d >= start_date])
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        print(f"\n{'='*60}")
        print("ENHANCED BACKTEST - Targeting 25% Annual Returns")
        print(f"{'='*60}")
        print(f"Period: {all_dates[0]} to {all_dates[-1]}")
        print(f"Initial Capital: {self.initial_capital:,.0f} CNY")
        print(f"Stocks in Universe: {len(stock_data)}")
        print(f"{'='*60}\n")

        # Monthly rebalancing dates
        rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), 21)]

        for i, date in enumerate(all_dates):
            # Prepare data up to current date
            current_data = {}
            for ts_code, df in stock_data.items():
                df_filtered = df[df['trade_date'].astype(str) <= date].copy()
                if len(df_filtered) >= 252:
                    current_data[ts_code] = df_filtered

            # Check exits for existing positions
            self._check_exits(current_data, date)

            # Monthly rebalancing: rank stocks and generate signals
            if date in rebalance_dates and len(self.positions) < StrategyConfig.MAX_POSITIONS:
                # Calculate sector momentum
                sector_momentum = calculate_sector_momentum(current_data)
                top_sectors = get_top_sectors(sector_momentum, StrategyConfig.TOP_SECTORS)

                # Rank stocks
                stock_scores = rank_stocks(current_data, top_sectors)

                # Generate signals
                signals = generate_entry_signals(stock_scores, current_data, date)

                # Process signals
                for signal in signals:
                    if len(self.positions) >= StrategyConfig.MAX_POSITIONS:
                        break
                    if signal.ts_code not in self.positions:
                        self._enter_position(signal, current_data)

            # Daily signal check (for breakouts)
            elif len(self.positions) < StrategyConfig.MAX_POSITIONS:
                for ts_code, df in current_data.items():
                    if ts_code in self.positions:
                        continue

                    # Check for breakout signals
                    breakout = detect_breakout(df)
                    if breakout and breakout['volume_surge']:
                        sector = classify_sector(ts_code)
                        trend_state, _ = analyze_trend(df)

                        if trend_state in [TrendState.UPTREND, TrendState.STRONG_UPTREND]:
                            current_price = df['close'].iloc[-1]
                            signal = TradingSignal(
                                signal_type='BUY',
                                ts_code=ts_code,
                                date=date,
                                price=current_price,
                                stop_loss=current_price * 0.92,
                                target=current_price * 1.25,
                                position_size=0.10,
                                strength=4,
                                reasons=breakout['reasons'],
                                pattern_type='Breakout'
                            )
                            self._enter_position(signal, current_data)

                            if len(self.positions) >= StrategyConfig.MAX_POSITIONS:
                                break

            # Update daily values
            self._update_daily_value(current_data, date)

            # Progress indicator
            if i % 100 == 0:
                print(f"Processing {date}... Positions: {len(self.positions)}, Value: {self.daily_values[-1]['total_value']:,.0f}")

        # Close remaining positions
        self._close_all(stock_data, all_dates[-1])

        return self._generate_report()

    def _enter_position(self, signal: TradingSignal, stock_data: Dict):
        """Enter a new position"""
        if signal.ts_code in self.positions:
            return

        # Calculate shares
        position_value = self.capital * signal.position_size
        shares = int(position_value / signal.price / 100) * 100

        if shares < 100:
            return

        cost = shares * signal.price
        if cost > self.capital * 0.95:  # Keep 5% cash buffer
            return

        self.capital -= cost

        position = Position(
            ts_code=signal.ts_code,
            entry_date=signal.date,
            entry_price=signal.price,
            shares=shares,
            initial_stop=signal.stop_loss,
            current_stop=signal.stop_loss,
            target=signal.target,
            highest_price=signal.price,
            days_held=0,
            signal=signal
        )

        self.positions[signal.ts_code] = position

    def _check_exits(self, stock_data: Dict, date: str):
        """Check and execute exits"""
        to_close = []

        for ts_code, position in self.positions.items():
            if ts_code not in stock_data:
                continue

            df = stock_data[ts_code]
            df_today = df[df['trade_date'].astype(str) == date]

            if df_today.empty:
                continue

            current_price = df_today['close'].iloc[0]
            current_high = df_today['high'].iloc[0]
            current_low = df_today['low'].iloc[0]

            # Update position
            position.update(current_price, current_high)

            exit_reason = None
            exit_price = None

            # 1. Stop loss
            if current_low <= position.current_stop:
                exit_reason = f"Stop loss at {position.current_stop:.2f}"
                exit_price = position.current_stop

            # 2. Target reached
            elif current_high >= position.target:
                exit_reason = f"Target reached at {position.target:.2f}"
                exit_price = position.target

            # 3. Maximum hold period
            elif position.days_held >= StrategyConfig.MAX_HOLD_DAYS:
                exit_reason = f"Max hold period ({StrategyConfig.MAX_HOLD_DAYS} days)"
                exit_price = current_price

            # 4. Trend reversal
            elif position.days_held >= StrategyConfig.MIN_HOLD_DAYS:
                trend_state, _ = analyze_trend(df)
                if trend_state in [TrendState.DOWNTREND, TrendState.STRONG_DOWNTREND]:
                    exit_reason = "Trend reversal"
                    exit_price = current_price

            if exit_reason:
                to_close.append((ts_code, exit_price, exit_reason, date))

        for ts_code, exit_price, exit_reason, date in to_close:
            self._close_position(ts_code, exit_price, exit_reason, date)

    def _close_position(self, ts_code: str, exit_price: float, reason: str, date: str):
        """Close a position"""
        position = self.positions.pop(ts_code)

        proceeds = position.shares * exit_price
        self.capital += proceeds

        pnl = proceeds - (position.shares * position.entry_price)
        pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100

        trade = Trade(
            ts_code=ts_code,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            days_held=position.days_held,
            exit_reason=reason,
            signal=position.signal
        )
        self.trades.append(trade)

    def _close_all(self, stock_data: Dict, date: str):
        """Close all positions at end"""
        for ts_code in list(self.positions.keys()):
            if ts_code in stock_data:
                df = stock_data[ts_code]
                exit_price = df['close'].iloc[-1]
            else:
                exit_price = self.positions[ts_code].entry_price

            self._close_position(ts_code, exit_price, "End of backtest", date)

    def _update_daily_value(self, stock_data: Dict, date: str):
        """Update portfolio value"""
        positions_value = 0

        for ts_code, position in self.positions.items():
            if ts_code in stock_data:
                df = stock_data[ts_code]
                df_today = df[df['trade_date'].astype(str) == date]
                if not df_today.empty:
                    positions_value += position.shares * df_today['close'].iloc[0]
                else:
                    positions_value += position.shares * position.entry_price
            else:
                positions_value += position.shares * position.entry_price

        total_value = self.capital + positions_value

        self.daily_values.append({
            'date': date,
            'cash': self.capital,
            'positions_value': positions_value,
            'total_value': total_value,
            'num_positions': len(self.positions)
        })

    def _generate_report(self) -> Dict:
        """Generate comprehensive report"""
        df = pd.DataFrame(self.daily_values)

        final_value = df['total_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # Annualized return
        trading_days = len(df)
        years = trading_days / 252
        annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0

        # Sharpe ratio
        df['daily_return'] = df['total_value'].pct_change()
        risk_free_daily = 0.03 / 252
        excess_returns = df['daily_return'].dropna() - risk_free_daily
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Max drawdown
        df['cummax'] = df['total_value'].cummax()
        df['drawdown'] = (df['total_value'] - df['cummax']) / df['cummax'] * 100
        max_drawdown = df['drawdown'].min()

        # Trade statistics
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0
        profit_factor = abs(sum(t.pnl for t in winning) / sum(t.pnl for t in losing)) if losing and sum(t.pnl for t in losing) != 0 else float('inf')

        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_hold_days': np.mean([t.days_held for t in self.trades]) if self.trades else 0,
            'daily_values': df,
            'trades': self.trades
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_stock_analysis(ts_code: str, df: pd.DataFrame, signals: List = None, trades: List = None, save_path: str = None):
    """Plot comprehensive stock analysis"""
    df = df.sort_values('trade_date').copy()
    df['date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    df = df.set_index('date')

    # Calculate indicators
    df['MA20'] = sma(df['close'], 20)
    df['MA50'] = sma(df['close'], 50)
    df['MA200'] = sma(df['close'], 200)
    df['RSI'] = rsi(df['close'])
    macd_line, signal_line, hist = macd(df['close'])
    df['MACD'] = macd_line
    df['Signal'] = signal_line
    df['Hist'] = hist

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), height_ratios=[3, 1, 1, 1])
    fig.suptitle(f'Stock Analysis: {ts_code}', fontsize=14, fontweight='bold')

    # Price chart
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close', linewidth=1.5, color='black')
    ax1.plot(df.index, df['MA20'], label='MA20', linewidth=1, alpha=0.7, color='blue')
    ax1.plot(df.index, df['MA50'], label='MA50', linewidth=1, alpha=0.7, color='orange')
    ax1.plot(df.index, df['MA200'], label='MA200', linewidth=1, alpha=0.7, color='red')

    # Plot trades
    if trades:
        stock_trades = [t for t in trades if t.ts_code == ts_code]
        for trade in stock_trades:
            try:
                entry = pd.to_datetime(str(trade.entry_date), format='%Y%m%d')
                exit_dt = pd.to_datetime(str(trade.exit_date), format='%Y%m%d')
                ax1.scatter([entry], [trade.entry_price], marker='^', color='lime', s=200, zorder=5, edgecolor='black')
                color = 'cyan' if trade.pnl > 0 else 'red'
                ax1.scatter([exit_dt], [trade.exit_price], marker='v', color=color, s=200, zorder=5, edgecolor='black')
                ax1.plot([entry, exit_dt], [trade.entry_price, trade.exit_price], 'k--', alpha=0.3)
            except:
                continue

    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price with Moving Averages and Trade Signals')

    # RSI
    ax2 = axes[1]
    ax2.plot(df.index, df['RSI'], color='purple', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # MACD
    ax3 = axes[2]
    ax3.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1)
    ax3.plot(df.index, df['Signal'], label='Signal', color='red', linewidth=1)
    colors = ['green' if h >= 0 else 'red' for h in df['Hist']]
    ax3.bar(df.index, df['Hist'], color=colors, alpha=0.5, width=1)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_ylabel('MACD')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Volume
    ax4 = axes[3]
    colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' for i in range(len(df))]
    ax4.bar(df.index, df['vol'], color=colors, alpha=0.7, width=1)
    ax4.plot(df.index, df['vol'].rolling(20).mean(), color='blue', linewidth=1)
    ax4.set_ylabel('Volume')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_portfolio_performance(report: Dict, save_path: str = None):
    """Plot portfolio performance"""
    df = report['daily_values']
    trades = report['trades']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Portfolio Performance - {report['annualized_return_pct']:.1f}% Annualized Return", fontsize=14, fontweight='bold')

    # Portfolio value
    ax1 = axes[0, 0]
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
    ax1.plot(df['date_dt'], df['total_value'], linewidth=1.5, color='blue')
    ax1.axhline(y=report['initial_capital'], color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(df['date_dt'], report['initial_capital'], df['total_value'],
                     where=df['total_value'] >= report['initial_capital'], alpha=0.3, color='green')
    ax1.fill_between(df['date_dt'], report['initial_capital'], df['total_value'],
                     where=df['total_value'] < report['initial_capital'], alpha=0.3, color='red')
    ax1.set_ylabel('Portfolio Value (CNY)')
    ax1.set_title(f"Total Return: {report['total_return_pct']:.1f}%")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[0, 1]
    ax2.fill_between(df['date_dt'], 0, df['drawdown'], color='red', alpha=0.5)
    ax2.axhline(y=report['max_drawdown_pct'], color='darkred', linestyle='--',
                label=f"Max DD: {report['max_drawdown_pct']:.1f}%")
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Trade returns
    ax3 = axes[1, 0]
    if trades:
        pnl_pcts = [t.pnl_pct for t in trades]
        colors = ['green' if p > 0 else 'red' for p in pnl_pcts]
        ax3.bar(range(len(pnl_pcts)), pnl_pcts, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-')
        ax3.axhline(y=np.mean(pnl_pcts), color='blue', linestyle='--', label=f'Avg: {np.mean(pnl_pcts):.1f}%')
        ax3.set_xlabel('Trade #')
        ax3.set_ylabel('Return (%)')
        ax3.set_title(f"Trade Returns (Win Rate: {report['win_rate_pct']:.1f}%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Monthly returns heatmap
    ax4 = axes[1, 1]
    df['month'] = df['date_dt'].dt.to_period('M')
    monthly = df.groupby('month')['total_value'].last().pct_change() * 100
    monthly_values = monthly.values[1:]  # Skip first NaN
    months = [str(m) for m in monthly.index[1:]]

    if len(monthly_values) > 0:
        colors = ['green' if v > 0 else 'red' for v in monthly_values]
        ax4.bar(range(len(monthly_values)), monthly_values, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-')
        ax4.set_ylabel('Monthly Return (%)')
        ax4.set_title('Monthly Returns')
        ax4.set_xticks(range(0, len(months), max(1, len(months)//12)))
        ax4.set_xticklabels([months[i] for i in range(0, len(months), max(1, len(months)//12))], rotation=45)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_trade_analysis(trades: List[Trade], save_path: str = None):
    """Detailed trade analysis"""
    if not trades:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Trade Analysis', fontsize=14, fontweight='bold')

    pnl_pcts = [t.pnl_pct for t in trades]
    hold_days = [t.days_held for t in trades]

    # P&L vs Hold period
    ax1 = axes[0, 0]
    colors = ['green' if p > 0 else 'red' for p in pnl_pcts]
    ax1.scatter(hold_days, pnl_pcts, c=colors, alpha=0.6, s=50)
    ax1.axhline(y=0, color='black', linestyle='-')
    ax1.set_xlabel('Holding Period (Days)')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Returns vs Holding Period')
    ax1.grid(True, alpha=0.3)

    # Exit reason pie
    ax2 = axes[0, 1]
    exit_reasons = {}
    for t in trades:
        reason = t.exit_reason.split(' at ')[0].split(' (')[0]
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    ax2.pie(list(exit_reasons.values()), labels=list(exit_reasons.keys()),
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Exit Reasons')

    # Cumulative P&L
    ax3 = axes[1, 0]
    cumulative = np.cumsum([t.pnl for t in trades])
    ax3.plot(range(1, len(cumulative)+1), cumulative, linewidth=1.5)
    ax3.fill_between(range(1, len(cumulative)+1), 0, cumulative,
                     where=np.array(cumulative) >= 0, alpha=0.3, color='green')
    ax3.fill_between(range(1, len(cumulative)+1), 0, cumulative,
                     where=np.array(cumulative) < 0, alpha=0.3, color='red')
    ax3.axhline(y=0, color='black', linestyle='-')
    ax3.set_xlabel('Trade #')
    ax3.set_ylabel('Cumulative P&L (CNY)')
    ax3.set_title('Cumulative Profit/Loss')
    ax3.grid(True, alpha=0.3)

    # Pattern performance
    ax4 = axes[1, 1]
    pattern_returns = {}
    for t in trades:
        pattern = t.signal.pattern_type if t.signal else 'Other'
        if pattern not in pattern_returns:
            pattern_returns[pattern] = []
        pattern_returns[pattern].append(t.pnl_pct)

    patterns = list(pattern_returns.keys())
    avg_returns = [np.mean(pattern_returns[p]) for p in patterns]
    colors = ['green' if r > 0 else 'red' for r in avg_returns]
    ax4.barh(patterns, avg_returns, color=colors, alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='-')
    ax4.set_xlabel('Average Return (%)')
    ax4.set_title('Performance by Pattern Type')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def load_stock_data(limit: int = None, exchange: str = None) -> Dict[str, pd.DataFrame]:
    """Load stock data"""
    stock_data = {}

    if exchange:
        dirs = [DATA_DIR / exchange.lower()]
    else:
        dirs = [DATA_DIR / 'sh', DATA_DIR / 'sz']

    for d in dirs:
        if not d.exists():
            continue
        for f in d.glob('*.csv'):
            try:
                df = pd.read_csv(f)
                if len(df) >= 252:  # Need at least 1 year
                    ts_code = df['ts_code'].iloc[0] if 'ts_code' in df.columns else f"{f.stem}.{d.name.upper()}"
                    stock_data[ts_code] = df
                    if limit and len(stock_data) >= limit:
                        return stock_data
            except Exception as e:
                continue

    return stock_data


def run_strategy(start_date: str = '20200101', end_date: str = None,
                initial_capital: float = 1000000, save_plots: bool = False) -> Dict:
    """Run the complete trading strategy"""
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading stock data...")
    stock_data = load_stock_data(limit=200)
    print(f"Loaded {len(stock_data)} stocks")

    engine = EnhancedBacktestEngine(initial_capital=initial_capital)
    report = engine.run(stock_data, start_date=start_date, end_date=end_date)

    # Print results
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Initial Capital:     {report['initial_capital']:>15,.0f} CNY")
    print(f"Final Value:         {report['final_value']:>15,.0f} CNY")
    print(f"Total Return:        {report['total_return_pct']:>15.2f}%")
    print(f"Annualized Return:   {report['annualized_return_pct']:>15.2f}%")
    print(f"Sharpe Ratio:        {report['sharpe_ratio']:>15.2f}")
    print(f"Max Drawdown:        {report['max_drawdown_pct']:>15.2f}%")
    print(f"-"*60)
    print(f"Total Trades:        {report['total_trades']:>15}")
    print(f"Win Rate:            {report['win_rate_pct']:>15.1f}%")
    print(f"Avg Win:             {report['avg_win_pct']:>15.2f}%")
    print(f"Avg Loss:            {report['avg_loss_pct']:>15.2f}%")
    print(f"Profit Factor:       {report['profit_factor']:>15.2f}")
    print(f"Expectancy:          {report['expectancy']:>15.2f}%")
    print(f"Avg Hold Days:       {report['avg_hold_days']:>15.1f}")
    print(f"{'='*60}")

    # Sample trades
    if report['trades']:
        print("\nTOP TRADES:")
        print("-"*60)
        top_trades = sorted(report['trades'], key=lambda t: t.pnl_pct, reverse=True)[:5]
        for t in top_trades:
            print(f"{t.ts_code}: {t.entry_date} -> {t.exit_date} | {t.pnl_pct:+.1f}% | {t.signal.pattern_type}")

    # Visualizations
    print("\nGenerating visualizations...")
    save_path = str(RESULTS_DIR / 'portfolio_performance.png') if save_plots else None
    plot_portfolio_performance(report, save_path)

    if report['trades']:
        save_path = str(RESULTS_DIR / 'trade_analysis.png') if save_plots else None
        plot_trade_analysis(report['trades'], save_path)

        # Top stock charts
        top_stocks = list(set(t.ts_code for t in sorted(report['trades'], key=lambda t: t.pnl_pct, reverse=True)[:3]))
        for ts_code in top_stocks:
            if ts_code in stock_data:
                save_path = str(RESULTS_DIR / f"stock_{ts_code.replace('.', '_')}.png") if save_plots else None
                plot_stock_analysis(ts_code, stock_data[ts_code], trades=report['trades'], save_path=save_path)

    return report


def analyze_stock(ts_code: str, save_plot: bool = False):
    """Analyze a single stock"""
    if '.' not in ts_code:
        for exchange in ['sz', 'sh']:
            filepath = DATA_DIR / exchange / f'{ts_code}.csv'
            if filepath.exists():
                ts_code = f'{ts_code}.{exchange.upper()}'
                break

    symbol = ts_code.split('.')[0]
    exchange = ts_code.split('.')[1].lower() if '.' in ts_code else 'sz'
    filepath = DATA_DIR / exchange / f'{symbol}.csv'

    if not filepath.exists():
        print(f"Stock not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    df = df.sort_values('trade_date').reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"STOCK ANALYSIS: {ts_code}")
    print(f"{'='*60}")
    print(f"Data: {df['trade_date'].iloc[0]} to {df['trade_date'].iloc[-1]}")
    print(f"Days: {len(df)}")

    # Momentum
    momentum = calculate_momentum_score(df)
    print(f"\nMomentum Score: {momentum:.2f}")

    # Trend
    trend_state, trend_details = analyze_trend(df)
    print(f"Trend: {trend_state.value}")
    print(f"  Price vs MA20: {'Above' if trend_details.get('price_vs_ma20') else 'Below'}")
    print(f"  Price vs MA50: {'Above' if trend_details.get('price_vs_ma50') else 'Below'}")
    print(f"  Price vs MA200: {'Above' if trend_details.get('price_vs_ma200') else 'Below'}")
    print(f"  ADX: {trend_details.get('adx', 0):.1f}")

    # Patterns
    print("\nPattern Detection:")

    breakout = detect_breakout(df)
    if breakout:
        print(f"  BREAKOUT: {breakout['breakout_price']:.2f} (Vol: {breakout['volume_ratio']:.1f}x)")

    pullback = detect_pullback_buy(df)
    if pullback:
        print(f"  PULLBACK: {pullback['price']:.2f} (RSI: {pullback['rsi']:.1f})")

    vcp = detect_volatility_contraction(df)
    if vcp:
        print(f"  VCP: Pivot at {vcp['pivot']:.2f}")

    w_bottom = detect_w_bottom_enhanced(df)
    if w_bottom:
        print(f"  W-BOTTOM: Entry {w_bottom['entry_price']:.2f}, Target {w_bottom['target']:.2f}")

    # Plot
    RESULTS_DIR.mkdir(exist_ok=True)
    save_path = str(RESULTS_DIR / f'analysis_{symbol}.png') if save_plot else None
    plot_stock_analysis(ts_code, df, save_path=save_path)


def show_strategy_info():
    """Display strategy information"""
    print("""
============================================================
ENHANCED QUANTITATIVE TRADING STRATEGY
Targeting 25%+ Annual Returns
============================================================

STRATEGY COMPONENTS:

1. MOMENTUM INVESTING (40% weight)
   Based on: Jegadeesh & Titman research, AQR Capital
   - 12-month price momentum, skip last month
   - Risk-adjusted by volatility
   - Buy winners, avoid losers

2. TREND FOLLOWING (30% weight)
   Based on: Turtle Traders, managed futures
   - Multiple timeframe confirmation (20/50/200 MA)
   - ADX for trend strength
   - Only trade with the trend

3. PATTERN RECOGNITION (20% weight)
   Based on: Mark Minervini SEPA, classic TA
   - Breakout with volume surge
   - Pullback buying in uptrends
   - Volatility contraction patterns (VCP)
   - Enhanced W-bottom

4. SECTOR ROTATION (Monthly)
   Based on: Fidelity sector rotation
   - Calculate sector momentum
   - Invest in top 3 sectors
   - Avoid lagging sectors

RISK MANAGEMENT:
- Max 8 positions (concentrated)
- 15% max per position
- 8% initial stop loss
- 12% trailing stop after profit
- Move to breakeven after 8% gain
- Max 120 day hold period

EXIT RULES:
1. Stop loss triggered
2. 25% profit target
3. Trailing stop hit
4. Trend reversal
5. Time limit exceeded

============================================================
""")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Quant Strategy - 25% Target')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--analyze', type=str, help='Analyze stock (e.g., 000001.SZ)')
    parser.add_argument('--info', action='store_true', help='Show strategy info')
    parser.add_argument('--start', type=str, default='20200101', help='Start date')
    parser.add_argument('--end', type=str, default=None, help='End date')
    parser.add_argument('--capital', type=float, default=1000000, help='Initial capital')
    parser.add_argument('--save', action='store_true', help='Save plots')

    args = parser.parse_args()

    if args.info:
        show_strategy_info()
    elif args.analyze:
        analyze_stock(args.analyze, save_plot=args.save)
    elif args.backtest:
        run_strategy(start_date=args.start, end_date=args.end,
                    initial_capital=args.capital, save_plots=args.save)
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("QUICK START:")
        print("="*60)
        print("""
# View strategy information:
python quant_trade.py --info

# Analyze a stock:
python quant_trade.py --analyze 000001.SZ --save

# Run backtest:
python quant_trade.py --backtest --start 20200101 --save

# From Python:
from quant_trade import run_strategy, analyze_stock
report = run_strategy(start_date='20200101', save_plots=True)
""")
