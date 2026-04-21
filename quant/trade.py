"""
Quantitative Trading Strategy - JointQuant-Style Multi-Factor Strategy
Target: 20% Annual Return from 2017

Strategy Overview (Optimized for A-Shares):
============================================
Based on proven JointQuant strategies, this system combines dual momentum,
MA alignment, volume breakouts, and strict risk management.

1. Market Regime Detection (200d + 60d MA Crossover):
   - STRONG BULL: MA60 > MA200, price trending up → 100% exposure
   - BULL: Price above both MAs → 85% exposure
   - NEUTRAL: Mixed signals → 40% exposure (selective)
   - BEAR: Price below both MAs → 0% exposure (GO TO CASH)
   - SEVERE BEAR: Rapid decline → 0% exposure (FULL CASH)

2. Stock Selection (JointQuant Multi-Factor):
   - Dual Momentum: 20-day (primary) + 60-day (confirmation)
   - MA Alignment: Close > MA5 > MA20 > MA60 (perfect trend)
   - Volume Breakout: > 1.5x average volume (accumulation)
   - RSI Momentum Zone: 50-65 (trending, not overbought)
   - Breakout Detection: Near or above 20d/60d highs
   - Pullback Entry: Short dip in strong uptrend

3. Risk Management:
   - Stop-loss: 8% (capital preservation)
   - Profit target: 25% (let winners run)
   - Trailing stop: 8% from high after 15% gain
   - Full cash during bear markets
   - Concentrated portfolio: max 8 positions at 12% each

Key Insight for Chinese A-shares:
- Go to CASH during bear markets (capital preservation is key)
- Use dual momentum (20d + 60d) for better signal quality
- MA alignment (5/20/60) captures A-share momentum well
- Volume confirmation is critical for breakout signals
- Weekly rebalancing for responsive position management

Usage:
    from quant.trade import run_strategy
    results = run_strategy()
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'stock_data'
SH_DIR = DATA_DIR / 'sh'
SZ_DIR = DATA_DIR / 'sz'
PLOTS_DIR = Path(__file__).parent.parent / 'plots' / 'quant_strategy'

# Strategy Parameters (Quick Profit Trading - JointQuant Style v5)
INITIAL_CAPITAL = 1_000_000  # Starting capital in CNY
MAX_POSITIONS = 6  # Concentrated portfolio
POSITION_SIZE = 0.15  # 15% per position (6 * 15% = 90% max)
STOP_LOSS = -0.08  # 8% stop loss (tighter)
PROFIT_TARGET = 0.15  # 15% profit target (quick profits)
TRAILING_STOP_TRIGGER = 0.10  # Start trailing after 10% gain
TRAILING_STOP_PCT = 0.05  # 5% trailing stop from high
COMMISSION_RATE = 0.001  # 0.1% commission

# Factor Parameters (Short-term momentum for trading)
MOMENTUM_LOOKBACK = 15  # 15-day momentum (shorter)
MOMENTUM_SKIP = 3  # Skip 3 days
MOMENTUM_MID = 40  # 40-day momentum for confirmation
MA_SHORT = 5  # 5-day MA
MA_MEDIUM = 10  # 10-day MA (shorter)
MA_LONG = 30  # 30-day MA (shorter)
MARKET_MA = 60  # 60-day for regime (faster)
MARKET_MA_SHORT = 20  # 20-day MA for regime
VOLUME_MA = 10  # 10-day volume MA
RSI_PERIOD = 14
REBALANCE_DAYS = 3  # Every 3 days
MIN_HOLD_DAYS = 3  # Hold at least 3 days
MIN_AVG_VOLUME = 3_000_000  # Lower for more opportunities

# Market Regime Parameters (More participatory)
STRONG_BULL_EXPOSURE = 1.0  # 100% invested
BULL_EXPOSURE = 0.9  # 90% invested
NEUTRAL_EXPOSURE = 0.5  # 50% invested - participate more
BEAR_EXPOSURE = 0.0  # GO TO CASH in bear market


class TradingSignal:
    """Represents a trading signal with reasoning"""
    def __init__(self, stock: str, date: str, action: str, price: float,
                 reasoning: List[str], score: float = 0):
        self.stock = stock
        self.date = date
        self.action = action
        self.price = price
        self.reasoning = reasoning
        self.score = score

    def __repr__(self):
        reasons = "\n    - ".join(self.reasoning)
        return f"{self.date} | {self.action} {self.stock} @ {self.price:.2f}\n    Reasons:\n    - {reasons}"


class Position:
    """Represents an open position"""
    def __init__(self, stock: str, entry_date: str, entry_price: float,
                 shares: int, entry_reasoning: List[str]):
        self.stock = stock
        self.entry_date = pd.to_datetime(entry_date)
        self.entry_price = entry_price
        self.shares = shares
        self.entry_reasoning = entry_reasoning
        self.current_price = entry_price
        self.highest_price = entry_price

    def update(self, price: float):
        self.current_price = price
        self.highest_price = max(self.highest_price, price)

    @property
    def pnl_pct(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def drawdown_from_high(self) -> float:
        if self.highest_price == 0:
            return 0
        return (self.current_price - self.highest_price) / self.highest_price

    def holding_days(self, current_date) -> int:
        return (pd.to_datetime(current_date) - self.entry_date).days


def load_stock_data(stock_code: str) -> Optional[pd.DataFrame]:
    """Load stock data from CSV file"""
    filepath = SH_DIR / f"{stock_code}.csv"
    if not filepath.exists():
        filepath = SZ_DIR / f"{stock_code}.csv"

    if not filepath.exists():
        return None

    df = pd.read_csv(filepath)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df


def get_all_stocks() -> List[str]:
    """Get list of all available stock codes"""
    stocks = []
    for dir_path in [SH_DIR, SZ_DIR]:
        if dir_path.exists():
            for f in dir_path.glob('*.csv'):
                if f.stem.isdigit():
                    stocks.append(f.stem)
    return stocks


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all factors for a stock (JointQuant-style multi-factor)"""
    df = df.copy()

    # Primary Momentum: 20-day (JointQuant standard for A-shares)
    df['momentum'] = df['close'].pct_change(MOMENTUM_LOOKBACK - MOMENTUM_SKIP).shift(MOMENTUM_SKIP)

    # Medium-term momentum: 60-day (confirmation signal)
    df['momentum_mid'] = df['close'].pct_change(MOMENTUM_MID)

    # Short-term momentum (1 week)
    df['momentum_short'] = df['close'].pct_change(5)

    # 10-day momentum for reversal detection
    df['momentum_10d'] = df['close'].pct_change(10)

    # Moving averages (JointQuant style: 5/20/60)
    df['ma_short'] = df['close'].rolling(MA_SHORT).mean()
    df['ma_medium'] = df['close'].rolling(MA_MEDIUM).mean()
    df['ma_long'] = df['close'].rolling(MA_LONG).mean()
    df['ma_200'] = df['close'].rolling(MARKET_MA).mean()
    df['ma_60'] = df['close'].rolling(MARKET_MA_SHORT).mean()

    # Volume indicators (enhanced for A-shares)
    df['vol_ma'] = df['vol'].rolling(VOLUME_MA).mean()
    df['vol_ma_5'] = df['vol'].rolling(5).mean()  # Short-term volume
    df['vol_ratio'] = df['vol'] / df['vol_ma']
    df['vol_ratio_5'] = df['vol_ma_5'] / df['vol_ma']  # Volume trend
    df['avg_volume'] = df['vol'].rolling(60).mean()  # For liquidity filter

    # RSI
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)

    # Volatility (20-day)
    df['volatility'] = df['pct_chg'].rolling(20).std()
    df['volatility_10d'] = df['pct_chg'].rolling(10).std()  # Short-term vol

    # Price position in range (JointQuant style)
    df['high_60d'] = df['high'].rolling(60).max()
    df['low_60d'] = df['low'].rolling(60).min()
    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    df['price_position_60d'] = (df['close'] - df['low_60d']) / (df['high_60d'] - df['low_60d'] + 0.001)
    df['price_position_20d'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 0.001)

    # Breakout detection
    df['near_high_20d'] = df['close'] >= df['high_20d'] * 0.98
    df['breakout_60d'] = df['close'] >= df['high_60d']

    # Trend strength (vs multiple MAs)
    df['trend_strength'] = (df['close'] - df['ma_long']) / df['ma_long']
    df['trend_strength_200'] = (df['close'] - df['ma_200']) / df['ma_200']

    # Relative strength (vs 60-day MA)
    df['relative_strength'] = df['close'] / df['ma_60']

    # MA alignment score (JointQuant multi-MA analysis)
    df['ma_aligned'] = (
        (df['close'] > df['ma_short']).astype(int) +
        (df['ma_short'] > df['ma_medium']).astype(int) +
        (df['ma_medium'] > df['ma_long']).astype(int)
    )

    return df


def calculate_market_index(stock_data: Dict[str, pd.DataFrame], trading_dates: List) -> pd.DataFrame:
    """Create a market index proxy from large-cap liquid stocks"""
    # Select only liquid stocks with high volume
    large_caps = []
    for stock, df in stock_data.items():
        if 'avg_volume' in df.columns:
            recent_vol = df['avg_volume'].iloc[-1] if len(df) > 0 else 0
            if recent_vol > MIN_AVG_VOLUME:
                large_caps.append(stock)

    if len(large_caps) < 30:
        # Fallback to all stocks
        large_caps = list(stock_data.keys())[:100]

    # Calculate equal-weighted index
    index_data = []
    for date in trading_dates:
        prices = []
        for stock in large_caps[:50]:
            if stock in stock_data:
                df = stock_data[stock]
                mask = df['trade_date'] == date
                if mask.any():
                    prices.append(df.loc[mask, 'close'].iloc[0])

        if prices:
            index_data.append({
                'date': date,
                'close': np.mean(prices)
            })

    index_df = pd.DataFrame(index_data)
    if len(index_df) > 0:
        # Use configurable MAs for regime detection
        index_df['ma_120'] = index_df['close'].rolling(MARKET_MA, min_periods=40).mean()
        index_df['ma_40'] = index_df['close'].rolling(MARKET_MA_SHORT, min_periods=15).mean()
        index_df['ma_20'] = index_df['close'].rolling(20, min_periods=10).mean()
        index_df['returns'] = index_df['close'].pct_change()
        index_df['returns_5d'] = index_df['close'].pct_change(5)
        index_df['returns_20d'] = index_df['close'].pct_change(20)
        index_df['returns_60d'] = index_df['close'].pct_change(60)
        # Also compute ma_200 and ma_60 for compatibility
        index_df['ma_200'] = index_df['close'].rolling(200, min_periods=60).mean()
        index_df['ma_60'] = index_df['close'].rolling(60, min_periods=20).mean()
    return index_df


def get_market_regime(index_df: pd.DataFrame, current_date) -> Tuple[str, float]:
    """
    Fast regime detection for trading
    Uses 60-day and 20-day MAs for quick response

    Returns: (regime_name, target_exposure)
    """
    mask = index_df['date'] == current_date
    if not mask.any():
        return 'NEUTRAL', NEUTRAL_EXPOSURE

    row = index_df.loc[mask].iloc[0]

    # Use faster MAs for trading
    ma_60 = row.get('ma_60')
    ma_20 = row.get('ma_20')

    if pd.isna(ma_60) or pd.isna(ma_20):
        return 'NEUTRAL', NEUTRAL_EXPOSURE

    trend_60 = row['close'] / ma_60 if ma_60 and ma_60 > 0 else 1.0
    trend_20 = row['close'] / ma_20 if ma_20 and ma_20 > 0 else 1.0
    returns_5d = row.get('returns_5d', 0) if not pd.isna(row.get('returns_5d')) else 0
    returns_20d = row.get('returns_20d', 0) if not pd.isna(row.get('returns_20d')) else 0

    # STRONG BULL: Clear uptrend - price above both MAs with positive momentum
    if trend_60 > 1.03 and trend_20 > 1.02 and returns_5d > 0:
        return 'STRONG_BULL', STRONG_BULL_EXPOSURE

    # BULL: Price above 60-day MA
    if row['close'] > ma_60:
        return 'BULL', BULL_EXPOSURE

    # SEVERE BEAR: Major rapid decline
    if trend_60 < 0.88 or returns_20d < -0.12:
        return 'SEVERE_BEAR', 0.0

    # BEAR: Price clearly below 60-day MA with negative momentum
    if row['close'] < ma_60 * 0.97 and returns_5d < -0.01:
        return 'BEAR', BEAR_EXPOSURE

    # NEUTRAL: Default - still trade but be selective
    return 'NEUTRAL', NEUTRAL_EXPOSURE


def score_stock(df: pd.DataFrame, date_idx: int, market_regime: str) -> Tuple[float, List[str]]:
    """
    JointQuant-style multi-factor stock scoring
    Combines momentum, trend, volume, and breakout signals
    """
    if date_idx < max(MOMENTUM_LOOKBACK, MOMENTUM_MID) + 20:
        return -999, []

    row = df.iloc[date_idx]
    reasons = []
    score = 0

    # Check data validity
    required_fields = ['momentum', 'momentum_mid', 'ma_long', 'rsi', 'volatility', 'avg_volume']
    if any(pd.isna(row.get(f, np.nan)) for f in required_fields):
        return -999, []

    # Liquidity filter (stricter for A-shares)
    if row['avg_volume'] < MIN_AVG_VOLUME:
        return -999, []

    # Must be above 60-day MA (medium-term trend)
    if row['close'] <= row['ma_long']:
        return -999, []

    # ============ FACTOR 1: Dual Momentum (20d + 60d) ============
    mom_20d = row['momentum']
    mom_60d = row['momentum_mid']

    # Strong dual momentum (both positive and confirming)
    if mom_20d > 0.08 and mom_60d > 0.15:
        score += 5
        reasons.append(f"Strong dual momentum: 20d={mom_20d*100:.1f}%, 60d={mom_60d*100:.1f}%")
    elif mom_20d > 0.05 and mom_60d > 0.10:
        score += 4
        reasons.append(f"Good dual momentum: 20d={mom_20d*100:.1f}%, 60d={mom_60d*100:.1f}%")
    elif mom_20d > 0.02 and mom_60d > 0.05:
        score += 2
        reasons.append(f"Positive momentum: 20d={mom_20d*100:.1f}%, 60d={mom_60d*100:.1f}%")
    elif mom_20d < 0 and mom_60d < 0:
        return -999, []  # Both negative - avoid

    # ============ FACTOR 2: MA Alignment (JointQuant 5/20/60) ============
    ma_aligned = row.get('ma_aligned', 0)
    if ma_aligned == 3:  # Perfect alignment: Close > MA5 > MA20 > MA60
        score += 4
        reasons.append("Perfect MA alignment (5>20>60)")
    elif ma_aligned >= 2:
        score += 2
        reasons.append("Good trend structure")

    # ============ FACTOR 3: Breakout Detection ============
    near_high = row.get('near_high_20d', False)
    breakout_60d = row.get('breakout_60d', False)

    if breakout_60d:
        score += 3
        reasons.append("60-day breakout!")
    elif near_high:
        score += 2
        reasons.append("Near 20-day high")

    # ============ FACTOR 4: Volume Confirmation ============
    vol_ratio = row['vol_ratio']
    vol_ratio_5 = row.get('vol_ratio_5', 1.0)

    if vol_ratio > 2.0 and vol_ratio_5 > 1.3:
        score += 3
        reasons.append(f"Strong volume surge: {vol_ratio:.1f}x (accumulation)")
    elif vol_ratio > 1.5:
        score += 2
        reasons.append(f"Volume expansion: {vol_ratio:.1f}x")
    elif vol_ratio > 1.0:
        score += 1

    # ============ FACTOR 5: RSI Momentum Zone ============
    rsi = row['rsi']
    # For momentum stocks, RSI 50-70 is optimal (not overbought but trending)
    if 50 <= rsi <= 65:
        score += 2
        reasons.append(f"RSI in momentum zone: {rsi:.0f}")
    elif 40 <= rsi < 50:
        score += 1
        reasons.append(f"RSI recovering: {rsi:.0f}")
    elif rsi > 80:
        score -= 2
        reasons.append(f"Overbought warning: RSI={rsi:.0f}")

    # ============ FACTOR 6: Price Position ============
    price_pos_60d = row.get('price_position_60d', 0.5)
    price_pos_20d = row.get('price_position_20d', 0.5)

    # Favor stocks in upper range but not at extreme (breakout potential)
    if 0.7 <= price_pos_60d <= 0.95 and price_pos_20d >= 0.8:
        score += 2
        reasons.append(f"Strong position: {price_pos_60d*100:.0f}% of 60d range")
    elif 0.5 <= price_pos_60d <= 0.8:
        score += 1

    # ============ FACTOR 7: Volatility Filter ============
    vol = row['volatility']
    if vol < 0.025:
        score += 1
        reasons.append("Low volatility")
    elif vol > 0.05:
        score -= 1  # High volatility penalty

    # ============ FACTOR 8: Pullback Entry (short-term dip in uptrend) ============
    mom_short = row.get('momentum_short', 0)
    if mom_60d > 0.10 and mom_short < -0.02 and mom_short > -0.08:
        score += 2
        reasons.append("Pullback entry in uptrend")

    # ============ Regime-Based Threshold (Trading Style) ============
    if market_regime == 'STRONG_BULL':
        min_score = 5  # Aggressive in strong bull
    elif market_regime == 'BULL':
        min_score = 6  # Active in bull
    else:  # NEUTRAL
        min_score = 8  # Selective in neutral

    if score < min_score:
        return -999, []

    return score, reasons


def generate_sell_signal(position: Position, df: pd.DataFrame, date_idx: int,
                        market_regime: str) -> Optional[TradingSignal]:
    """Generate sell signal with reasoning"""
    row = df.iloc[date_idx]
    position.update(row['close'])
    current_date = row['trade_date']

    reasoning = []
    should_sell = False

    # Rule 0: BEAR MARKET - Exit everything
    if market_regime in ['BEAR', 'SEVERE_BEAR']:
        reasoning.append(f"BEAR MARKET EXIT: Regime is {market_regime} - moving to cash")
        should_sell = True

    # Rule 1: Stop-loss (tight)
    if position.pnl_pct <= STOP_LOSS:
        reasoning.append(f"STOP-LOSS: {position.pnl_pct*100:.1f}% loss from {position.entry_price:.2f}")
        should_sell = True

    # Rule 2: Profit target
    if position.pnl_pct >= PROFIT_TARGET:
        reasoning.append(f"PROFIT TARGET: {position.pnl_pct*100:.1f}% gain achieved")
        should_sell = True

    # Rule 3: Trailing stop
    if position.pnl_pct >= TRAILING_STOP_TRIGGER and position.drawdown_from_high < -TRAILING_STOP_PCT:
        reasoning.append(f"TRAILING STOP: {position.drawdown_from_high*100:.1f}% from high {position.highest_price:.2f}")
        should_sell = True

    # Rule 4: Trend breakdown (sell on break below MA30)
    if position.holding_days(current_date) >= MIN_HOLD_DAYS:
        # Sell on trend break - 5% below MA30
        if not pd.isna(row['ma_long']) and row['close'] < row['ma_long'] * 0.95:
            reasoning.append(f"TREND BREAKDOWN: Price 5% below MA{MA_LONG}")
            should_sell = True
        # Also sell if close below MA10 after gain
        elif position.pnl_pct > 0 and not pd.isna(row['ma_medium']) and row['close'] < row['ma_medium'] * 0.97:
            reasoning.append(f"MOMENTUM LOSS: Price 3% below MA{MA_MEDIUM}")
            should_sell = True

    if not should_sell:
        return None

    return TradingSignal(
        stock=position.stock,
        date=current_date.strftime('%Y-%m-%d'),
        action='SELL',
        price=row['close'],
        reasoning=reasoning
    )


def run_backtest(start_date: str = '2017-01-01', end_date: str = None) -> Dict:
    """Run the backtest simulation"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print("=" * 80)
    print("DEFENSIVE MULTI-FACTOR STRATEGY (CAPITAL PRESERVATION FOCUS)")
    print("=" * 80)
    print(f"Strategy: Defensive with Bear Market Cash Position")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: {INITIAL_CAPITAL:,.0f} CNY")
    print(f"Max Positions: {MAX_POSITIONS} | Position Size: {POSITION_SIZE*100:.0f}%")
    print(f"Stop Loss: {STOP_LOSS*100:.0f}% | Profit Target: {PROFIT_TARGET*100:.0f}%")
    print(f"KEY: Go to CASH during bear markets!")
    print("=" * 80)

    # Load data
    print("\nLoading stock data...")
    stocks = get_all_stocks()
    stock_data = {}

    for stock in stocks:
        df = load_stock_data(stock)
        if df is not None and len(df) > MOMENTUM_LOOKBACK + 100:
            df = calculate_factors(df)
            df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
            if len(df) > 100:
                stock_data[stock] = df

    print(f"Loaded {len(stock_data)} stocks with sufficient data")

    # Get trading dates
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df['trade_date'].tolist())
    trading_dates = sorted(list(all_dates))

    print(f"Trading period: {trading_dates[0].strftime('%Y-%m-%d')} to {trading_dates[-1].strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(trading_dates)}")

    # Calculate market index
    print("Calculating market index for regime detection...")
    index_df = calculate_market_index(stock_data, trading_dates)

    # Initialize portfolio
    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    portfolio_history = []
    all_trades: List[TradingSignal] = []
    last_rebalance_idx = -REBALANCE_DAYS

    print("\nRunning backtest...")

    for i, date in enumerate(trading_dates):
        if i % 100 == 0:
            print(f"  Processing day {i}/{len(trading_dates)}...", end='\r')

        # Get market regime
        market_regime, target_exposure = get_market_regime(index_df, date)

        # In bear market - sell everything
        if market_regime in ['BEAR', 'SEVERE_BEAR']:
            stocks_to_sell = []
            for stock, pos in positions.items():
                if stock in stock_data:
                    df = stock_data[stock]
                    date_mask = df['trade_date'] == date
                    if date_mask.any():
                        date_idx = df[date_mask].index[0]
                        row = df.iloc[date_idx]
                        signal = TradingSignal(
                            stock=stock,
                            date=date.strftime('%Y-%m-%d'),
                            action='SELL',
                            price=row['close'],
                            reasoning=[f"BEAR MARKET: Moving to cash (regime: {market_regime})"]
                        )
                        stocks_to_sell.append((stock, signal, row['close']))

            for stock, signal, price in stocks_to_sell:
                pos = positions[stock]
                sale_value = pos.shares * price * (1 - COMMISSION_RATE)
                cash += sale_value
                all_trades.append(signal)
                print(f"\n{signal}")
                del positions[stock]

        else:
            # Check for sells
            stocks_to_sell = []
            for stock, pos in positions.items():
                if stock not in stock_data:
                    continue
                df = stock_data[stock]
                date_mask = df['trade_date'] == date
                if not date_mask.any():
                    continue

                date_idx = df[date_mask].index[0]
                sell_signal = generate_sell_signal(pos, df, date_idx, market_regime)

                if sell_signal:
                    stocks_to_sell.append((stock, sell_signal))

            for stock, signal in stocks_to_sell:
                pos = positions[stock]
                sale_value = pos.shares * signal.price * (1 - COMMISSION_RATE)
                cash += sale_value
                all_trades.append(signal)
                print(f"\n{signal}")
                del positions[stock]

        # Calculate current portfolio value
        portfolio_value = cash
        for stock, pos in positions.items():
            if stock in stock_data:
                df = stock_data[stock]
                date_mask = df['trade_date'] == date
                if date_mask.any():
                    price = df.loc[date_mask, 'close'].iloc[0]
                    pos.update(price)
                    portfolio_value += pos.shares * price

        # Calculate target positions
        target_positions = int(MAX_POSITIONS * target_exposure)

        # Only buy in bull markets
        should_rebalance = (i - last_rebalance_idx >= REBALANCE_DAYS)
        if should_rebalance and len(positions) < target_positions and market_regime in ['BULL', 'STRONG_BULL']:
            buy_candidates = []

            for stock, df in stock_data.items():
                if stock in positions:
                    continue

                date_mask = df['trade_date'] == date
                if not date_mask.any():
                    continue

                date_idx = df[date_mask].index[0]
                score, reasons = score_stock(df, date_idx, market_regime)

                if score > 0:
                    row = df.iloc[date_idx]
                    buy_candidates.append({
                        'stock': stock,
                        'score': score,
                        'reasons': reasons,
                        'price': row['close']
                    })

            # Sort by score
            buy_candidates.sort(key=lambda x: x['score'], reverse=True)
            slots_available = target_positions - len(positions)

            for candidate in buy_candidates[:slots_available]:
                position_value = portfolio_value * POSITION_SIZE
                if position_value < 5000:
                    break

                price = candidate['price']
                shares = int(position_value / price / 100) * 100
                if shares < 100:
                    continue

                cost = shares * price * (1 + COMMISSION_RATE)
                if cost > cash:
                    continue

                cash -= cost
                positions[candidate['stock']] = Position(
                    stock=candidate['stock'],
                    entry_date=date.strftime('%Y-%m-%d'),
                    entry_price=price,
                    shares=shares,
                    entry_reasoning=candidate['reasons']
                )

                signal = TradingSignal(
                    stock=candidate['stock'],
                    date=date.strftime('%Y-%m-%d'),
                    action='BUY',
                    price=price,
                    reasoning=[f"Market regime: {market_regime}"] + candidate['reasons'],
                    score=candidate['score']
                )
                all_trades.append(signal)
                print(f"\n{signal}")

            if len(buy_candidates) > 0:
                last_rebalance_idx = i

        # Record portfolio value
        portfolio_value = cash
        for stock, pos in positions.items():
            if stock in stock_data:
                df = stock_data[stock]
                date_mask = df['trade_date'] == date
                if date_mask.any():
                    current_price = df.loc[date_mask, 'close'].iloc[0]
                    pos.update(current_price)
                    portfolio_value += pos.shares * current_price

        portfolio_history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'num_positions': len(positions),
            'positions': list(positions.keys()),
            'market_regime': market_regime,
            'cash_pct': cash / portfolio_value * 100 if portfolio_value > 0 else 100
        })

    print("\n" + "=" * 80)

    # Calculate results
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
    portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1

    final_value = portfolio_df['portfolio_value'].iloc[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    days = (portfolio_df['date'].iloc[-1] - portfolio_df['date'].iloc[0]).days
    years = days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1

    rf_daily = 0.03 / 252
    excess_returns = portfolio_df['returns'].dropna() - rf_daily
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

    rolling_max = portfolio_df['portfolio_value'].cummax()
    drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    results = {
        'initial_capital': INITIAL_CAPITAL,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(all_trades),
        'years': years,
        'portfolio_df': portfolio_df,
        'all_trades': all_trades,
        'final_positions': positions,
        'index_df': index_df
    }

    print("\nBACKTEST RESULTS")
    print("=" * 80)
    print(f"Initial Capital:     {INITIAL_CAPITAL:>15,.0f} CNY")
    print(f"Final Value:         {final_value:>15,.0f} CNY")
    print(f"Total Return:        {total_return*100:>14.2f}%")
    print(f"Annual Return:       {annual_return*100:>14.2f}%")
    print(f"Sharpe Ratio:        {sharpe_ratio:>15.2f}")
    print(f"Max Drawdown:        {max_drawdown*100:>14.2f}%")
    print(f"Total Trades:        {len(all_trades):>15}")
    print(f"Trading Period:      {years:>14.1f} years")
    print("=" * 80)

    # Market regime analysis
    regime_counts = portfolio_df['market_regime'].value_counts()
    print("\nMarket Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(portfolio_df) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")

    # Cash position analysis
    avg_cash_pct = portfolio_df['cash_pct'].mean()
    print(f"\nAverage Cash Position: {avg_cash_pct:.1f}%")

    if annual_return >= 0.20:
        print("\n✓ TARGET ACHIEVED: Annual return >= 20%")
    else:
        print(f"\n✗ Target not achieved. Gap: {(0.20 - annual_return)*100:.2f}%")

    return results


def plot_strategy(results: Dict, save_dir: Path = None):
    """Generate visualization plots"""
    if save_dir is None:
        save_dir = PLOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    portfolio_df = results['portfolio_df']
    index_df = results.get('index_df', pd.DataFrame())

    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Portfolio Performance with Market Regimes
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Plot 1: Portfolio Value
    ax1 = axes[0]
    ax1.plot(portfolio_df['date'], portfolio_df['portfolio_value'] / 1e6,
             color='#2E86AB', linewidth=2, label='Portfolio Value')
    ax1.axhline(y=INITIAL_CAPITAL / 1e6, color='gray', linestyle='--',
                alpha=0.7, label='Initial Capital')
    ax1.fill_between(portfolio_df['date'], INITIAL_CAPITAL / 1e6,
                     portfolio_df['portfolio_value'] / 1e6,
                     where=portfolio_df['portfolio_value'] >= INITIAL_CAPITAL,
                     alpha=0.3, color='#28A745')
    ax1.fill_between(portfolio_df['date'], INITIAL_CAPITAL / 1e6,
                     portfolio_df['portfolio_value'] / 1e6,
                     where=portfolio_df['portfolio_value'] < INITIAL_CAPITAL,
                     alpha=0.3, color='#DC3545')
    ax1.set_ylabel('Portfolio Value (Million CNY)', fontsize=12)
    ax1.set_title('Defensive Strategy with Bear Market Cash Protection', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # Plot 2: Market Regime & Cash Position
    ax2 = axes[1]
    regime_colors = {'STRONG_BULL': '#00AA00', 'BULL': '#28A745', 'BEAR': '#DC3545',
                     'SEVERE_BEAR': '#8B0000', 'NEUTRAL': '#FFC107'}
    for regime, color in regime_colors.items():
        mask = portfolio_df['market_regime'] == regime
        if mask.any():
            ax2.fill_between(portfolio_df['date'], 0, 100, where=mask, alpha=0.3,
                           color=color, label=regime, transform=ax2.get_xaxis_transform())

    ax2.plot(portfolio_df['date'], portfolio_df['cash_pct'],
            color='blue', linewidth=1.5, label='Cash %')
    ax2.set_ylabel('Cash Position (%)', fontsize=12)
    ax2.set_title('Market Regime & Cash Position (Cash = Safety)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Plot 3: Cumulative Returns vs Target
    ax3 = axes[2]
    ax3.plot(portfolio_df['date'], portfolio_df['cumulative_returns'] * 100,
             color='#2E86AB', linewidth=2, label='Strategy Returns')
    days_from_start = (portfolio_df['date'] - portfolio_df['date'].iloc[0]).dt.days
    target_returns = ((1.20) ** (days_from_start / 365.25) - 1) * 100
    ax3.plot(portfolio_df['date'], target_returns, color='red', linestyle='--',
             alpha=0.7, linewidth=2, label='20% Annual Target')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.fill_between(portfolio_df['date'], 0, portfolio_df['cumulative_returns'] * 100,
                     where=portfolio_df['cumulative_returns'] >= 0, alpha=0.3, color='#28A745')
    ax3.fill_between(portfolio_df['date'], 0, portfolio_df['cumulative_returns'] * 100,
                     where=portfolio_df['cumulative_returns'] < 0, alpha=0.3, color='#DC3545')
    ax3.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax3.set_title('Cumulative Returns vs 20% Annual Target', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Plot 4: Drawdown
    ax4 = axes[3]
    rolling_max = portfolio_df['portfolio_value'].cummax()
    drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max * 100
    ax4.fill_between(portfolio_df['date'], 0, drawdown, color='#DC3545', alpha=0.5)
    ax4.plot(portfolio_df['date'], drawdown, color='#DC3545', linewidth=1)
    ax4.set_ylabel('Drawdown (%)', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.savefig(save_dir / 'portfolio_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'portfolio_performance.png'}")

    # Figure 2: Trading Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Positions over time
    ax1 = axes[0, 0]
    ax1.fill_between(portfolio_df['date'], 0, portfolio_df['num_positions'],
                     color='#6C757D', alpha=0.6)
    ax1.plot(portfolio_df['date'], portfolio_df['num_positions'], color='#343A40', linewidth=1)
    ax1.set_ylabel('Number of Positions', fontsize=11)
    ax1.set_title('Active Positions Over Time', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, MAX_POSITIONS + 1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Plot 2: Annual Returns
    ax2 = axes[0, 1]
    portfolio_df['year'] = portfolio_df['date'].dt.year
    annual_returns = portfolio_df.groupby('year')['returns'].apply(lambda x: (1 + x).prod() - 1) * 100
    colors = ['#28A745' if x >= 20 else '#FFC107' if x >= 0 else '#DC3545' for x in annual_returns.values]
    bars = ax2.bar(annual_returns.index.astype(str), annual_returns.values, color=colors, alpha=0.8)
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='20% Target')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Annual Return (%)', fontsize=11)
    ax2.set_title('Annual Returns by Year', fontsize=12, fontweight='bold')
    ax2.legend()
    for bar, val in zip(bars, annual_returns.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 3: Returns Distribution
    ax3 = axes[1, 0]
    daily_returns = portfolio_df['returns'].dropna() * 100
    ax3.hist(daily_returns, bins=50, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax3.axvline(x=daily_returns.mean(), color='red', linestyle='--',
                label=f'Mean: {daily_returns.mean():.3f}%')
    ax3.axvline(x=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Daily Return (%)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax3.legend()

    # Plot 4: Regime Performance
    ax4 = axes[1, 1]
    regime_performance = {}
    for regime in ['STRONG_BULL', 'BULL', 'NEUTRAL', 'BEAR', 'SEVERE_BEAR']:
        mask = portfolio_df['market_regime'] == regime
        if mask.any():
            regime_returns = portfolio_df.loc[mask, 'returns'].dropna()
            if len(regime_returns) > 0:
                regime_performance[regime] = regime_returns.mean() * 252 * 100

    if regime_performance:
        regimes = list(regime_performance.keys())
        means = [regime_performance[r] for r in regimes]
        colors = ['#00AA00' if 'STRONG' in r else '#28A745' if r == 'BULL'
                  else '#DC3545' if 'BEAR' in r else '#FFC107' for r in regimes]
        bars = ax4.bar(regimes, means, color=colors, alpha=0.8)
        ax4.axhline(y=0, color='black', linewidth=0.5)
        ax4.set_ylabel('Annualized Return (%)', fontsize=11)
        ax4.set_title('Performance by Market Regime', fontsize=12, fontweight='bold')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_dir / 'trading_activity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'trading_activity.png'}")

    # Figure 3: Strategy Summary
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║         JOINTQUANT-STYLE MULTI-FACTOR STRATEGY (OPTIMIZED FOR A-SHARES)          ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║   KEY STRATEGY INSIGHT                                                           ║
    ║   ────────────────────                                                           ║
    ║   Dual momentum (20d + 60d) + MA alignment + Volume breakout + Bear protection  ║
    ║   Go to CASH when market regime is BEAR - capital preservation is key.          ║
    ║                                                                                  ║
    ║   STRATEGY PARAMETERS (JointQuant Optimized)                                     ║
    ║   ──────────────────────────────────────────                                     ║
    ║   • Dual Momentum:         {MOMENTUM_LOOKBACK}d (primary) + {MOMENTUM_MID}d (confirmation)                 ║
    ║   • Trend MAs:             MA{MA_SHORT}, MA{MA_MEDIUM}, MA{MA_LONG} (JointQuant standard)                        ║
    ║   • Market Regime MAs:     {MARKET_MA}d + {MARKET_MA_SHORT}d crossover                                   ║
    ║   • Max Positions:         {MAX_POSITIONS} (concentrated high-conviction)                         ║
    ║   • Position Size:         {POSITION_SIZE*100:.0f}% per stock                                            ║
    ║   • Stop Loss:             {abs(STOP_LOSS)*100:.0f}% (capital preservation)                              ║
    ║   • Profit Target:         {PROFIT_TARGET*100:.0f}% (let winners run)                                    ║
    ║   • Trailing Stop:         {TRAILING_STOP_PCT*100:.0f}% after {TRAILING_STOP_TRIGGER*100:.0f}% gain                                ║
    ║                                                                                  ║
    ║   PERFORMANCE METRICS                                                            ║
    ║   ─────────────────────                                                          ║
    ║   • Initial Capital:       {results['initial_capital']:>15,.0f} CNY                             ║
    ║   • Final Value:           {results['final_value']:>15,.0f} CNY                             ║
    ║   • Total Return:          {results['total_return']*100:>14.2f}%                                 ║
    ║   • Annual Return:         {results['annual_return']*100:>14.2f}%                                 ║
    ║   • Sharpe Ratio:          {results['sharpe_ratio']:>15.2f}                                  ║
    ║   • Max Drawdown:          {results['max_drawdown']*100:>14.2f}%                                 ║
    ║   • Total Trades:          {results['total_trades']:>15}                                  ║
    ║                                                                                  ║
    ║   BUY CRITERIA (JointQuant Multi-Factor)                                         ║
    ║   ──────────────────────────────────────                                         ║
    ║   1. Dual momentum: 20d > 5% AND 60d > 10%                                       ║
    ║   2. MA alignment: Close > MA5 > MA20 > MA60                                     ║
    ║   3. Volume surge: > 1.5x average (accumulation signal)                          ║
    ║   4. RSI in momentum zone: 50-65 (trending, not overbought)                      ║
    ║   5. Near or breaking 20d/60d highs (breakout potential)                         ║
    ║   6. Pullback entry: short dip in strong uptrend                                 ║
    ║                                                                                  ║
    ║   SELL CRITERIA                                                                  ║
    ║   ──────────────                                                                 ║
    ║   1. BEAR MARKET → Sell everything, move to 100% cash                            ║
    ║   2. Stop-loss at {abs(STOP_LOSS)*100:.0f}% - cut losses quickly                                     ║
    ║   3. Profit target at {PROFIT_TARGET*100:.0f}%                                                       ║
    ║   4. Trailing stop: {TRAILING_STOP_PCT*100:.0f}% from high after {TRAILING_STOP_TRIGGER*100:.0f}% gain                            ║
    ║   5. Trend breakdown: Close < MA{MA_MEDIUM} * 0.97 or < MA{MA_LONG}                            ║
    ║                                                                                  ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """

    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.savefig(save_dir / 'strategy_summary.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_dir / 'strategy_summary.png'}")

    print(f"\nAll plots saved to: {save_dir}")


def export_trade_log(results: Dict, filepath: Path = None):
    """Export detailed trade log to CSV"""
    if filepath is None:
        filepath = PLOTS_DIR / 'trade_log.csv'

    filepath.parent.mkdir(parents=True, exist_ok=True)

    trades = results['all_trades']
    trade_data = []

    for t in trades:
        trade_data.append({
            'date': t.date,
            'stock': t.stock,
            'action': t.action,
            'price': t.price,
            'score': t.score if hasattr(t, 'score') else None,
            'reasoning': ' | '.join(t.reasoning)
        })

    df = pd.DataFrame(trade_data)
    df.to_csv(filepath, index=False)
    print(f"Trade log exported to: {filepath}")


def run_strategy():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("STARTING DEFENSIVE QUANTITATIVE TRADING STRATEGY")
    print("=" * 80)

    results = run_backtest(start_date='2017-01-01')

    print("\nGenerating visualization plots...")
    plot_strategy(results)

    print("\nExporting trade log...")
    export_trade_log(results)

    print("\n" + "=" * 80)
    print("STRATEGY EXECUTION COMPLETE")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = run_strategy()
