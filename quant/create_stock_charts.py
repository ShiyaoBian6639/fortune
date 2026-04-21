"""
Create individual stock price charts for Trump visit analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from datetime import datetime

# Configure matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = Path(__file__).parent.parent / 'stock_data'
PLOTS_DIR = Path(__file__).parent.parent / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# Load stock names
stock_list = pd.read_csv(DATA_DIR / 'stock_list.csv')
stock_dict = dict(zip(stock_list['ts_code'], stock_list['name']))

# Key stocks to plot (representing different sectors from the trade deals)
KEY_STOCKS = [
    # Semiconductor sector (Qualcomm deals)
    ('300655.SZ', 'Semiconductor'),  # 晶瑞电材
    ('002049.SZ', 'Semiconductor'),  # 紫光国微

    # Energy sector (LNG, Oil deals)
    ('002221.SZ', 'Energy'),  # 东华能源
    ('601088.SH', 'Energy'),  # 中国神华

    # Aviation sector (Boeing deals)
    ('601021.SH', 'Aviation'),  # 春秋航空
    ('600029.SH', 'Aviation'),  # 南方航空

    # Equipment/Manufacturing (Caterpillar, GE)
    ('600031.SH', 'Manufacturing'),  # 三一重工
    ('000425.SZ', 'Manufacturing'),  # 徐工机械

    # Top gainers
    ('002409.SZ', 'Top Gainer'),  # 雅克科技
    ('603501.SH', 'Top Gainer'),  # 豪威集团
]

# Date ranges
CHART_START = 20171020  # 3 weeks before
CHART_END = 20171130    # 3 weeks after
EVENT_START = 20171108
EVENT_END = 20171110


def load_stock_data(ts_code: str, start_date: int, end_date: int) -> pd.DataFrame:
    """Load stock data for a given date range."""
    symbol = ts_code.split('.')[0]
    exchange = ts_code.split('.')[1].lower()
    filepath = DATA_DIR / exchange / f"{symbol}.csv"

    if not filepath.exists():
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df['trade_date'] = df['trade_date'].astype(int)
    df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
    df = df.sort_values('trade_date')

    # Convert trade_date to datetime
    df['date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')

    return df


def plot_candlestick(ts_code: str, sector: str, save_path: str):
    """Create a candlestick-style chart for a stock."""
    df = load_stock_data(ts_code, CHART_START, CHART_END)

    if df.empty:
        print(f"No data for {ts_code}")
        return

    # Reset index for proper plotting
    df = df.reset_index(drop=True)

    name = stock_dict.get(ts_code, ts_code)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True)

    # Price chart with OHLC
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]

    # Plot candlesticks
    width = 0.6
    width2 = 0.1

    # Up days (green)
    ax1.bar(up.index, up['close'] - up['open'], width, bottom=up['open'], color='green', alpha=0.8)
    ax1.bar(up.index, up['high'] - up['close'], width2, bottom=up['close'], color='green', alpha=0.8)
    ax1.bar(up.index, up['low'] - up['open'], width2, bottom=up['open'], color='green', alpha=0.8)

    # Down days (red)
    ax1.bar(down.index, down['close'] - down['open'], width, bottom=down['open'], color='red', alpha=0.8)
    ax1.bar(down.index, down['high'] - down['open'], width2, bottom=down['open'], color='red', alpha=0.8)
    ax1.bar(down.index, down['low'] - down['close'], width2, bottom=down['close'], color='red', alpha=0.8)

    # Mark event period
    event_df = df[(df['trade_date'] >= EVENT_START) & (df['trade_date'] <= EVENT_END)]
    if not event_df.empty:
        ax1.axvspan(event_df.index.min() - 0.5, event_df.index.max() + 0.5,
                   alpha=0.2, color='yellow', label='Trump Visit (Nov 8-10)')

    # Add price annotations
    pre_event = df[df['trade_date'] < EVENT_START]
    post_event = df[df['trade_date'] > EVENT_END]

    if not pre_event.empty and not event_df.empty:
        pre_close = pre_event.iloc[-1]['close']
        event_close = event_df.iloc[-1]['close']
        change_pct = ((event_close - pre_close) / pre_close) * 100

        ax1.annotate(f'Pre-event: {pre_close:.2f}',
                    xy=(pre_event.index[-1], pre_close),
                    xytext=(-50, 20), textcoords='offset points',
                    fontsize=10, color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue'))

        ax1.annotate(f'During event: {event_close:.2f}\n({change_pct:+.2f}%)',
                    xy=(event_df.index[-1], event_close),
                    xytext=(30, 20), textcoords='offset points',
                    fontsize=10, color='green' if change_pct > 0 else 'red',
                    arrowprops=dict(arrowstyle='->', color='green' if change_pct > 0 else 'red'))

    ax1.set_ylabel('Price (CNY)', fontsize=12)
    ax1.set_title(f'{ts_code} {name} - {sector} Sector\nTrump China Visit Impact (Nov 8-10, 2017)',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # Volume chart
    colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
    ax2.bar(df.index, df['vol'] / 10000, color=colors, alpha=0.7)
    ax2.set_ylabel('Volume (10k)', fontsize=12)
    ax2.grid(alpha=0.3)

    # Mark event period on volume chart
    if not event_df.empty:
        ax2.axvspan(event_df.index.min() - 0.5, event_df.index.max() + 0.5,
                   alpha=0.2, color='yellow')

    # X-axis labels
    tick_positions = range(0, len(df), max(1, len(df)//10))
    ax2.set_xticks(list(tick_positions))
    ax2.set_xticklabels([df.iloc[i]['trade_date'] for i in tick_positions], rotation=45)
    ax2.set_xlabel('Date', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sector_comparison():
    """Create a comparison chart of key stocks from each sector."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    sectors = {
        'Semiconductor': ['300655.SZ', '002049.SZ'],
        'Energy': ['002221.SZ', '601088.SH'],
        'Aviation': ['601021.SH', '600029.SH'],
        'Manufacturing': ['600031.SH', '000425.SZ'],
        'Top Gainers': ['002409.SZ', '603501.SH'],
    }

    for idx, (sector, stocks) in enumerate(sectors.items()):
        ax = axes[idx]

        for ts_code in stocks:
            df = load_stock_data(ts_code, CHART_START, CHART_END)
            if df.empty:
                continue

            # Reset index and normalize to 100 at start
            df = df.reset_index(drop=True)
            df['normalized'] = 100 * df['close'] / df['close'].iloc[0]
            name = stock_dict.get(ts_code, ts_code)[:8]

            ax.plot(df.index, df['normalized'], label=f'{ts_code[:6]} {name}', linewidth=2)

        # Mark event period
        sample_df = load_stock_data(stocks[0], CHART_START, CHART_END)
        if not sample_df.empty:
            sample_df = sample_df.reset_index(drop=True)
            event_df = sample_df[(sample_df['trade_date'] >= EVENT_START) &
                                 (sample_df['trade_date'] <= EVENT_END)]
            if not event_df.empty:
                ax.axvspan(event_df.index.min(), event_df.index.max(),
                          alpha=0.3, color='yellow')

        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{sector}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylabel('Normalized Price (Base=100)')

    # Hide the 6th subplot
    axes[5].axis('off')

    fig.suptitle('Sector Performance During Trump China Visit (Nov 8-10, 2017)\nNormalized Price Comparison',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'sector_comparison_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'sector_comparison_normalized.png'}")


def plot_market_overview():
    """Create a market overview showing overall market reaction."""
    # Sample 50 random stocks for market overview
    import random

    all_stocks = []
    for exchange in ['sh', 'sz']:
        exchange_dir = DATA_DIR / exchange
        if exchange_dir.exists():
            all_stocks.extend([f.stem + '.' + exchange.upper() for f in exchange_dir.glob('*.csv')])

    random.seed(42)
    sample_stocks = random.sample(all_stocks, min(100, len(all_stocks)))

    fig, ax = plt.subplots(figsize=(14, 8))

    gains = []
    for ts_code in sample_stocks:
        df = load_stock_data(ts_code, CHART_START, CHART_END)
        if df.empty or len(df) < 10:
            continue

        df = df.reset_index(drop=True)
        df['normalized'] = 100 * df['close'] / df['close'].iloc[0]

        # Calculate gain during event
        pre = df[df['trade_date'] < EVENT_START]
        event = df[(df['trade_date'] >= EVENT_START) & (df['trade_date'] <= EVENT_END)]
        if not pre.empty and not event.empty:
            gain = ((event.iloc[-1]['close'] - pre.iloc[-1]['close']) / pre.iloc[-1]['close']) * 100
            gains.append(gain)

            alpha = 0.1 if abs(gain) < 5 else 0.3
            color = 'green' if gain > 0 else 'red'
            ax.plot(df.index, df['normalized'], color=color, alpha=alpha, linewidth=0.5)

    # Plot mean line
    mean_df = None
    for ts_code in sample_stocks[:50]:
        df = load_stock_data(ts_code, CHART_START, CHART_END)
        if df.empty:
            continue
        df['normalized'] = 100 * df['close'] / df['close'].iloc[0]
        if mean_df is None:
            mean_df = df[['normalized']].copy()
            mean_df.columns = ['mean']
            count = 1
        else:
            if len(df) == len(mean_df):
                mean_df['mean'] = mean_df['mean'] + df['normalized'].values
                count += 1

    if mean_df is not None:
        mean_df['mean'] = mean_df['mean'] / count
        ax.plot(mean_df.index, mean_df['mean'], color='blue', linewidth=3, label='Market Average')

    # Mark event period
    sample_df = load_stock_data(sample_stocks[0], CHART_START, CHART_END)
    if not sample_df.empty:
        event_df = sample_df[(sample_df['trade_date'] >= EVENT_START) &
                             (sample_df['trade_date'] <= EVENT_END)]
        if not event_df.empty:
            ax.axvspan(event_df.index.min(), event_df.index.max(),
                      alpha=0.3, color='yellow', label='Trump Visit')

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Normalized Price (Base=100)', fontsize=12)
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_title(f'Market Overview: 100 Stock Sample During Trump China Visit\nGreen = Gainers, Red = Losers',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # Add stats
    stats_text = f"Sample Size: {len(gains)} stocks\n"
    stats_text += f"Gainers: {sum(1 for g in gains if g > 0)} ({100*sum(1 for g in gains if g > 0)/len(gains):.1f}%)\n"
    stats_text += f"Avg Gain: {np.mean(gains):.2f}%"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'market_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'market_overview.png'}")


if __name__ == '__main__':
    print("Generating individual stock charts...")

    # Generate individual stock charts
    for ts_code, sector in KEY_STOCKS:
        symbol = ts_code.split('.')[0]
        save_path = PLOTS_DIR / f'stock_{symbol}_{sector.lower()}.png'
        plot_candlestick(ts_code, sector, str(save_path))

    # Generate sector comparison
    print("\nGenerating sector comparison chart...")
    plot_sector_comparison()

    # Generate market overview
    print("\nGenerating market overview...")
    plot_market_overview()

    print("\nAll charts generated!")
