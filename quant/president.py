"""
Presidential Visit Stock Impact Analysis
- Analyzes stock price movements during significant political events
- Supports analysis of Trump's China visits and other presidential events
- Identifies sector winners based on trade deals and announcements

Usage:
    from quant.president import analyze_event, run

    # Analyze Trump's 2017 China visit
    run('trump_2017')

    # Custom event analysis
    analyze_event(
        event_name="Trump China Visit",
        pre_start=20171030,
        pre_end=20171107,
        event_start=20171108,
        event_end=20171110
    )
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os

# Use non-interactive backend if no display available
if os.environ.get('DISPLAY') is None and os.name != 'nt':
    matplotlib.use('Agg')

# Configure matplotlib for Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Global flag for interactive mode
INTERACTIVE_PLOTS = True

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'stock_data'
STOCK_LIST_FILE = DATA_DIR / 'stock_list.csv'


def load_stock_list() -> Dict[str, str]:
    """Load stock list and return ts_code to name mapping."""
    if STOCK_LIST_FILE.exists():
        df = pd.read_csv(STOCK_LIST_FILE)
        return dict(zip(df['ts_code'], df['name']))
    return {}


def analyze_single_stock(
    filepath: Path,
    pre_start: int,
    pre_end: int,
    event_start: int,
    event_end: int,
    ipo_cutoff: Optional[int] = None
) -> Optional[Dict]:
    """
    Analyze a single stock's performance during an event.

    Args:
        filepath: Path to the stock CSV file
        pre_start: Start date of pre-event period (YYYYMMDD)
        pre_end: End date of pre-event period (YYYYMMDD)
        event_start: Start date of event (YYYYMMDD)
        event_end: End date of event (YYYYMMDD)
        ipo_cutoff: Exclude stocks IPO'd after this date (YYYYMMDD)

    Returns:
        Dictionary with analysis results or None if insufficient data
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty or 'trade_date' not in df.columns:
            return None

        df['trade_date'] = df['trade_date'].astype(int)

        # Filter out recent IPOs
        earliest_date = df['trade_date'].min()
        if ipo_cutoff and earliest_date > ipo_cutoff:
            return None

        # Get data for different periods
        pre_df = df[(df['trade_date'] >= pre_start) & (df['trade_date'] <= pre_end)]
        event_df = df[(df['trade_date'] >= event_start) & (df['trade_date'] <= event_end)]

        if pre_df.empty or event_df.empty or len(pre_df) < 3:
            return None

        # Calculate metrics
        # Sort to ensure we get the correct day (last day before event)
        pre_close = pre_df.sort_values('trade_date', ascending=True)['close'].iloc[-1]
        event_close = event_df.sort_values('trade_date', ascending=True)['close'].iloc[-1]
        event_high = event_df['high'].max()
        event_low = event_df['low'].min()

        price_change_pct = ((event_close - pre_close) / pre_close) * 100

        pre_avg_vol = pre_df['vol'].mean()
        event_avg_vol = event_df['vol'].mean()
        vol_change_pct = ((event_avg_vol - pre_avg_vol) / pre_avg_vol) * 100 if pre_avg_vol > 0 else 0

        ts_code = df['ts_code'].iloc[0]

        return {
            'ts_code': ts_code,
            'symbol': ts_code.split('.')[0],
            'exchange': ts_code.split('.')[1],
            'pre_close': pre_close,
            'event_close': event_close,
            'event_high': event_high,
            'event_low': event_low,
            'price_change_pct': price_change_pct,
            'vol_change_pct': vol_change_pct,
            'earliest_date': earliest_date
        }
    except Exception as e:
        return None


def analyze_event(
    event_name: str,
    pre_start: int,
    pre_end: int,
    event_start: int,
    event_end: int,
    ipo_cutoff: Optional[int] = None,
    save_results: bool = True
) -> pd.DataFrame:
    """
    Analyze stock market impact of a presidential/political event.

    Args:
        event_name: Name of the event for output files
        pre_start: Start of pre-event period (YYYYMMDD)
        pre_end: End of pre-event period (YYYYMMDD)
        event_start: Start of event (YYYYMMDD)
        event_end: End of event (YYYYMMDD)
        ipo_cutoff: Exclude stocks IPO'd after this date
        save_results: Whether to save results to CSV

    Returns:
        DataFrame with analysis results for all stocks
    """
    stock_dict = load_stock_list()
    results = []

    print(f"\n{'='*70}")
    print(f"Analyzing: {event_name}")
    print(f"Pre-event period: {pre_start} to {pre_end}")
    print(f"Event period: {event_start} to {event_end}")
    print(f"{'='*70}\n")

    # Analyze all stocks in sh and sz directories
    for exchange in ['sh', 'sz']:
        exchange_dir = DATA_DIR / exchange
        if not exchange_dir.exists():
            continue

        files = list(exchange_dir.glob('*.csv'))
        for i, filepath in enumerate(files):
            result = analyze_single_stock(
                filepath, pre_start, pre_end, event_start, event_end, ipo_cutoff
            )
            if result:
                result['name'] = stock_dict.get(result['ts_code'], 'Unknown')
                results.append(result)

        print(f"Processed {len(files)} stocks from {exchange.upper()}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No valid data found for analysis.")
        return results_df

    print(f"\nTotal stocks analyzed: {len(results_df)}")

    # Save results
    if save_results:
        output_file = DATA_DIR.parent / f"{event_name.lower().replace(' ', '_')}_analysis.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

    return results_df


def get_top_gainers(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Get top N stocks by price gain."""
    return df.nlargest(n, 'price_change_pct')


def get_top_losers(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Get top N stocks by price loss."""
    return df.nsmallest(n, 'price_change_pct')


def analyze_by_sector(
    df: pd.DataFrame,
    sector_keywords: Dict[str, List[str]],
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Analyze stock performance by sector based on keywords.

    Args:
        df: DataFrame with analysis results (must have 'name' column)
        sector_keywords: Dict mapping sector name to list of keywords
        top_n: Number of top stocks to return per sector

    Returns:
        Dict mapping sector name to DataFrame of top performers
    """
    sector_results = {}

    for sector, keywords in sector_keywords.items():
        pattern = '|'.join(keywords)
        sector_df = df[df['name'].str.contains(pattern, na=False)]

        if not sector_df.empty:
            sector_results[sector] = sector_df.nlargest(top_n, 'price_change_pct')

    return sector_results


def print_results(df: pd.DataFrame, title: str, n: int = 30):
    """Print formatted results table."""
    top = df.nlargest(n, 'price_change_pct')

    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")
    print(f"{'Stock':<12} {'Company Name':<24} {'Pre-Close':>10} {'Event Close':>12} {'Change %':>10}")
    print("-"*90)

    for _, row in top.iterrows():
        name = str(row['name'])[:22] if row['name'] else 'Unknown'
        print(f"{row['ts_code']:<12} {name:<24} {row['pre_close']:>10.2f} {row['event_close']:>12.2f} {row['price_change_pct']:>10.2f}%")


def print_sector_results(sector_results: Dict[str, pd.DataFrame]):
    """Print formatted sector analysis results."""
    print(f"\n{'='*90}")
    print("SECTOR ANALYSIS")
    print(f"{'='*90}")

    for sector, df in sector_results.items():
        if df.empty:
            continue

        print(f"\n### {sector} ###")
        print(f"{'Stock':<12} {'Company Name':<24} {'Change %':>10} {'Vol Chg %':>10}")
        print("-"*70)

        for _, row in df.iterrows():
            name = str(row['name'])[:22] if row['name'] else 'Unknown'
            print(f"{row['ts_code']:<12} {name:<24} {row['price_change_pct']:>10.2f}% {row['vol_change_pct']:>9.1f}%")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_top_gainers(df: pd.DataFrame, title: str = "Top Gainers", n: int = 20,
                     save_path: Optional[str] = None, show: bool = True):
    """
    Plot horizontal bar chart of top gaining stocks.

    Args:
        df: DataFrame with analysis results
        title: Chart title
        n: Number of top stocks to show
        save_path: Optional path to save the figure
        show: Whether to display the plot interactively
    """
    top = df.nlargest(n, 'price_change_pct').sort_values('price_change_pct')

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.RdYlGn(np.linspace(0.6, 0.9, len(top)))

    bars = ax.barh(range(len(top)), top['price_change_pct'], color=colors)

    # Add stock names as y-tick labels
    labels = [f"{row['ts_code']} {str(row['name'])[:10]}" for _, row in top.iterrows()]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top['price_change_pct'])):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=9)

    ax.set_xlabel('Price Change (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show and INTERACTIVE_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_sector_performance(sector_results: Dict[str, pd.DataFrame],
                           title: str = "Sector Performance",
                           save_path: Optional[str] = None):
    """
    Plot sector performance comparison.

    Args:
        sector_results: Dict mapping sector name to DataFrame
        title: Chart title
        save_path: Optional path to save the figure
    """
    # Calculate average gain per sector
    sector_avg = {}
    sector_max = {}
    sector_count = {}

    for sector, df in sector_results.items():
        if not df.empty:
            sector_avg[sector] = df['price_change_pct'].mean()
            sector_max[sector] = df['price_change_pct'].max()
            sector_count[sector] = len(df)

    if not sector_avg:
        print("No sector data to plot")
        return None

    sectors = list(sector_avg.keys())
    avgs = [sector_avg[s] for s in sectors]
    maxs = [sector_max[s] for s in sectors]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(sectors))
    width = 0.35

    bars1 = ax.bar(x - width/2, avgs, width, label='Average Gain', color='steelblue')
    bars2 = ax.bar(x + width/2, maxs, width, label='Max Gain', color='coral')

    ax.set_xlabel('Sector', fontsize=12)
    ax.set_ylabel('Price Change (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('/')[0] for s in sectors], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if INTERACTIVE_PLOTS and not save_path:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_distribution(df: pd.DataFrame, title: str = "Price Change Distribution",
                     save_path: Optional[str] = None):
    """
    Plot histogram of price changes across all stocks.

    Args:
        df: DataFrame with analysis results
        title: Chart title
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    data = df['price_change_pct'].dropna()

    # Color bins by positive/negative
    n, bins, patches = ax1.hist(data, bins=50, edgecolor='white', alpha=0.7)

    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')

    ax1.axvline(x=0, color='black', linewidth=1, linestyle='--')
    ax1.axvline(x=data.mean(), color='blue', linewidth=2, linestyle='-',
                label=f'Mean: {data.mean():.2f}%')
    ax1.axvline(x=data.median(), color='orange', linewidth=2, linestyle='-',
                label=f'Median: {data.median():.2f}%')

    ax1.set_xlabel('Price Change (%)', fontsize=12)
    ax1.set_ylabel('Number of Stocks', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Summary stats box
    stats_text = f"Total Stocks: {len(data)}\n"
    stats_text += f"Gainers: {(data > 0).sum()} ({100*(data > 0).mean():.1f}%)\n"
    stats_text += f"Losers: {(data < 0).sum()} ({100*(data < 0).mean():.1f}%)\n"
    stats_text += f"Mean: {data.mean():.2f}%\n"
    stats_text += f"Std: {data.std():.2f}%\n"
    stats_text += f"Max: {data.max():.2f}%\n"
    stats_text += f"Min: {data.min():.2f}%"

    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Box plot by exchange
    ax2 = axes[1]
    if 'exchange' in df.columns:
        sh_data = df[df['exchange'] == 'SH']['price_change_pct'].dropna()
        sz_data = df[df['exchange'] == 'SZ']['price_change_pct'].dropna()

        bp = ax2.boxplot([sh_data, sz_data], labels=['Shanghai (SH)', 'Shenzhen (SZ)'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')

        ax2.set_ylabel('Price Change (%)', fontsize=12)
        ax2.set_title('Distribution by Exchange', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if INTERACTIVE_PLOTS and not save_path:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_price_volume(df: pd.DataFrame, title: str = "Price Change vs Volume Change",
                     save_path: Optional[str] = None):
    """
    Scatter plot of price change vs volume change.

    Args:
        df: DataFrame with analysis results
        title: Chart title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter out extreme outliers for better visualization
    data = df[(df['vol_change_pct'] > -100) & (df['vol_change_pct'] < 500)]

    # Color by gain/loss
    colors = ['green' if x > 0 else 'red' for x in data['price_change_pct']]

    scatter = ax.scatter(data['vol_change_pct'], data['price_change_pct'],
                        c=colors, alpha=0.5, s=30)

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--')

    ax.set_xlabel('Volume Change (%)', fontsize=12)
    ax.set_ylabel('Price Change (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add quadrant labels
    ax.text(0.95, 0.95, 'Price Up\nVol Up', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='darkgreen')
    ax.text(0.05, 0.95, 'Price Up\nVol Down', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, color='green')
    ax.text(0.95, 0.05, 'Price Down\nVol Up', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='red')
    ax.text(0.05, 0.05, 'Price Down\nVol Down', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10, color='darkred')

    # Annotate top gainers
    top_gainers = data.nlargest(5, 'price_change_pct')
    for _, row in top_gainers.iterrows():
        ax.annotate(row['ts_code'], (row['vol_change_pct'], row['price_change_pct']),
                   fontsize=8, alpha=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if INTERACTIVE_PLOTS and not save_path:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_sector_detail(sector_results: Dict[str, pd.DataFrame], sector_name: str,
                      save_path: Optional[str] = None):
    """
    Plot detailed view of stocks within a specific sector.

    Args:
        sector_results: Dict mapping sector name to DataFrame
        sector_name: Name of the sector to plot
        save_path: Optional path to save the figure
    """
    if sector_name not in sector_results:
        print(f"Sector '{sector_name}' not found")
        return None

    df = sector_results[sector_name].sort_values('price_change_pct')

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))

    colors = ['green' if x > 0 else 'red' for x in df['price_change_pct']]

    bars = ax.barh(range(len(df)), df['price_change_pct'], color=colors, alpha=0.7)

    labels = [f"{row['ts_code']} {str(row['name'])[:12]}" for _, row in df.iterrows()]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels)

    for bar, val in zip(bars, df['price_change_pct']):
        x_pos = val + 0.2 if val >= 0 else val - 0.2
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', ha=ha, fontsize=9)

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Price Change (%)', fontsize=12)
    ax.set_title(f'Sector: {sector_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if INTERACTIVE_PLOTS and not save_path:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_all(df: pd.DataFrame, sector_results: Dict[str, pd.DataFrame],
             event_name: str, save_dir: Optional[str] = None):
    """
    Generate all plots for an event analysis.

    Args:
        df: DataFrame with analysis results
        sector_results: Dict mapping sector name to DataFrame
        event_name: Name of the event (used for titles and filenames)
        save_dir: Optional directory to save all figures
    """
    global INTERACTIVE_PLOTS

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        INTERACTIVE_PLOTS = False  # Disable interactive mode when saving

    prefix = event_name.lower().replace(' ', '_')

    # 1. Top gainers
    plot_top_gainers(
        df,
        title=f"Top 20 Gainers - {event_name}",
        save_path=str(save_dir / f"{prefix}_top_gainers.png") if save_dir else None
    )

    # 2. Sector performance
    plot_sector_performance(
        sector_results,
        title=f"Sector Performance - {event_name}",
        save_path=str(save_dir / f"{prefix}_sectors.png") if save_dir else None
    )

    # 3. Distribution
    plot_distribution(
        df,
        title=f"Price Change Distribution - {event_name}",
        save_path=str(save_dir / f"{prefix}_distribution.png") if save_dir else None
    )

    # 4. Price vs Volume
    plot_price_volume(
        df,
        title=f"Price vs Volume Change - {event_name}",
        save_path=str(save_dir / f"{prefix}_price_volume.png") if save_dir else None
    )

    print(f"\nAll plots generated for: {event_name}")


# Predefined events
EVENTS = {
    'trump_2017': {
        'name': 'Trump China Visit Nov 2017',
        'pre_start': 20171030,
        'pre_end': 20171107,
        'event_start': 20171108,
        'event_end': 20171110,
        'ipo_cutoff': 20171008,  # 30 days before event
        'sectors': {
            'Semiconductor/Chips': ['半导体', '芯片', '集成电路', '晶', '微电子', '封装', '存储'],
            'Energy/Oil/Gas': ['能源', '石油', '天然气', 'LNG', '油气', '石化', '燃气'],
            'Aviation/Aerospace': ['航空', '飞机', '航天', '通航'],
            'Automotive/EV': ['汽车', '新能源车', '电动', '整车', '动力电池'],
            'Equipment/Manufacturing': ['机械', '设备', '工程机械', '重工', '液压']
        }
    },
    'trump_2026': {
        'name': 'Trump China Visit Mar 2026',
        'pre_start': 20260324,
        'pre_end': 20260328,
        'event_start': 20260331,
        'event_end': 20260402,
        'ipo_cutoff': 20260301,
        'sectors': {
            'Semiconductor/Chips': ['半导体', '芯片', '集成电路', '晶', '微电子'],
            'Energy/Oil/Gas': ['能源', '石油', '天然气', '油气', '石化'],
            'Aviation/Aerospace': ['航空', '飞机', '航天'],
            'Automotive/EV': ['汽车', '新能源', '电动', '电池'],
            'AI/Technology': ['人工智能', 'AI', '算力', '大模型', '机器人']
        }
    }
}


def run(event_key: str = 'trump_2017', show_plots: bool = False, save_plots: bool = False):
    """
    Run analysis for a predefined event.

    Args:
        event_key: Key from EVENTS dict ('trump_2017', 'trump_2026', etc.)
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files

    Examples:
        run('trump_2017')              # Text output only
        run('trump_2017', show_plots=True)  # With plots
        run('trump_2017', show_plots=True, save_plots=True)  # Save plots
    """
    if event_key not in EVENTS:
        print(f"Unknown event: {event_key}")
        print(f"Available events: {list(EVENTS.keys())}")
        return

    event = EVENTS[event_key]

    # Run analysis
    df = analyze_event(
        event_name=event['name'],
        pre_start=event['pre_start'],
        pre_end=event['pre_end'],
        event_start=event['event_start'],
        event_end=event['event_end'],
        ipo_cutoff=event.get('ipo_cutoff')
    )

    if df.empty:
        return

    # Print top gainers
    print_results(df, f"TOP 30 GAINERS - {event['name']}")

    # Sector analysis
    sector_results = {}
    if 'sectors' in event:
        sector_results = analyze_by_sector(df, event['sectors'])
        print_sector_results(sector_results)

    # Generate plots if requested
    if show_plots:
        save_dir = str(DATA_DIR.parent / 'plots') if save_plots else None
        plot_all(df, sector_results, event['name'], save_dir=save_dir)

    return df


def show_stock_detail(ts_code: str, pre_start: int, pre_end: int, event_start: int, event_end: int):
    """
    Show detailed price data for a specific stock during an event.

    Args:
        ts_code: Stock code (e.g., '002049.SZ')
        pre_start, pre_end: Pre-event period
        event_start, event_end: Event period
    """
    symbol = ts_code.split('.')[0]
    exchange = ts_code.split('.')[1].lower()
    filepath = DATA_DIR / exchange / f"{symbol}.csv"

    if not filepath.exists():
        print(f"Stock file not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    df['trade_date'] = df['trade_date'].astype(int)

    stock_dict = load_stock_list()
    name = stock_dict.get(ts_code, 'Unknown')

    print(f"\n{'='*70}")
    print(f"Stock: {ts_code} - {name}")
    print(f"{'='*70}")

    # Pre-event data
    pre_df = df[(df['trade_date'] >= pre_start) & (df['trade_date'] <= pre_end)]
    print(f"\nPre-Event Period ({pre_start} to {pre_end}):")
    print(pre_df[['trade_date', 'open', 'high', 'low', 'close', 'vol']].to_string(index=False))

    # Event data
    event_df = df[(df['trade_date'] >= event_start) & (df['trade_date'] <= event_end)]
    print(f"\nEvent Period ({event_start} to {event_end}):")
    print(event_df[['trade_date', 'open', 'high', 'low', 'close', 'vol']].to_string(index=False))

    # Calculate change
    if not pre_df.empty and not event_df.empty:
        pre_close = pre_df.sort_values('trade_date')['close'].iloc[-1]
        event_close = event_df.sort_values('trade_date')['close'].iloc[-1]
        change_pct = ((event_close - pre_close) / pre_close) * 100

        print(f"\n{'='*70}")
        print(f"Pre-event close: {pre_close:.2f}")
        print(f"Event close:     {event_close:.2f}")
        print(f"Change:          {change_pct:+.2f}%")
        print(f"{'='*70}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Presidential Visit Stock Impact Analysis')
    parser.add_argument('--event', type=str, default='trump_2017',
                        help='Event to analyze (trump_2017, trump_2026)')
    parser.add_argument('--stock', type=str, default=None,
                        help='Show detail for specific stock (e.g., 002049.SZ)')
    parser.add_argument('--list-events', action='store_true',
                        help='List available predefined events')
    parser.add_argument('--plot', action='store_true',
                        help='Show visualization plots')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files (implies --plot)')

    args = parser.parse_args()

    if args.list_events:
        print("Available events:")
        for key, event in EVENTS.items():
            print(f"  {key}: {event['name']}")
    elif args.stock:
        event = EVENTS.get(args.event, EVENTS['trump_2017'])
        show_stock_detail(
            args.stock,
            event['pre_start'],
            event['pre_end'],
            event['event_start'],
            event['event_end']
        )
    else:
        show_plots = args.plot or args.save_plots
        run(args.event, show_plots=show_plots, save_plots=args.save_plots)
