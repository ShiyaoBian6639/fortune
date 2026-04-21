"""
Global Market Events & Breaking News Module using Tushare API

Downloads 5 years of global events that impact stock markets:
- US Federal Reserve interest rate decisions
- Central bank rates (SHIBOR, LIBOR, LPR)
- Major geopolitical events (wars, conflicts)
- Global economic news and events
- CCTV financial news (covers major world events)

Usage (PyCharm/Interactive):
    from features.events import run
    run('download')           # Download all event types
    run('download', 'rates')  # Download interest rates only
    run('download', 'news')   # Download event news only
    run('status')             # Check progress
    run('reset')              # Reset progress

Quick queries:
    run('fed')                # Get US Treasury rates
    run('shibor')             # Get SHIBOR rates
    run('lpr')                # Get LPR rates
    run('cctv')               # Get CCTV news
    run('events')             # Get major events today
"""

import tushare as ts
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configuration - reuse token from project
TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR = Path('./stock_data')
EVENTS_DIR = DATA_DIR / 'events'
CHECKPOINT_FILE = EVENTS_DIR / 'events_checkpoint.json'

# Rate limiting
CALL_INTERVAL = 0.3

# Date range (5 years)
END_DATE = datetime.now().strftime('%Y%m%d')
START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')

# Keywords for filtering global events
EVENT_KEYWORDS = [
    # Geopolitical
    '战争', '冲突', '军事', '制裁', '俄罗斯', '乌克兰', '中东', '以色列',
    'war', 'conflict', 'military', 'sanction',
    # Central banks & rates
    '美联储', 'Fed', '加息', '降息', '利率', '货币政策', '量化宽松', 'QE',
    '央行', '欧洲央行', 'ECB', '日本央行', 'BOJ', '英国央行',
    # Economic events
    '通胀', '通货膨胀', 'CPI', 'PPI', 'GDP', '失业率', '非农',
    '经济衰退', '金融危机', '股市崩盘', '熔断',
    # Trade & policy
    '贸易战', '关税', '脱欧', 'Brexit',
    # Major events
    '疫情', 'COVID', '新冠', '封锁',
    # Market events
    '暴跌', '暴涨', '熊市', '牛市', '黑天鹅',
]


def init_tushare():
    """Initialize Tushare API connection."""
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)
    return pro


def setup_directories():
    """Create necessary directories."""
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EVENTS_DIR / 'rates').mkdir(exist_ok=True)
    (EVENTS_DIR / 'news').mkdir(exist_ok=True)


def load_checkpoint():
    """Load download progress checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {'completed': [], 'failed': []}
    return {'completed': [], 'failed': []}


def save_checkpoint(checkpoint):
    """Save download progress checkpoint."""
    temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    temp_file.replace(CHECKPOINT_FILE)


# ============================================================
# Interest Rate Data Functions
# ============================================================

def download_us_treasury_rates(pro, start_date, end_date, max_retries=3):
    """
    Download US Treasury rates (proxy for Fed policy).

    Returns DataFrame with Treasury yields.
    """
    all_data = []

    # Try different US rate endpoints
    endpoints = [
        ('us_tltr', 'US Treasury Long-Term Rate'),
        ('us_tbr', 'US Treasury Bill Rate'),
        ('us_trltr', 'US Treasury Real Long-Term Rate'),
    ]

    for endpoint, name in endpoints:
        for attempt in range(max_retries):
            try:
                func = getattr(pro, endpoint, None)
                if func:
                    df = func(start_date=start_date, end_date=end_date)
                    if df is not None and not df.empty:
                        df['rate_type'] = name
                        all_data.append(df)
                        print(f"  {name}: {len(df)} records")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"  {name}: Failed - {e}")
        time.sleep(CALL_INTERVAL)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def download_shibor(pro, start_date, end_date, max_retries=3):
    """
    Download SHIBOR (Shanghai Interbank Offered Rate).
    """
    for attempt in range(max_retries):
        try:
            df = pro.shibor(start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  SHIBOR: Failed - {e}")
    return pd.DataFrame()


def download_shibor_lpr(pro, start_date, end_date, max_retries=3):
    """
    Download LPR (Loan Prime Rate) - key Chinese interest rate.
    """
    for attempt in range(max_retries):
        try:
            df = pro.shibor_lpr(start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  LPR: Failed - {e}")
    return pd.DataFrame()


def download_libor(pro, start_date, end_date, max_retries=3):
    """
    Download LIBOR (London Interbank Offered Rate).
    """
    for attempt in range(max_retries):
        try:
            df = pro.libor(start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  LIBOR: Failed - {e}")
    return pd.DataFrame()


def download_hibor(pro, start_date, end_date, max_retries=3):
    """
    Download HIBOR (Hong Kong Interbank Offered Rate).
    """
    for attempt in range(max_retries):
        try:
            df = pro.hibor(start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  HIBOR: Failed - {e}")
    return pd.DataFrame()


def download_all_rates(pro, start_date, end_date):
    """Download all interest rate data."""
    print("\nDownloading interest rate data...")

    results = {}

    # US Treasury rates
    print("  US Treasury rates...")
    df = download_us_treasury_rates(pro, start_date, end_date)
    if df is not None and not df.empty:
        results['us_treasury'] = df

    # SHIBOR
    print("  SHIBOR rates...")
    df = download_shibor(pro, start_date, end_date)
    if df is not None and not df.empty:
        results['shibor'] = df
        print(f"    {len(df)} records")

    # LPR
    print("  LPR rates...")
    df = download_shibor_lpr(pro, start_date, end_date)
    if df is not None and not df.empty:
        results['lpr'] = df
        print(f"    {len(df)} records")

    # LIBOR
    print("  LIBOR rates...")
    df = download_libor(pro, start_date, end_date)
    if df is not None and not df.empty:
        results['libor'] = df
        print(f"    {len(df)} records")

    # HIBOR
    print("  HIBOR rates...")
    df = download_hibor(pro, start_date, end_date)
    if df is not None and not df.empty:
        results['hibor'] = df
        print(f"    {len(df)} records")

    return results


# ============================================================
# Event News Functions
# ============================================================

def download_cctv_news(pro, start_date, end_date, max_retries=3):
    """
    Download CCTV news (covers major global events).
    Downloads day by day due to API limitations.
    """
    all_news = []
    current = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    total_days = (end - current).days
    processed = 0

    while current <= end:
        date_str = current.strftime('%Y%m%d')

        for attempt in range(max_retries):
            try:
                df = pro.cctv_news(date=date_str)
                if df is not None and not df.empty:
                    all_news.append(df)
                break
            except Exception as e:
                error_msg = str(e).lower()
                if 'limit' in error_msg or '频率' in str(e):
                    time.sleep(60)
                elif attempt < max_retries - 1:
                    time.sleep(2)

        processed += 1
        if processed % 100 == 0:
            print(f"    Progress: {processed}/{total_days} days")

        current += timedelta(days=1)
        time.sleep(CALL_INTERVAL)

    if all_news:
        return pd.concat(all_news, ignore_index=True)
    return pd.DataFrame()


def download_major_events(pro, start_date, end_date, max_retries=3):
    """
    Download major news and filter for global events.
    """
    all_events = []

    # Split into monthly chunks
    current = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    while current < end:
        chunk_end = min(current + timedelta(days=30), end)
        chunk_start_str = current.strftime('%Y%m%d')
        chunk_end_str = chunk_end.strftime('%Y%m%d')

        for attempt in range(max_retries):
            try:
                df = pro.major_news(
                    start_date=chunk_start_str,
                    end_date=chunk_end_str,
                    limit=2000
                )

                if df is not None and not df.empty:
                    # Filter for event keywords
                    mask = pd.Series([False] * len(df))

                    for keyword in EVENT_KEYWORDS:
                        if 'title' in df.columns:
                            mask |= df['title'].str.contains(keyword, na=False, regex=False)
                        if 'content' in df.columns:
                            mask |= df['content'].str.contains(keyword, na=False, regex=False)

                    filtered = df[mask].copy()
                    if not filtered.empty:
                        all_events.append(filtered)

                break
            except Exception as e:
                error_msg = str(e).lower()
                if 'limit' in error_msg or '频率' in str(e):
                    time.sleep(60)
                elif attempt < max_retries - 1:
                    time.sleep(2)

        print(f"    Processed: {chunk_start_str} to {chunk_end_str}")
        current = chunk_end
        time.sleep(CALL_INTERVAL)

    if all_events:
        result = pd.concat(all_events, ignore_index=True)
        result = result.drop_duplicates(subset=['title'] if 'title' in result.columns else None)
        return result
    return pd.DataFrame()


def download_all_news(pro, start_date, end_date):
    """Download all event-related news."""
    print("\nDownloading event news...")

    results = {}

    # Major events (filtered)
    print("  Major global events...")
    df = download_major_events(pro, start_date, end_date)
    if df is not None and not df.empty:
        results['major_events'] = df
        print(f"    Found {len(df)} event-related news items")

    # CCTV news (comprehensive)
    print("  CCTV news (this may take a while)...")
    df = download_cctv_news(pro, start_date, end_date)
    if df is not None and not df.empty:
        results['cctv_news'] = df
        print(f"    Found {len(df)} CCTV news items")

    return results


# ============================================================
# Main Download Functions
# ============================================================

def save_data(data_dict, category):
    """Save data to CSV files."""
    for name, df in data_dict.items():
        if df is not None and not df.empty:
            filepath = EVENTS_DIR / category / f"{name}.csv"
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"  Saved {name}: {len(df)} records")


def download_all(pro, download_type='all'):
    """
    Download all global events data.

    Args:
        pro: Tushare API instance
        download_type: 'all', 'rates', or 'news'
    """
    checkpoint = load_checkpoint()

    print(f"\n{'='*60}")
    print("Global Events Download")
    print(f"Date Range: {START_DATE} to {END_DATE} (5 years)")
    print(f"Download Type: {download_type}")
    print(f"{'='*60}")

    try:
        # Interest rates
        if download_type in ['all', 'rates']:
            if 'rates' not in checkpoint['completed']:
                rates_data = download_all_rates(pro, START_DATE, END_DATE)
                if rates_data:
                    save_data(rates_data, 'rates')
                    checkpoint['completed'].append('rates')
                    save_checkpoint(checkpoint)
                    print("\nInterest rates download complete!")
            else:
                print("\nInterest rates already downloaded. Use run('reset') to re-download.")

        # Event news
        if download_type in ['all', 'news']:
            if 'news' not in checkpoint['completed']:
                news_data = download_all_news(pro, START_DATE, END_DATE)
                if news_data:
                    save_data(news_data, 'news')
                    checkpoint['completed'].append('news')
                    save_checkpoint(checkpoint)
                    print("\nEvent news download complete!")
            else:
                print("\nEvent news already downloaded. Use run('reset') to re-download.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")
        save_checkpoint(checkpoint)

    print(f"\n{'='*60}")
    print("Download session complete!")
    print(f"Completed: {', '.join(checkpoint['completed']) or 'None'}")
    print(f"{'='*60}")


def show_status():
    """Show download status."""
    checkpoint = load_checkpoint()

    print(f"\n{'='*60}")
    print("Global Events Download Status")
    print(f"{'='*60}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Completed: {', '.join(checkpoint['completed']) or 'None'}")
    print(f"Failed: {', '.join(checkpoint['failed']) or 'None'}")

    # Check saved files
    print("\nSaved files:")
    for subdir in ['rates', 'news']:
        subpath = EVENTS_DIR / subdir
        if subpath.exists():
            files = list(subpath.glob('*.csv'))
            for f in files:
                size = f.stat().st_size / 1024
                print(f"  {subdir}/{f.name}: {size:.1f} KB")

    print(f"{'='*60}")


def reset_progress():
    """Reset download progress."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    print("Progress reset. Data files retained.")


# ============================================================
# Quick Query Functions
# ============================================================

def get_fed_rates(pro, start_date=None, end_date=None):
    """Get US Treasury rates (Fed policy proxy)."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

    return download_us_treasury_rates(pro, start_date, end_date)


def get_shibor(pro, start_date=None, end_date=None):
    """Get SHIBOR rates."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

    return download_shibor(pro, start_date, end_date)


def get_lpr(pro, start_date=None, end_date=None):
    """Get LPR rates."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

    return download_shibor_lpr(pro, start_date, end_date)


def get_cctv(pro, date=None):
    """Get CCTV news for a date."""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    try:
        return pro.cctv_news(date=date)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def get_events_today(pro):
    """Get major events from today's news."""
    today = datetime.now().strftime('%Y%m%d')

    try:
        df = pro.major_news(start_date=today, end_date=today, limit=500)

        if df is None or df.empty:
            return pd.DataFrame()

        # Filter for event keywords
        mask = pd.Series([False] * len(df))
        for keyword in EVENT_KEYWORDS:
            if 'title' in df.columns:
                mask |= df['title'].str.contains(keyword, na=False, regex=False)

        return df[mask]
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def display_data(df, title="Data", max_rows=10):
    """Display data in readable format."""
    if df is None or df.empty:
        print(f"No {title.lower()} found.")
        return

    print(f"\n{'='*60}")
    print(f"{title} ({len(df)} records, showing first {min(len(df), max_rows)})")
    print(f"{'='*60}\n")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head(max_rows).to_string())
    print()


def run(command, arg=None, **kwargs):
    """
    Interactive entry point for PyCharm/Jupyter.

    Download commands:
        run('download')           # Download all (rates + news)
        run('download', 'rates')  # Download interest rates only
        run('download', 'news')   # Download event news only
        run('status')             # Check progress
        run('reset')              # Reset progress

    Quick query commands:
        run('fed')                # US Treasury rates (1 year)
        run('shibor')             # SHIBOR rates (30 days)
        run('lpr')                # LPR rates (1 year)
        run('libor')              # LIBOR rates
        run('cctv')               # Today's CCTV news
        run('cctv', '20260301')   # CCTV news for date
        run('events')             # Today's major events
    """
    setup_directories()

    # Download commands
    if command == 'download':
        pro = init_tushare()
        download_type = arg if arg in ['rates', 'news'] else 'all'
        download_all(pro, download_type)
        return

    elif command == 'status':
        show_status()
        return

    elif command == 'reset':
        reset_progress()
        return

    # Quick query commands
    pro = init_tushare()
    save = kwargs.pop('save', False)
    df = None
    filename = None
    title = "Data"

    if command == 'fed':
        df = get_fed_rates(pro, **kwargs)
        title = "US Treasury Rates"
        filename = f"us_treasury_{datetime.now().strftime('%Y%m%d')}.csv"

    elif command == 'shibor':
        df = get_shibor(pro, **kwargs)
        title = "SHIBOR Rates"
        filename = f"shibor_{datetime.now().strftime('%Y%m%d')}.csv"

    elif command == 'lpr':
        df = get_lpr(pro, **kwargs)
        title = "LPR Rates"
        filename = f"lpr_{datetime.now().strftime('%Y%m%d')}.csv"

    elif command == 'libor':
        start = kwargs.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'))
        end = kwargs.get('end_date', datetime.now().strftime('%Y%m%d'))
        df = download_libor(pro, start, end)
        title = "LIBOR Rates"
        filename = f"libor_{datetime.now().strftime('%Y%m%d')}.csv"

    elif command == 'hibor':
        start = kwargs.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'))
        end = kwargs.get('end_date', datetime.now().strftime('%Y%m%d'))
        df = download_hibor(pro, start, end)
        title = "HIBOR Rates"
        filename = f"hibor_{datetime.now().strftime('%Y%m%d')}.csv"

    elif command == 'cctv':
        date = arg
        df = get_cctv(pro, date=date)
        title = "CCTV News"
        filename = f"cctv_{date or datetime.now().strftime('%Y%m%d')}.csv"

    elif command == 'events':
        df = get_events_today(pro)
        title = "Today's Major Events"
        filename = f"events_{datetime.now().strftime('%Y%m%d')}.csv"

    else:
        print("Download commands:")
        print("  run('download')           # Download all data")
        print("  run('download', 'rates')  # Interest rates only")
        print("  run('download', 'news')   # Event news only")
        print("  run('status')             # Check progress")
        print("  run('reset')              # Reset progress")
        print("\nQuick query commands:")
        print("  run('fed')                # US Treasury rates")
        print("  run('shibor')             # SHIBOR rates")
        print("  run('lpr')                # LPR rates")
        print("  run('libor')              # LIBOR rates")
        print("  run('hibor')              # HIBOR rates")
        print("  run('cctv')               # CCTV news")
        print("  run('events')             # Today's events")
        return None

    display_data(df, title)

    if save and filename and df is not None and not df.empty:
        filepath = EVENTS_DIR / filename
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Saved to {filepath}")

    return df


if __name__ == '__main__':
    print("Global Market Events Module - 5 Year Historical Download")
    print("=" * 60)
    print("Coverage:")
    print("  - US Federal Reserve / Treasury rates")
    print("  - Central bank rates (SHIBOR, LIBOR, LPR, HIBOR)")
    print("  - Geopolitical events (wars, conflicts, sanctions)")
    print("  - Economic events (Fed decisions, inflation data)")
    print("  - CCTV financial news")
    print("=" * 60)
    print("\nUsage:")
    print("  from features.events import run")
    print("  run('download')     # Download all data")
    print("  run('status')       # Check progress")
