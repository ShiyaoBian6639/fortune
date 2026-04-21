"""
Master Daily Update Script
Runs all incremental data downloads in dependency order.
Run this once after market close each trading day to keep all datasets current.

Usage:
    ./venv/Scripts/python update_all.py             # full update
    ./venv/Scripts/python update_all.py --skip-stocks  # skip slow OHLCV extension
    ./venv/Scripts/python update_all.py --only index   # named group only

Groups:
    stocks  – stock OHLCV (extend_stock_data)
    basics  – daily_basic, moneyflow, stk_limit (per-date files)
    index   – index_dailybasic, idx_factor_pro, index_global, index_weight
    fund    – fund_share, fund_nav, fund_factor_pro
    fina    – quarterly financial indicators (fina_indicator)
    block   – block trade data
"""

import sys
import time
from datetime import datetime

# ─── Optional CLI args ────────────────────────────────────────────────────────

args         = set(sys.argv[1:])
skip_stocks  = '--skip-stocks' in args
only_group   = None
for a in args:
    if a.startswith('--only'):
        only_group = a.split('=')[-1] if '=' in a else (sys.argv[sys.argv.index(a)+1] if sys.argv.index(a)+1 < len(sys.argv) else None)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _run_group(name, fn):
    if only_group and only_group != name:
        return
    _section(name.upper())
    t0 = time.time()
    try:
        fn()
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")


# ─── Update functions ─────────────────────────────────────────────────────────

def update_stocks():
    """Extend all stock OHLCV files forward to today (parallel)."""
    from extend_stock_data import run
    # forward-only + parallel: checkpoint bypassed (backward=False), 8 threads,
    # rate-limited to 8 API calls/sec (480/min — safe for 7000-point accounts).
    # Raise workers/rate if you have a higher-tier account and want more speed.
    run('update', workers=8, rate=8.0)


def update_basics():
    """Catch up daily_basic, moneyflow, and stk_limit to yesterday."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    # daily_basic
    from dailybasic import run as db_run
    print("[daily_basic] incremental update ...")
    db_run('range')   # range with no args → 2017-01-01 to yesterday, skip existing

    # moneyflow
    from api.moneyflow import run as mf_run
    print("\n[moneyflow] incremental update ...")
    mf_run('download')

    # stk_limit
    from api.stk_limit import run as sl_run
    print("\n[stk_limit] incremental update ...")
    sl_run('download')


def update_index():
    """Update all index time-series datasets."""
    from api.get_data import run
    for ds in ('index_dailybasic', 'idx_factor_pro', 'index_global', 'index_weight'):
        print(f"\n[{ds}] incremental update ...")
        run(ds)


def update_fund():
    """Update fund time-series datasets (nav, share, factor)."""
    import os
    from api.get_data import run

    # fund_basic.csv is a prerequisite — download it if missing
    fund_basic_path = os.path.join('stock_data', 'fund', 'fund_basic.csv')
    if not os.path.exists(fund_basic_path):
        print("[fund_basic] Not found — downloading prerequisite ...")
        run('fund_basic')

    for ds in ('fund_share', 'fund_nav', 'fund_factor_pro'):
        print(f"\n[{ds}] incremental update ...")
        run(ds)


def update_fina():
    """Update quarterly financial indicators for all stocks."""
    from api.fina_indicator import run
    run('update')


def update_sector_info():
    """Download enriched SW sector classification and static stock attributes."""
    from api.sector_info import run
    run()


def update_block():
    """Update block trade data to yesterday."""
    from api.block_trade import run
    run('download')


# ─── Main ─────────────────────────────────────────────────────────────────────

def _ensure_dirs():
    """Create required data subdirectories if they don't exist."""
    import os
    for d in ('stock_data/sh', 'stock_data/sz', 'stock_data/daily_basic',
              'stock_data/moneyflow', 'stock_data/stk_limit', 'stock_data/block_trade',
              'stock_data/index/idx_factor_pro', 'stock_data/index/index_global',
              'stock_data/index/index_weight', 'stock_data/index/index_dailybasic',
              'stock_data/fina_indicator'):
        os.makedirs(d, exist_ok=True)


def _check_fresh_install():
    """Detect empty data folder and prompt for initial download."""
    import os
    sh_count = len([f for f in os.listdir('stock_data/sh') if f.endswith('.csv')]) if os.path.isdir('stock_data/sh') else 0
    if sh_count == 0:
        print("\n  [WARNING] No stock data found (stock_data/sh is empty).")
        print("  For a fresh install, run the initial download first:")
        print("    python get_original_data.py")
        print("  Then re-run update_all.py for incremental updates.\n")
        return False
    return True


if __name__ == '__main__':
    print(f"\nStarting full data update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    _ensure_dirs()

    if not _check_fresh_install():
        sys.exit(0)

    if not skip_stocks:
        _run_group('stocks', update_stocks)

    _run_group('basics', update_basics)
    _run_group('index',  update_index)
    _run_group('fund',   update_fund)
    _run_group('fina',    update_fina)
    _run_group('block',   update_block)
    _run_group('sectors', update_sector_info)

    print(f"\nAll updates complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
