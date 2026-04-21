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
    from api.get_data import run
    for ds in ('fund_share', 'fund_nav', 'fund_factor_pro'):
        print(f"\n[{ds}] incremental update ...")
        run(ds)


def update_fina():
    """Update quarterly financial indicators for all stocks."""
    from api.fina_indicator import run
    run('update')


def update_block():
    """Update block trade data to yesterday."""
    from api.block_trade import run
    run('download')


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\nStarting full data update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not skip_stocks:
        _run_group('stocks', update_stocks)

    _run_group('basics', update_basics)
    _run_group('index',  update_index)
    _run_group('fund',   update_fund)
    _run_group('fina',   update_fina)
    _run_group('block',  update_block)

    print(f"\nAll updates complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
