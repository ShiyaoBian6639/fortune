"""
Master Daily Update Script
Runs all incremental data downloads, optionally in parallel across groups.
Run once after market close each trading day to keep datasets current.

Usage:
    ./venv/Scripts/python update_all.py                 # full update, parallel groups
    ./venv/Scripts/python update_all.py --xgb           # only the groups xgbmodel needs (parallel)
    ./venv/Scripts/python update_all.py --sequential    # legacy serial mode
    ./venv/Scripts/python update_all.py --skip-stocks   # skip the slow OHLCV extension
    ./venv/Scripts/python update_all.py --only index    # named group only
    ./venv/Scripts/python update_all.py --check-gaps    # report missing dates without downloading
    ./venv/Scripts/python update_all.py --fill-gaps     # fill ALL missing dates (not just recent)
    ./venv/Scripts/python update_all.py --group-workers 6  # concurrent groups (default 4)

Groups (xgb-critical marked ★):
    stocks  ★ stock OHLCV (extend_stock_data)
    basics  ★ daily_basic, moneyflow, stk_limit (per-date files)
    index   ★ index_dailybasic, idx_factor_pro, index_global, index_weight
    fina    ★ quarterly financial indicators (fina_indicator)
    block   ★ block trade data
    sectors ★ SW sector classification (stock_sectors.csv)
    fund      fund_share, fund_nav, fund_factor_pro
    fina_ext  financial statements (income, balancesheet, cashflow)
    fina_extras forecast, express, dividend, fina_audit, etc.
    limits    limit_list_d, top_list, dragon-tiger data
    ths       THS index list, daily, and members
    chips     chip distribution data (cyq_perf, cyq_chips)
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

# ─── CLI parsing ──────────────────────────────────────────────────────────────

def _parse_cli():
    p = argparse.ArgumentParser('update_all', add_help=False)
    p.add_argument('--skip-stocks', action='store_true',
                   help='Skip the OHLCV extension group (it is the slowest).')
    p.add_argument('--xgb', action='store_true',
                   help='Run only the groups xgbmodel needs: '
                        'stocks, basics, index, fina, block, sectors.')
    p.add_argument('--sequential', action='store_true',
                   help='Run groups one after another (legacy mode). Default is parallel.')
    p.add_argument('--group-workers', type=int, default=4,
                   help='Max groups to run concurrently (default 4). '
                        'Each group has its own internal thread pool, so pushing this '
                        'too high multiplies concurrent Tushare connections.')
    p.add_argument('--only', default=None,
                   help='Run a single named group, e.g. --only index')
    p.add_argument('--check-gaps', action='store_true',
                   help='Report missing dates without downloading.')
    p.add_argument('--fill-gaps', action='store_true',
                   help='Fill ALL missing trading days, not just recent ones.')
    p.add_argument('-h', '--help', action='help')
    return p.parse_args()


_cli          = _parse_cli()
skip_stocks   = _cli.skip_stocks
check_gaps    = _cli.check_gaps
fill_gaps     = _cli.fill_gaps
only_group    = _cli.only
xgb_only      = _cli.xgb
run_parallel  = not _cli.sequential
group_workers = max(1, _cli.group_workers)


# ─── Trading calendar helper ─────────────────────────────────────────────────

TUSHARE_TOKEN = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
_TRADE_CAL_CACHE = {}


def get_trading_calendar(start_date: str, end_date: str) -> list:
    """
    Fetch list of trading dates from Tushare trade_cal API.
    Returns sorted list of YYYYMMDD strings.
    Cached to avoid repeated API calls.
    """
    cache_key = (start_date, end_date)
    if cache_key in _TRADE_CAL_CACHE:
        return _TRADE_CAL_CACHE[cache_key]

    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)

    for attempt in range(3):
        try:
            df = pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date,
                is_open='1'
            )
            if df is not None and not df.empty:
                dates = sorted(df['cal_date'].astype(str).tolist())
                _TRADE_CAL_CACHE[cache_key] = dates
                return dates
        except Exception as e:
            print(f"  [warn] trade_cal attempt {attempt+1} failed: {e}")
            time.sleep(1)

    return []


def get_yesterday() -> str:
    """Return yesterday's date as YYYYMMDD string."""
    return (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')


# ─── Gap detection ───────────────────────────────────────────────────────────

def detect_gaps_per_date_files(data_dir: str, prefix: str, start_date: str = '20170101') -> list:
    """
    Detect missing trading dates for per-date file datasets.
    E.g. daily_basic_YYYYMMDD.csv, moneyflow_YYYYMMDD.csv
    Returns list of missing YYYYMMDD strings.
    """
    end_date = get_yesterday()
    trade_dates = set(get_trading_calendar(start_date, end_date))

    data_path = Path(data_dir)
    if not data_path.exists():
        return sorted(trade_dates)

    existing = set()
    for f in data_path.glob(f'{prefix}_*.csv'):
        date_str = f.stem.replace(f'{prefix}_', '')
        if len(date_str) == 8 and date_str.isdigit():
            existing.add(date_str)

    missing = sorted(trade_dates - existing)
    return missing


def detect_gaps_stock_ohlcv(stock_dir: str, sample_size: int = 100) -> dict:
    """
    Detect missing trading dates in stock OHLCV files.
    Samples a subset of stocks to check for internal gaps.
    Returns dict with summary stats.
    """
    import pandas as pd
    import random

    stock_path = Path(stock_dir)
    if not stock_path.exists():
        return {'error': f'Directory not found: {stock_dir}'}

    csv_files = list(stock_path.glob('*.csv'))
    if not csv_files:
        return {'error': 'No stock files found'}

    # Sample stocks for gap check
    sample = random.sample(csv_files, min(sample_size, len(csv_files)))

    end_date = get_yesterday()
    gaps_found = []
    stale_stocks = []

    for filepath in sample:
        try:
            df = pd.read_csv(filepath, usecols=['trade_date'])
            if df.empty:
                continue

            dates = sorted(df['trade_date'].astype(str).unique())
            if not dates:
                continue

            min_date, max_date = dates[0], dates[-1]

            # Check if stock is stale (latest date > 5 trading days old)
            if max_date < end_date:
                trade_dates = get_trading_calendar(max_date, end_date)
                if len(trade_dates) > 5:
                    stale_stocks.append({
                        'file': filepath.name,
                        'latest': max_date,
                        'missing_days': len(trade_dates) - 1
                    })

            # Check for internal gaps (missing dates between min and max)
            expected = set(get_trading_calendar(min_date, max_date))
            actual = set(dates)
            internal_gaps = expected - actual
            if internal_gaps:
                gaps_found.append({
                    'file': filepath.name,
                    'gaps': len(internal_gaps),
                    'sample_gaps': sorted(internal_gaps)[:5]
                })

        except Exception as e:
            pass

    return {
        'sampled': len(sample),
        'total_files': len(csv_files),
        'stale_stocks': stale_stocks[:20],  # limit output
        'internal_gaps': gaps_found[:20],
    }


def report_all_gaps():
    """Check and report all data gaps without downloading."""
    print("\n" + "="*60)
    print("  GAP ANALYSIS REPORT")
    print("="*60)

    # Per-date datasets
    datasets = [
        ('stock_data/daily_basic', 'daily_basic'),
        ('stock_data/moneyflow', 'moneyflow'),
        ('stock_data/stk_limit', 'stk_limit'),
        ('stock_data/block_trade', 'block_trade'),
    ]

    for data_dir, prefix in datasets:
        missing = detect_gaps_per_date_files(data_dir, prefix)
        print(f"\n[{prefix}]")
        if missing:
            print(f"  Missing: {len(missing)} trading days")
            if len(missing) <= 10:
                print(f"  Dates: {missing}")
            else:
                print(f"  First 5: {missing[:5]}")
                print(f"  Last 5:  {missing[-5:]}")
        else:
            print("  Complete (no gaps)")

    # Stock OHLCV
    for subdir in ['sh', 'sz']:
        stock_dir = f'stock_data/{subdir}'
        print(f"\n[stocks/{subdir}]")
        result = detect_gaps_stock_ohlcv(stock_dir, sample_size=50)
        if 'error' in result:
            print(f"  {result['error']}")
        else:
            print(f"  Sampled {result['sampled']} of {result['total_files']} stocks")
            if result['stale_stocks']:
                print(f"  Stale stocks (>5 days behind): {len(result['stale_stocks'])}")
                for s in result['stale_stocks'][:5]:
                    print(f"    {s['file']}: latest={s['latest']}, missing={s['missing_days']} days")
            if result['internal_gaps']:
                print(f"  Stocks with internal gaps: {len(result['internal_gaps'])}")
                for g in result['internal_gaps'][:3]:
                    print(f"    {g['file']}: {g['gaps']} gaps, e.g. {g['sample_gaps']}")
            if not result['stale_stocks'] and not result['internal_gaps']:
                print("  No gaps detected in sample")

    print("\n" + "="*60)


def fill_stock_ohlcv_gaps():
    """
    Fill internal gaps in stock OHLCV files by re-downloading missing dates.
    This is slower than extend_stock_data but ensures completeness.
    """
    import pandas as pd
    import tushare as ts

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)

    print("\n[fill_stock_ohlcv_gaps] Scanning for internal gaps...")

    for subdir in ['sh', 'sz']:
        stock_dir = Path(f'stock_data/{subdir}')
        if not stock_dir.exists():
            continue

        csv_files = list(stock_dir.glob('*.csv'))
        fixed_count = 0

        for i, filepath in enumerate(csv_files):
            try:
                df = pd.read_csv(filepath)
                if df.empty or 'trade_date' not in df.columns:
                    continue

                df['trade_date'] = df['trade_date'].astype(str)
                dates = sorted(df['trade_date'].unique())
                if len(dates) < 2:
                    continue

                min_date, max_date = dates[0], dates[-1]
                expected = set(get_trading_calendar(min_date, max_date))
                actual = set(dates)
                missing = sorted(expected - actual)

                if not missing:
                    continue

                # Download missing dates
                ts_code = filepath.stem
                print(f"  [{subdir}/{ts_code}] Filling {len(missing)} gaps...")

                new_rows = []
                for gap_date in missing:
                    try:
                        gap_df = pro.daily(ts_code=ts_code, start_date=gap_date, end_date=gap_date)
                        if gap_df is not None and not gap_df.empty:
                            new_rows.append(gap_df)
                        time.sleep(0.15)  # rate limit
                    except Exception:
                        pass

                if new_rows:
                    new_df = pd.concat(new_rows, ignore_index=True)
                    combined = pd.concat([df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['trade_date'])
                    combined = combined.sort_values('trade_date', ascending=False)
                    combined.to_csv(filepath, index=False)
                    fixed_count += 1

            except Exception as e:
                pass

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(csv_files)} stocks in {subdir}/, fixed {fixed_count}")

        print(f"  {subdir}/: scanned {len(csv_files)} stocks, fixed {fixed_count} with gaps")


# ─── Helpers ──────────────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def _section(title: str):
    with _print_lock:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")


def _run_group(name: str, fn):
    """Run a single group serially. Used when --sequential or --only is set."""
    if only_group and only_group != name:
        return
    _section(name.upper())
    t0 = time.time()
    try:
        fn()
    except Exception as e:
        with _print_lock:
            print(f"  [ERROR] {name}: {e}")
    with _print_lock:
        print(f"  Elapsed: {time.time()-t0:.1f}s")


class _TaggedStdout:
    """Line-buffered stdout wrapper that prefixes every line with a tag.

    Each group runs in its own thread; redirecting sys.stdout per-thread
    keeps their logs identifiable when they interleave. Uses threading.local
    so each thread writes to its own buffer.
    """
    def __init__(self, real_stdout, tag_getter):
        self._real = real_stdout
        self._tag_getter = tag_getter
        self._buffers = threading.local()

    def _buf(self):
        if not hasattr(self._buffers, 'b'):
            self._buffers.b = []
        return self._buffers.b

    def write(self, text):
        if not text:
            return
        tag = self._tag_getter()
        if tag is None:
            return self._real.write(text)
        buf = self._buf()
        buf.append(text)
        joined = ''.join(buf)
        *complete, leftover = joined.split('\n')
        # keep the partial-line tail for next write
        self._buffers.b = [leftover] if leftover else []
        if complete:
            with _print_lock:
                for line in complete:
                    self._real.write(f'[{tag}] {line}\n')
                self._real.flush()

    def flush(self):
        buf = self._buf()
        if buf:
            tag = self._tag_getter()
            leftover = ''.join(buf).rstrip('\n')
            if leftover:
                with _print_lock:
                    if tag:
                        self._real.write(f'[{tag}] {leftover}\n')
                    else:
                        self._real.write(leftover + '\n')
            self._buffers.b = []
        self._real.flush()


_thread_tag = threading.local()
_tagged_stdout: _TaggedStdout | None = None


def _install_tagged_stdout():
    """Install the tagged stdout wrapper once. Safe to call repeatedly."""
    global _tagged_stdout
    if _tagged_stdout is None:
        _tagged_stdout = _TaggedStdout(sys.stdout, lambda: getattr(_thread_tag, 'tag', None))
        sys.stdout = _tagged_stdout


def _run_group_parallel(groups: list):
    """Run the given (name, fn) groups concurrently, tagging log output per group."""
    # Filter by --only if set
    if only_group:
        groups = [(n, f) for n, f in groups if n == only_group]
        if not groups:
            print(f"[update_all] no group matches --only {only_group}")
            return

    _install_tagged_stdout()
    timings: dict = {}
    errors: dict = {}

    def _worker(name, fn):
        _thread_tag.tag = name
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            errors[name] = str(e)
        timings[name] = time.time() - t0

    _section(f"RUNNING {len(groups)} GROUPS IN PARALLEL "
             f"(up to {group_workers} concurrent)")
    for name, _ in groups:
        print(f"  queued: {name}")

    workers = min(group_workers, len(groups))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, n, f): n for n, f in groups}
        for fut in as_completed(futs):
            name = futs[fut]
            with _print_lock:
                status = 'ERROR' if name in errors else 'done'
                print(f"\n[update_all] {name} {status} in {timings.get(name, 0):.1f}s")

    # Restore untagged output and print summary
    _thread_tag.tag = None
    _section("PARALLEL RUN SUMMARY")
    for name, _ in groups:
        t = timings.get(name, 0)
        if name in errors:
            print(f"  {name:<12} FAILED  ({t:.1f}s)  {errors[name][:120]}")
        else:
            print(f"  {name:<12} ok     ({t:.1f}s)")


# ─── Update functions ─────────────────────────────────────────────────────────

def update_stocks(fill_all_gaps: bool = False):
    """Extend all stock OHLCV files forward to today (parallel)."""
    from extend_stock_data import run
    # forward-only + parallel: checkpoint bypassed (backward=False), 8 threads,
    # rate-limited to 8 API calls/sec (480/min — safe for 7000-point accounts).
    # Raise workers/rate if you have a higher-tier account and want more speed.
    run('update', workers=8, rate=8.0)

    if fill_all_gaps:
        # Additionally fill internal gaps in stock files
        fill_stock_ohlcv_gaps()


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


def update_fina_statements():
    """Update financial statements (income, balancesheet, cashflow)."""
    from api.fina_statements import run
    run('update')


def update_fina_extras():
    """Update financial extras (forecast, express, dividend, etc.)."""
    from api.fina_extras import run
    run('update')


def update_limits():
    """Update limit and dragon-tiger data."""
    from api.limit_data import run
    run('update')
    # Aggregate to per-stock files for easier lookup
    run('aggregate')


def update_ths():
    """Update THS index data (list, daily, members)."""
    from api.ths_index import run
    run('update')


def update_chips():
    """Update chip distribution data."""
    from api.chip_data import run
    run('update')


# ─── Main ─────────────────────────────────────────────────────────────────────

def _ensure_dirs():
    """Create required data subdirectories if they don't exist."""
    import os
    for d in ('stock_data/sh', 'stock_data/sz', 'stock_data/daily_basic',
              'stock_data/moneyflow', 'stock_data/stk_limit', 'stock_data/block_trade',
              'stock_data/index/idx_factor_pro', 'stock_data/index/index_global',
              'stock_data/index/index_weight', 'stock_data/index/index_dailybasic',
              'stock_data/fina_indicator',
              # New directories for extended data
              'stock_data/fina_statements/income',
              'stock_data/fina_statements/balancesheet',
              'stock_data/fina_statements/cashflow',
              'stock_data/fina_extras/forecast',
              'stock_data/fina_extras/express',
              'stock_data/fina_extras/dividend',
              'stock_data/fina_extras/fina_audit',
              'stock_data/fina_extras/fina_mainbz',
              'stock_data/fina_extras/disclosure_date',
              'stock_data/limit_data/limit_list_d',
              'stock_data/limit_data/top_list',
              'stock_data/limit_data/top_inst',
              'stock_data/ths_index',
              'stock_data/ths_index/ths_daily',
              'stock_data/ths_index/ths_member',
              'stock_data/dc_index',
              'stock_data/chip_data/cyq_perf',
              'stock_data/chip_data/cyq_chips'):
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


def _build_groups():
    """Return (xgb_groups, extra_groups) as lists of (name, fn) tuples."""
    xgb_groups = []
    if not skip_stocks:
        xgb_groups.append(('stocks', update_stocks))
    xgb_groups += [
        ('basics',  update_basics),
        ('index',   update_index),
        ('fina',    update_fina),
        ('block',   update_block),
        ('sectors', update_sector_info),
    ]
    extra_groups = [
        ('fund',        update_fund),
        ('fina_ext',    update_fina_statements),
        ('fina_extras', update_fina_extras),
        ('limits',      update_limits),
        ('ths',         update_ths),
        ('chips',       update_chips),
    ]
    return xgb_groups, extra_groups


if __name__ == '__main__':
    print(f"\nStarting data update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  mode: {'xgb-priority' if xgb_only else 'full'}  "
          f"{'sequential' if not run_parallel else f'parallel (group_workers={group_workers})'}")

    _ensure_dirs()

    # --check-gaps: report missing dates and exit
    if check_gaps:
        report_all_gaps()
        sys.exit(0)

    if not _check_fresh_install():
        sys.exit(0)

    # --fill-gaps: fill ALL missing dates (not just recent)
    if fill_gaps:
        print("\n[fill-gaps mode] Filling ALL missing trading days...")

        # Per-date datasets: they already handle gaps via skip_existing
        _run_group('basics', update_basics)
        _run_group('block',  update_block)

        # Stock OHLCV: extend forward then fill internal gaps
        if not skip_stocks:
            _run_group('stocks', lambda: update_stocks(fill_all_gaps=True))

        print(f"\nGap filling complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(0)

    # ─── Normal incremental update ────────────────────────────────────────────
    xgb_groups, extra_groups = _build_groups()
    groups = xgb_groups if xgb_only else (xgb_groups + extra_groups)

    t0_total = time.time()

    if run_parallel and not only_group:
        _run_group_parallel(groups)
    else:
        # Sequential path (legacy) — or --only filter
        for name, fn in groups:
            _run_group(name, fn)

    print(f"\nAll updates complete in {time.time()-t0_total:.1f}s — "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
