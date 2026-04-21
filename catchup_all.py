"""
One-time catch-up script to bring all datasets up to date.
Run order chosen to respect dependencies and rate limits.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from datetime import datetime

def _section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def _elapsed(t0):
    print(f"  Elapsed: {time.time()-t0:.1f}s")

start = datetime.now()
print(f"Catch-up started — {start.strftime('%Y-%m-%d %H:%M:%S')}")

# ── 1. fina_indicator ──────────────────────────────────────────────────────────
_section("fina_indicator (incremental — all stocks)")
t0 = time.time()
try:
    from api.fina_indicator import run as fi_run
    fi_run('update')
except Exception as e:
    print(f"  [ERROR] fina_indicator: {e}")
_elapsed(t0)

# ── 2. index_weight (gap: 20260401 → today) ───────────────────────────────────
_section("index_weight")
t0 = time.time()
try:
    from api.get_data import run as gd_run
    gd_run('index_weight')
except Exception as e:
    print(f"  [ERROR] index_weight: {e}")
_elapsed(t0)

# ── 3. index_global (catch up global indices) ─────────────────────────────────
_section("index_global")
t0 = time.time()
try:
    gd_run('index_global')
except Exception as e:
    print(f"  [ERROR] index_global: {e}")
_elapsed(t0)

# ── 4. block_trade ────────────────────────────────────────────────────────────
_section("block_trade")
t0 = time.time()
try:
    from api.block_trade import run as bt_run
    bt_run('download')
except Exception as e:
    print(f"  [ERROR] block_trade: {e}")
_elapsed(t0)

# ── 5. fund_share (722 funds with data + incremental) ─────────────────────────
_section("fund_share")
t0 = time.time()
try:
    gd_run('fund_share')
except Exception as e:
    print(f"  [ERROR] fund_share: {e}")
_elapsed(t0)

# ── 6. fund_nav ───────────────────────────────────────────────────────────────
_section("fund_nav")
t0 = time.time()
try:
    gd_run('fund_nav')
except Exception as e:
    print(f"  [ERROR] fund_nav: {e}")
_elapsed(t0)

# ── 7. fund_factor_pro ────────────────────────────────────────────────────────
_section("fund_factor_pro")
t0 = time.time()
try:
    gd_run('fund_factor_pro')
except Exception as e:
    print(f"  [ERROR] fund_factor_pro: {e}")
_elapsed(t0)

# ── 8. fund_portfolio (reset checkpoint so Q4 2025 filings are fetched) ───────
_section("fund_portfolio (checkpoint reset for new quarterly data)")
t0 = time.time()
try:
    import json
    from pathlib import Path
    ckpt = Path('stock_data/fund/fund_portfolio/_checkpoint.json')
    ckpt.write_text(json.dumps({'completed': [], 'failed': []}), encoding='utf-8')
    print("  Checkpoint cleared.")
    gd_run('fund_portfolio')
except Exception as e:
    print(f"  [ERROR] fund_portfolio: {e}")
_elapsed(t0)

end = datetime.now()
print(f"\nAll catch-up complete — {end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {(end-start).total_seconds()/60:.1f} min")
