"""
Financial Statements Data Acquisition from Tushare Pro
- income: Income statement
- balancesheet: Balance sheet
- cashflow: Cash flow statement
- Saves one file per stock: stock_data/fina_statements/{statement_type}/{ts_code}.csv
- Incremental: only fetches reports newer than last end_date

Usage:
    from api.fina_statements import run

    run()                              # update all stocks for all statements
    run('update')                      # same as above
    run('income')                      # update income statements only
    run('balancesheet')                # update balance sheets only
    run('cashflow')                    # update cash flow only
    run('status')                      # show coverage summary
    run('stock', ts_code='000001.SZ')  # single stock
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import tushare as ts

# ─── Configuration ────────────────────────────────────────────────────────────

TUSHARE_TOKEN   = '54bad211769c2ef9c4a89798a9a3a804dd370db5873119ff2d005573'
DATA_DIR        = Path('./stock_data/fina_statements')
STOCK_LIST_FILE = Path('./stock_data/stock_list.csv')
START_DATE      = '20170101'

# Statement types and their API methods + fields
STATEMENT_TYPES = {
    'income': {
        'method': 'income',
        'fields': [
            'ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type',
            'end_type', 'basic_eps', 'diluted_eps', 'total_revenue', 'revenue',
            'int_income', 'prem_earned', 'comm_income', 'n_commis_income',
            'n_oth_income', 'n_oth_b_income', 'prem_income', 'out_prem',
            'une_prem_reser', 'reins_income', 'n_sec_tb_income', 'n_sec_uw_income',
            'n_asset_mg_income', 'oth_b_income', 'fv_value_chg_gain',
            'invest_income', 'ass_invest_income', 'forex_gain', 'total_cogs',
            'oper_cost', 'int_exp', 'comm_exp', 'biz_tax_surchg', 'sell_exp',
            'admin_exp', 'fin_exp', 'assets_impair_loss', 'prem_refund',
            'compens_payout', 'reser_insur_liab', 'div_payt', 'reins_exp',
            'oper_exp', 'compens_payout_refu', 'insur_reser_refu', 'reins_cost_refund',
            'other_bus_cost', 'operate_profit', 'non_oper_income', 'non_oper_exp',
            'nca_disploss', 'total_profit', 'income_tax', 'n_income', 'n_income_attr_p',
            'minority_gain', 'oth_compr_income', 't_compr_income', 'compr_inc_attr_p',
            'compr_inc_attr_m_s', 'ebit', 'ebitda', 'insurance_exp', 'undist_profit',
            'distable_profit', 'rd_exp', 'fin_exp_int_exp', 'fin_exp_int_inc',
            'transfer_surplus_rese', 'transfer_housing_imprest', 'transfer_oth',
            'adj_lossgain', 'withdra_legal_surplus', 'withdra_legal_pubfund',
            'withdra_biz_devfund', 'withdra_rese_fund', 'withdra_oth_eram',
            'workers_welfare', 'distr_profit_shrhder', 'prfshare_payable_dvd',
            'comshare_payable_dvd', 'capit_comstock_div', 'continued_net_profit',
        ],
    },
    'balancesheet': {
        'method': 'balancesheet',
        'fields': [
            'ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type',
            'end_type', 'total_share', 'cap_rese', 'undistr_porfit', 'surplus_rese',
            'special_rese', 'money_cap', 'trad_asset', 'notes_receiv', 'accounts_receiv',
            'oth_receiv', 'prepayment', 'div_receiv', 'int_receiv', 'inventories',
            'amor_exp', 'nca_within_1y', 'sett_rsrv', 'loanto_oth_bank_fi',
            'premium_receiv', 'reinsur_receiv', 'reinsur_res_receiv', 'pur_resale_fa',
            'oth_cur_assets', 'total_cur_assets', 'fa_avail_for_sale', 'htm_invest',
            'lt_eqt_invest', 'invest_real_estate', 'time_deposits', 'oth_assets',
            'lt_rec', 'fix_assets', 'cip', 'const_materials', 'fixed_assets_disp',
            'produc_bio_assets', 'oil_and_gas_assets', 'intan_assets', 'r_and_d',
            'goodwill', 'lt_amor_exp', 'defer_tax_assets', 'decr_in_disbur',
            'oth_nca', 'total_nca', 'cash_reser_cb', 'depos_in_oth_bfi',
            'prec_metals', 'deriv_assets', 'rr_reins_une_prem', 'rr_reins_outstd_cla',
            'rr_reins_lins_liab', 'rr_reins_lthins_liab', 'refund_depos',
            'ph_pledge_loans', 'refund_cap_depos', 'indep_acct_assets',
            'client_depos', 'client_prov', 'transac_seat_fee', 'invest_as_receiv',
            'total_assets', 'lt_borr', 'st_borr', 'cb_borr', 'depos_ib_deposits',
            'loan_oth_bank', 'trading_fl', 'notes_payable', 'acct_payable',
            'adv_receipts', 'sold_for_repur_fa', 'comm_payable', 'payroll_payable',
            'taxes_payable', 'int_payable', 'div_payable', 'oth_payable',
            'acc_exp', 'deferred_inc', 'st_bonds_payable', 'payable_to_reinsurer',
            'rsrv_insur_cont', 'acting_trading_sec', 'acting_uw_sec', 'non_cur_liab_due_1y',
            'oth_cur_liab', 'total_cur_liab', 'bond_payable', 'lt_payable',
            'specific_payables', 'estimated_liab', 'defer_tax_liab', 'defer_inc_non_cur_liab',
            'oth_ncl', 'total_ncl', 'depos_oth_bfi', 'deriv_liab', 'depos',
            'agency_bus_liab', 'oth_liab', 'prem_receiv_adva', 'depos_received',
            'ph_invest', 'reser_une_prem', 'reser_outstd_claims', 'reser_lins_liab',
            'reser_lthins_liab', 'indept_acc_liab', 'pledge_borr', 'indem_payable',
            'policy_div_payable', 'total_liab', 'treasury_share', 'ordin_risk_reser',
            'forex_differ', 'invest_loss_unconf', 'minority_int', 'total_hldr_eqy_exc_min_int',
            'total_hldr_eqy_inc_min_int', 'total_liab_hldr_eqy', 'lt_payroll_payable',
            'oth_comp_income', 'oth_eqt_tools', 'oth_eqt_tools_p_shr', 'lending_funds',
            'acc_receivable', 'st_fin_payable', 'payables', 'hfs_assets', 'hfs_sales',
            'cost_fin_assets', 'fair_value_fin_assets', 'cip_total', 'oth_pay_total',
            'long_pay_total', 'debt_invest', 'oth_debt_invest', 'oth_eq_invest',
            'oth_illiq_fin_assets', 'oth_eq_ppbond', 'receiv_financing', 'use_right_assets',
            'lease_liab', 'contract_assets', 'contract_liab', 'accounts_receiv_bill',
            'accounts_pay', 'oth_rcv_total', 'fix_assets_total',
        ],
    },
    'cashflow': {
        'method': 'cashflow',
        'fields': [
            'ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type',
            'end_type', 'net_profit', 'finan_exp', 'c_fr_sale_sg', 'recp_tax_rends',
            'n_depos_incr_fi', 'n_incr_loans_cb', 'n_inc_borr_oth_fi', 'prem_fr_orig_contr',
            'n_incr_insured_dep', 'n_reinsur_prem', 'n_incr_disp_tfa', 'ifc_cash_incr',
            'n_incr_disp_faas', 'n_incr_loans_oth_bank', 'n_cap_incr_repur',
            'c_fr_oth_operate_a', 'c_inf_fr_operate_a', 'c_paid_goods_s', 'c_paid_to_for_empl',
            'c_paid_for_taxes', 'n_incr_clt_loan_adv', 'n_incr_dep_cbob', 'c_pay_claims_orig_inco',
            'pay_handling_chrg', 'pay_comm_insur_plcy', 'oth_cash_pay_oper_act',
            'st_cash_out_act', 'n_cashflow_act', 'oth_recp_ral_inv_act', 'c_disp_withdrwl_invest',
            'c_recp_return_invest', 'n_recp_disp_fiam', 'n_recp_disp_soam', 'stot_inflows_inv_act',
            'c_pay_acq_const_fiam', 'c_paid_invest', 'n_disp_subs_oth_biz',
            'oth_pay_ral_inv_act', 'n_incr_pledge_loan', 'stot_out_inv_act',
            'n_cashflow_inv_act', 'c_recp_borrow', 'proc_issue_bonds', 'oth_cash_recp_ral_fnc_act',
            'stot_cash_in_fnc_act', 'free_cashflow', 'c_prepay_amt_borr', 'c_pay_dist_dpcp_int_exp',
            'incl_dvd_profit_paid_sc_ms', 'oth_cashpay_ral_fnc_act', 'stot_cashout_fnc_act',
            'n_cash_flows_fnc_act', 'eff_fx_flu_cash', 'n_incr_cash_cash_equ',
            'c_cash_equ_beg_period', 'c_cash_equ_end_period', 'c_recp_cap_contrib',
            'incl_cash_rec_saam', 'uncon_invest_loss', 'prov_depr_assets',
            'depr_fa_coga_dpam', 'amort_intang_assets', 'lt_amort_deferred_exp',
            'decr_deferred_exp', 'incr_acc_exp', 'loss_disp_fiam', 'loss_scr_fa',
            'loss_fv_chg', 'invest_loss', 'decr_def_inc_tax_assets', 'incr_def_inc_tax_liab',
            'decr_inventories', 'decr_oper_payable', 'incr_oper_payable', 'others',
            'im_net_cashflow_oper_act', 'conv_debt_into_cap', 'conv_copbonds_due_within_1y',
            'fa_fnc_leases', 'im_n_incr_cash_equ', 'net_dism_capital_add',
            'net_cash_rece_sec', 'credit_impa_loss', 'use_right_asset_dep',
            'oth_loss_asset', 'end_bal_cash', 'beg_bal_cash', 'end_bal_cash_equ',
            'beg_bal_cash_equ',
        ],
    },
}

# Rate limiting: financial APIs are stricter
WORKERS       = 4
CALLS_PER_SEC = 2.0  # Conservative for financial statements
MAX_RETRIES   = 3
RETRY_DELAY   = 2


# ─── Rate limiter ─────────────────────────────────────────────────────────────

class _RateLimiter:
    def __init__(self, rate):
        self._interval = 1.0 / rate
        self._lock     = threading.Lock()
        self._last     = 0.0

    def acquire(self):
        while True:
            with self._lock:
                now  = time.monotonic()
                wait = self._last + self._interval - now
                if wait <= 0:
                    self._last = now
                    return
            time.sleep(max(0.001, wait))


_limiter  = _RateLimiter(CALLS_PER_SEC)
_pro      = None
_pro_lock = threading.Lock()


def _get_pro():
    global _pro
    with _pro_lock:
        if _pro is None:
            ts.set_token(TUSHARE_TOKEN)
            _pro = ts.pro_api(TUSHARE_TOKEN)
        return _pro


def _fetch(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if any(k in err for k in ('exceed', 'limit', '频率', 'too many')):
                wait = 60 * (attempt + 1)
                print(f"    [rate limit] sleeping {wait}s ...")
                time.sleep(wait)
            elif any(k in err.lower() for k in ('permission', '权限', 'not subscribed')):
                print(f"    [permission denied] {err[:100]}")
                return None
            else:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [{type(e).__name__}] retry {attempt+1}/{MAX_RETRIES} in {wait}s ...")
                time.sleep(wait)
    return None


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _load_ckpt(stmt_type: str):
    ckpt_file = DATA_DIR / stmt_type / '_checkpoint.json'
    if ckpt_file.exists():
        try:
            data = json.loads(ckpt_file.read_text(encoding='utf-8'))
            return set(data.get('completed', [])), set(data.get('failed', []))
        except Exception:
            pass
    return set(), set()


def _save_ckpt(stmt_type: str, completed, failed):
    ckpt_file = DATA_DIR / stmt_type / '_checkpoint.json'
    ckpt_file.write_text(
        json.dumps({'completed': sorted(completed), 'failed': sorted(failed)}, indent=2),
        encoding='utf-8'
    )


# ─── Per-stock helpers ────────────────────────────────────────────────────────

def _setup(stmt_type: str):
    (DATA_DIR / stmt_type).mkdir(parents=True, exist_ok=True)


def _last_end_date(stmt_type: str, ts_code: str) -> Optional[str]:
    fp = DATA_DIR / stmt_type / f"{ts_code.replace('.', '_')}.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, usecols=['end_date'])
        if df.empty:
            return None
        return str(int(float(df['end_date'].max())))
    except Exception:
        return None


def _load_stock_list() -> List[str]:
    if STOCK_LIST_FILE.exists():
        df = pd.read_csv(STOCK_LIST_FILE)
        for col in ('ts_code', 'code', 'symbol'):
            if col in df.columns:
                codes = df[col].astype(str).tolist()
                result = []
                for c in codes:
                    c = c.strip()
                    if '.' not in c:
                        c = c + ('.SH' if c.startswith('6') else '.SZ')
                    result.append(c)
                return result
    # Fallback: scan sh/ and sz/ directories
    codes = []
    for d, suffix in [('./stock_data/sh', 'SH'), ('./stock_data/sz', 'SZ')]:
        p = Path(d)
        if p.exists():
            codes += [f.stem + '.' + suffix for f in p.glob('*.csv')]
    return codes


def _download_one(stmt_type: str, ts_code: str) -> tuple:
    """Download statement for a single stock. Returns (ts_code, status)."""
    stmt_cfg = STATEMENT_TYPES[stmt_type]
    fp       = DATA_DIR / stmt_type / f"{ts_code.replace('.', '_')}.csv"
    last     = _last_end_date(stmt_type, ts_code)

    # Period filter for incremental updates
    period_start = last if last else START_DATE

    pro = _get_pro()
    method = getattr(pro, stmt_cfg['method'])

    # Try fetching with period filter
    df = _fetch(
        method,
        ts_code=ts_code,
        start_date=period_start,
    )

    if df is None:
        return ts_code, 'error'
    if df.empty:
        return ts_code, 'no_new_data'

    # Filter to available columns only
    available_cols = [c for c in stmt_cfg['fields'] if c in df.columns]
    df = df[available_cols].copy()
    df = df.sort_values('end_date')

    if fp.exists() and last is not None:
        existing = pd.read_csv(fp)
        # Normalize end_date
        if 'end_date' in existing.columns and pd.api.types.is_numeric_dtype(existing['end_date']):
            existing['end_date'] = existing['end_date'].apply(
                lambda x: str(int(x)) if pd.notna(x) else x)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=['ts_code', 'end_date', 'report_type'], keep='last')
        df = df.sort_values('end_date')

    df.to_csv(fp, index=False, encoding='utf-8-sig')
    return ts_code, f'+{len(df)} rows'


# ─── Update functions ─────────────────────────────────────────────────────────

def update_statement(stmt_type: str, batch: Optional[int] = None, retry_only: bool = False):
    """Update a single statement type for all stocks."""
    _setup(stmt_type)
    all_stocks = _load_stock_list()
    completed, failed = _load_ckpt(stmt_type)

    if retry_only:
        todo = [c for c in all_stocks if c in failed]
        print(f"[{stmt_type}] Retrying {len(todo)} previously-failed stocks ...")
    else:
        todo = all_stocks
        print(f"[{stmt_type}] {len(all_stocks)} total (incremental per end_date) ...")

    if batch:
        todo = todo[:batch]
        print(f"  (batch limit: processing first {len(todo)})")

    if not todo:
        print("Nothing to do.")
        return

    ckpt_lock  = threading.Lock()
    counters   = {'ok': 0, 'err': 0, 'done': 0}
    total      = len(todo)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_download_one, stmt_type, code): code for code in todo}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                _, st = fut.result()
                is_err = (st == 'error')
            except Exception as exc:
                st     = f'exception: {exc}'
                is_err = True

            with ckpt_lock:
                counters['done'] += 1
                if is_err:
                    counters['err'] += 1
                    failed.add(code)
                    completed.discard(code)
                else:
                    counters['ok'] += 1
                    completed.add(code)
                    failed.discard(code)
                if counters['done'] % 50 == 0:
                    _save_ckpt(stmt_type, completed, failed)

            if counters['done'] % 100 == 0 or is_err:
                print(f"  [{counters['done']}/{total}] {code}: {st}")

    _save_ckpt(stmt_type, completed, failed)
    print(f"\n[{stmt_type}] Done. ok={counters['ok']} errors={counters['err']}")


def update_all_statements(batch: Optional[int] = None):
    """Update all statement types for all stocks."""
    for stmt_type in STATEMENT_TYPES:
        update_statement(stmt_type, batch=batch)


def status():
    """Print coverage summary for all statement types."""
    for stmt_type in STATEMENT_TYPES:
        stmt_dir = DATA_DIR / stmt_type
        if not stmt_dir.exists():
            print(f"[{stmt_type}] Not downloaded yet")
            continue
        files = list(stmt_dir.glob('[!_]*.csv'))
        completed, failed = _load_ckpt(stmt_type)
        print(f"[{stmt_type}] {len(files)} stock files")
        print(f"  Checkpoint: {len(completed)} completed, {len(failed)} failed")
        if files:
            for fp in sorted(files)[:2]:
                try:
                    df = pd.read_csv(fp, usecols=['end_date'])
                    print(f"  {fp.stem}: {df['end_date'].min()} -> {df['end_date'].max()} ({len(df)} rows)")
                except Exception:
                    pass


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(action: str = 'update', **kwargs):
    """
    Entry point for financial statements operations.

    Actions:
        'update'       - update all statement types for all stocks
        'income'       - update income statements only
        'balancesheet' - update balance sheets only
        'cashflow'     - update cash flow statements only
        'retry'        - re-attempt failed stocks for all types
        'stock'        - update single stock (requires ts_code kwarg)
        'status'       - show coverage summary

    Keyword args:
        ts_code  (str) - stock code for 'stock' action
        batch    (int) - max stocks to process
    """
    if action == 'update':
        update_all_statements(batch=kwargs.get('batch'))

    elif action in STATEMENT_TYPES:
        update_statement(action, batch=kwargs.get('batch'))

    elif action == 'retry':
        for stmt_type in STATEMENT_TYPES:
            update_statement(stmt_type, batch=kwargs.get('batch'), retry_only=True)

    elif action == 'stock':
        ts_code = kwargs.get('ts_code')
        if not ts_code:
            print("Error: ts_code required for 'stock' action")
        else:
            for stmt_type in STATEMENT_TYPES:
                _setup(stmt_type)
                code, st = _download_one(stmt_type, ts_code)
                print(f"[{stmt_type}] {code}: {st}")

    elif action == 'status':
        status()

    else:
        print(f"Unknown action: {action!r}. Valid: update | income | balancesheet | cashflow | retry | stock | status")


if __name__ == '__main__':
    run('status')
    run('update')
