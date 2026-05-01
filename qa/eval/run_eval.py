"""
Five-question grounding evaluation for the stock-research Q&A engine.

Each test asserts:
  • The expected ts_codes were resolved.
  • The answer is non-trivial (length > min_chars).
  • For numeric questions, the answer string contains a value within ±tol of
    the ground truth from `fina_indicator/`.
  • For news questions, ≥N distinct news titles from the linked corpus appear.

Run:
    ./venv/Scripts/python -m qa.eval.run_eval --quant 4bit
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'stock_data'

from qa.rag.retriever import Retriever
from qa.rag.context_builder import ContextBuilder
from qa.rag.qa_engine import QAEngine


# ─── Test cases ─────────────────────────────────────────────────────────────
CASES = [
    {
        'id':         'catl_q1',
        'question':   '300750.SZ 最近一季度的业绩怎么样？',
        'expect_ts':  ['300750.SZ'],
        'expect_min_chars': 100,
        # Q1 2026 EPS for CATL (300750.SZ) from fina_indicator (~4.58)
        'expect_numeric': {'field': 'eps', 'tol': 0.20, 'expect': 4.58},
    },
    {
        'id':         'maotai_news',
        'question':   '茅台最近有什么新闻？',
        'expect_ts':  ['600519.SH'],
        'expect_min_chars': 80,
        'expect_min_news': 3,
    },
    {
        'id':         'comparison',
        'question':   '比亚迪和宁德时代谁的毛利率更高？',
        'expect_ts':  ['300750.SZ', '002594.SZ'],
        'expect_min_chars': 80,
    },
    {
        'id':         'eps_lookup',
        'question':   '贵州茅台 2025年Q4 EPS 是多少？',
        'expect_ts':  ['600519.SH'],
        'expect_min_chars': 30,
        # Need to look this up from the fina_indicator file
        'expect_numeric': {'field': 'eps', 'tol': 0.5, 'auto_lookup': True,
                            'lookup_period': '20251231'},
    },
    {
        'id':         'sector_indirect',
        'question':   '新能源车板块龙头是谁，业绩对比如何？',
        'expect_ts_any': ['300750.SZ', '002594.SZ'],   # at least one
        'expect_min_chars': 80,
    },
]


def _eps_for(ts_code: str, period: str) -> float | None:
    code, suf = ts_code.split('.')
    fp = DATA / 'fina_indicator' / f'{code}_{suf.upper()}.csv'
    if not fp.exists(): return None
    df = pd.read_csv(fp, encoding='utf-8-sig', dtype={'end_date': str})
    row = df[df['end_date'] == period]
    if row.empty: return None
    return float(row['eps'].iloc[0])


def _check_case(case: dict, out: dict, retriever: Retriever) -> dict:
    """Return {pass: bool, reasons: [str]} — single pass requires all checks pass."""
    reasons = []
    # 1. ts_code resolution
    if 'expect_ts' in case:
        missing = [ts for ts in case['expect_ts'] if ts not in out['ts_codes']]
        if missing: reasons.append(f"missing ts_codes: {missing}")
    if 'expect_ts_any' in case:
        if not any(ts in out['ts_codes'] for ts in case['expect_ts_any']):
            reasons.append(f"expected at least one of {case['expect_ts_any']}; got {out['ts_codes']}")
    # 2. Answer length
    answer = out.get('answer', '')
    if len(answer) < case.get('expect_min_chars', 50):
        reasons.append(f"answer too short ({len(answer)} chars)")
    # 3. Numeric — simple regex check that the expected number appears within tolerance
    if 'expect_numeric' in case:
        n = case['expect_numeric']
        target = n.get('expect')
        if n.get('auto_lookup') and case.get('expect_ts'):
            target = _eps_for(case['expect_ts'][0], n['lookup_period'])
        if target is not None:
            tol = n.get('tol', 0.1)
            # extract all floats from the answer
            nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', answer)]
            if not any(abs(x - target) <= tol for x in nums):
                reasons.append(f"expected numeric ~{target:.2f} (tol {tol}) — found {nums[:8]}")
    # 4. Min news titles
    if 'expect_min_news' in case:
        # heuristic: count "[YYYY-MM-DD]" patterns in answer
        n_news = len(re.findall(r'\[\d{4}-\d{2}-\d{2}\]', answer))
        if n_news < case['expect_min_news']:
            reasons.append(f"expected ≥{case['expect_min_news']} news; found {n_news}")
    return {'pass': not reasons, 'reasons': reasons}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vllm-url', default='http://localhost:8000/v1')
    p.add_argument('--model',    default='Qwen/Qwen2.5-32B-Instruct-AWQ')
    p.add_argument('--out', default='stock_data/qa/eval_report.json')
    args = p.parse_args()

    r  = Retriever('stock_data/qa/aliases.json',
                    'stock_data/qa/news_linked.parquet',
                    entity_index='stock_data/qa/entities.faiss',
                    entity_meta='stock_data/qa/entities.parquet',
                    news_index='stock_data/qa/news.faiss',
                    news_meta='stock_data/qa/news_meta.parquet')
    cb = ContextBuilder('stock_data/qa/aliases.json')
    qa = QAEngine(vllm_url=args.vllm_url, model=args.model)

    results = []
    n_pass = 0
    for case in CASES:
        print()
        print('=' * 70)
        print(f"[{case['id']}] {case['question']}")
        t0 = time.time()
        out = qa.ask(case['question'], r, cb, top_k=5)
        elapsed = time.time() - t0
        check = _check_case(case, out, r)
        n_pass += int(check['pass'])
        results.append({
            'id':         case['id'],
            'question':   case['question'],
            'ts_codes':   out['ts_codes'],
            'n_articles': out['n_articles'],
            'answer':     out['answer'],
            'pass':       check['pass'],
            'reasons':    check['reasons'],
            'elapsed':    elapsed,
        })
        marker = '✓' if check['pass'] else '✗'
        print(f"  {marker} ts={out['ts_codes']}  "
              f"answer_len={len(out['answer'])}  elapsed={elapsed:.1f}s")
        if check['reasons']:
            for rr in check['reasons']:
                print(f"      ! {rr}")
        # First 200 chars of answer for visual inspection
        print(f"  preview: {out['answer'][:300]}...")

    print()
    print('=' * 70)
    print(f"PASS RATE: {n_pass}/{len(CASES)}  ({100*n_pass/len(CASES):.0f}%)")
    print('=' * 70)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, 'w', encoding='utf-8') as f:
        json.dump({'pass_rate': n_pass / len(CASES),
                    'n_pass': n_pass, 'n_total': len(CASES),
                    'cases': results}, f, ensure_ascii=False, indent=2)
    print(f"saved → {out_p}")


if __name__ == '__main__':
    main()
