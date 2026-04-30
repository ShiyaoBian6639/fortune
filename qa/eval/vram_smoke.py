"""
VRAM smoke test for the QA engine.

Loads QAEngine + Retriever + ContextBuilder once, then runs a series of
queries while logging torch.cuda.memory_allocated() / reserved before
and after each call. Asserts that VRAM returns to within `tol` of the
post-load baseline after each generate — i.e. KV-cache + activations
are released between requests.

Also fires two threads at the same engine to confirm the GPU lock
serialises generations (peak alloc shouldn't be ~2× a single-call peak).

Run:
    PYTHONIOENCODING=utf-8 ./venv/Scripts/python -m qa.eval.vram_smoke --quant 4bit
"""
from __future__ import annotations

import argparse
import threading
import time

import torch

from qa.rag.retriever import Retriever
from qa.rag.context_builder import ContextBuilder
from qa.rag.qa_engine import QAEngine

QUERIES = [
    '300750.SZ最近一季度的业绩怎么样',           # alias path
    '茅台最近有什么新闻',                          # alias + news
    '比亚迪和宁德时代谁的毛利率更高',                # alias + comparison (long gen)
    '锂电池龙头股有哪些',                          # entity semantic
    '美联储加息对A股的影响',                       # news semantic
]


def _vram() -> tuple[float, float]:
    return (torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved()  / 1e9)


def _bar() -> str: return '─' * 70


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quant', choices=['none','8bit','4bit'], default='4bit')
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--tol_gb', type=float, default=0.5,
                   help='Allowed drift from baseline alloc after each call.')
    args = p.parse_args()

    print(_bar())
    print(' VRAM smoke test')
    print(_bar())

    if not torch.cuda.is_available():
        print(' CUDA not available — nothing to test'); return

    print(f' device: {torch.cuda.get_device_name(0)}')
    print(f' total : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(_bar())

    print(' loading retriever ...')
    r = Retriever('stock_data/qa/aliases.json',
                   'stock_data/qa/news_linked.parquet',
                   entity_index='stock_data/qa/entities.faiss',
                   entity_meta='stock_data/qa/entities.parquet',
                   news_index='stock_data/qa/news.faiss',
                   news_meta='stock_data/qa/news_meta.parquet')
    cb = ContextBuilder('stock_data/qa/aliases.json')

    print(' loading QAEngine ...')
    qa = QAEngine(model_id=args.model, quant=args.quant)
    torch.cuda.synchronize()

    base_alloc, base_rsrv = _vram()
    print(f' baseline (model loaded): allocated={base_alloc:.2f} GB  '
          f'reserved={base_rsrv:.2f} GB')
    print(_bar())

    # ── Sequential pass ────────────────────────────────────────────
    print(' SEQUENTIAL pass:')
    peak = base_alloc
    drifts = []
    for i, q in enumerate(QUERIES, 1):
        torch.cuda.reset_peak_memory_stats()
        a0, _ = _vram()
        t0 = time.time()
        out = qa.ask(q, r, cb)
        a1, _ = _vram()
        peak_call = torch.cuda.max_memory_allocated() / 1e9
        peak = max(peak, peak_call)
        drift = a1 - base_alloc
        drifts.append(drift)
        ans_preview = (out['answer'] or '').replace('\n', ' ')[:60]
        print(f'  [{i}/{len(QUERIES)}] alloc {a0:.2f}→{a1:.2f}  '
              f'peak {peak_call:.2f}  drift {drift:+.3f} GB  '
              f'{time.time()-t0:5.1f}s  ts={out["ts_codes"]}')
        print(f'         ↪ {ans_preview}…')

    print(_bar())
    print(f' SEQ peak alloc:  {peak:.2f} GB')
    print(f' SEQ max drift:   {max(drifts):+.3f} GB  (tol {args.tol_gb})')

    if max(abs(d) for d in drifts) > args.tol_gb:
        print(' ✗ drift exceeded — KV-cache or activations are leaking '
              'between calls')
        seq_ok = False
    else:
        print(' ✓ drift within tolerance — VRAM returns to baseline')
        seq_ok = True

    # ── Concurrent pass: 2 threads, same query ─────────────────────
    print(_bar())
    print(' CONCURRENT pass (2 threads against the GPU lock):')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pre_alloc, _ = _vram()
    results = [None, None]
    starts  = [None, None]
    ends    = [None, None]

    def _worker(i, q):
        starts[i] = time.time()
        results[i] = qa.ask(q, r, cb)
        ends[i] = time.time()

    threads = [
        threading.Thread(target=_worker, args=(0, QUERIES[0])),
        threading.Thread(target=_worker, args=(1, QUERIES[2])),
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    peak_conc = torch.cuda.max_memory_allocated() / 1e9
    post_alloc, _ = _vram()

    print(f'  pre {pre_alloc:.2f}  peak {peak_conc:.2f}  post {post_alloc:.2f} GB')
    overlap = min(ends) - max(starts)
    print(f'  thread A: {ends[0]-starts[0]:.1f}s   '
          f'thread B: {ends[1]-starts[1]:.1f}s   '
          f'overlap window: {overlap:+.1f}s '
          f'(negative = serialised)')

    serial = overlap < 0.0
    bounded = peak_conc - peak < 0.5    # not significantly more than seq peak
    print(_bar())
    if serial:
        print(' ✓ generations were serialised by the lock')
    else:
        print(' ✗ generations overlapped — lock not effective')
    if bounded:
        print(f' ✓ concurrent peak ({peak_conc:.2f}) within {0.5} GB of seq peak ({peak:.2f})')
    else:
        print(f' ✗ concurrent peak ({peak_conc:.2f}) exceeded seq peak ({peak:.2f}) by '
              f'{peak_conc-peak:.2f} GB — KV-caches stacked')

    print(_bar())
    ok = seq_ok and serial and bounded
    print(' OVERALL:', 'PASS' if ok else 'FAIL')
    print(_bar())
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
