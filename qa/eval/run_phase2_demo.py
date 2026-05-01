"""
Run all 50 demo questions through the live API and grade them.

Each question is parsed from qa/PHASE2_DEMO_QUESTIONS.md (tables A / B / C).
Per question we record:
  - resolved ts_codes
  - whether at least one matches the "expected" stocks named in the doc
  - answer length, latency
  - repetition signal (any 8-gram repeated ≥ 3×)
  - presence of "未提供" / "未识别" hedging tokens

Categories:
  A — entity-flavor leader queries  (expect ≥1 of the listed stocks)
  B — news-flavor meta queries      (ts_codes=[] is fine; require non-trivial answer)
  C — mixed concept/theme queries   (expect ≥1 of the listed stocks)

Run:
    PYTHONIOENCODING=utf-8 ./venv/Scripts/python -m qa.eval.run_phase2_demo \\
        --api http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_MD = ROOT / 'qa' / 'PHASE2_DEMO_QUESTIONS.md'

# ─── Parse the demo doc ────────────────────────────────────────────────────
_NAME_TO_TS = None

def _build_name_index():
    """name → ts_code, populated from aliases.json."""
    global _NAME_TO_TS
    if _NAME_TO_TS is not None: return _NAME_TO_TS
    aliases = json.loads((ROOT / 'stock_data' / 'qa' / 'aliases.json')
                          .read_text(encoding='utf-8'))
    out = {}
    for ts, v in aliases.items():
        for n in [v.get('name', ''), *v.get('aliases', [])]:
            if n and len(n) >= 2:
                out.setdefault(n, ts)
    _NAME_TO_TS = out
    return out


_TABLE_ROW = re.compile(r'^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$')
_CATEGORY  = re.compile(r'^##\s+([A-C])\.\s+')


def parse_demo() -> list:
    """Return list of {id, category, question, expected_names: list[str]}."""
    out = []
    cat = None
    for line in DEMO_MD.read_text(encoding='utf-8').splitlines():
        m_cat = _CATEGORY.match(line)
        if m_cat:
            cat = m_cat.group(1)
            continue
        m = _TABLE_ROW.match(line)
        if not m:
            continue
        idx, q, expected = m.groups()
        # Skip header / separator rows
        if not idx.isdigit(): continue
        if q.strip() in ('Question', 'expected behaviour'): continue
        names = [n.strip() for n in re.split(r'[/／]+', expected)
                  if n.strip() and not n.strip().startswith('(')]
        out.append({
            'id':            int(idx),
            'category':      cat,
            'question':      q,
            'expected_names': names,
        })
    return out


def names_to_ts(names: list) -> list:
    idx = _build_name_index()
    out = []
    for n in names:
        # Try exact, then prefix match
        if n in idx:
            out.append(idx[n]); continue
        for k, v in idx.items():
            if n in k or k in n:
                out.append(v); break
    return out


# ─── Quality checks ────────────────────────────────────────────────────────
def repetition_score(text: str, n: int = 8, threshold: int = 3) -> int:
    """Count distinct n-grams that repeat ≥ threshold times."""
    text = re.sub(r'\s+', '', text)
    grams = Counter(text[i: i+n] for i in range(len(text) - n + 1))
    return sum(1 for g, c in grams.items() if c >= threshold and g.strip())


_HEDGE_RE = re.compile(r'(资料中未提供|未能识别|未提及|资料中并未|无法回答|无法判断)')

def grade(case: dict, response: dict) -> dict:
    answer  = response.get('answer', '') or ''
    ts_got  = set(response.get('ts_codes', []))
    expected_ts = set(names_to_ts(case['expected_names']))
    n_arts = response.get('n_articles', 0)

    # Category-specific pass rule
    if case['category'] == 'B':
        # Meta: ts_codes=[] OK; answer must be substantive + low repetition
        ts_pass = True
        passed  = (len(answer) >= 200 and
                    repetition_score(answer) <= 1 and
                    n_arts >= 1)
    else:
        ts_pass = bool(expected_ts & ts_got) if expected_ts else (len(ts_got) > 0)
        passed  = (ts_pass and
                    len(answer) >= 100 and
                    repetition_score(answer) <= 1)
    return {
        'pass':         bool(passed),
        'ts_match':     ts_pass,
        'expected_ts':  sorted(expected_ts),
        'got_ts':       sorted(ts_got),
        'answer_len':   len(answer),
        'n_articles':   n_arts,
        'rep_score':    repetition_score(answer),
        'hedged':       bool(_HEDGE_RE.search(answer)),
        'elapsed_s':    response.get('elapsed_seconds', 0),
        'cached':       response.get('cached', False),
    }


# ─── Runner ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--api',   default='http://127.0.0.1:8080')
    p.add_argument('--out',   default='stock_data/qa/phase2_demo_report.json')
    p.add_argument('--limit', type=int, default=0,
                   help='If >0, only run first N questions.')
    args = p.parse_args()

    cases = parse_demo()
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"[demo] running {len(cases)} questions against {args.api}")

    results = []
    t_start = time.time()
    for i, c in enumerate(cases, 1):
        t0 = time.time()
        try:
            r = requests.post(f"{args.api}/ask",
                               json={'query': c['question'], 'top_k': 5},
                               timeout=180)
            r.raise_for_status()
            data = r.json()
            data['elapsed_seconds'] = data.get('elapsed_seconds', time.time() - t0)
        except Exception as e:
            data = {'answer': f'[ERROR] {e}', 'ts_codes': [],
                    'n_articles': 0, 'elapsed_seconds': time.time() - t0}
        g = grade(c, data)
        results.append({
            'id':            c['id'],
            'category':      c['category'],
            'question':      c['question'],
            'expected':      c['expected_names'],
            **g,
            'answer':        data.get('answer', ''),
        })
        marker = '✓' if g['pass'] else '✗'
        print(f"  [{i:>2}/{len(cases)}] {marker} cat={c['category']}  "
              f"got={g['got_ts'][:3]}  exp={g['expected_ts'][:2]}  "
              f"len={g['answer_len']:>4}  rep={g['rep_score']}  "
              f"hedge={int(g['hedged'])}  {g['elapsed_s']:.1f}s  "
              f"{c['question'][:30]}")

    n_pass = sum(1 for r in results if r['pass'])
    by_cat = Counter()
    pass_by_cat = Counter()
    for r in results:
        by_cat[r['category']] += 1
        pass_by_cat[r['category']] += int(r['pass'])
    print()
    print("=" * 60)
    print(f" PASS RATE: {n_pass}/{len(results)} "
          f"({100*n_pass/max(len(results),1):.0f}%)")
    for c in 'ABC':
        if by_cat[c]:
            print(f"   cat {c}: {pass_by_cat[c]}/{by_cat[c]}")
    print(f" total time: {(time.time()-t_start)/60:.1f} min")
    print("=" * 60)

    out_p = ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        'pass_rate':  n_pass / max(len(results), 1),
        'n_pass':     n_pass,
        'n_total':    len(results),
        'by_category': dict(by_cat),
        'pass_by_category': dict(pass_by_cat),
        'cases':      results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f" saved → {out_p}")


if __name__ == '__main__':
    main()
