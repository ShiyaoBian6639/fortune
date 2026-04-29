"""
Aho-Corasick lexical stock-news linker.

Builds an automaton from `stock_data/qa/aliases.json` and emits, per article,
the set of ts_codes whose name / com_name / short-name / 6-digit symbol
appears in the article's title+content.

Notes
-----
1. Multi-alias collisions: an alias may map to multiple ts_codes (e.g. "中国"
   appears in many full names). We require at least one alias of length ≥ 4
   (i.e. exclude bare 6-digit numerics if they would also match a substring
   of another stock code), AND exclude single-character matches.
2. Bare 6-digit symbol matches are gated by digit boundary (`\b`) so "600519"
   doesn't trigger inside "60051900".
3. Output is a per-article ts_codes list, deduped.

Usage:
    from qa.linker.ahocorasick_matcher import build_matcher, match_article
    matcher = build_matcher('stock_data/qa/aliases.json')
    codes   = match_article(matcher, title, content)

Or as a CLI:
    ./venv/Scripts/python -m qa.linker.ahocorasick_matcher --self_test
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import ahocorasick


# Aliases shorter than this are dropped from the lexical automaton because
# they collide too often (e.g. "中国" appears in 800+ company names).
MIN_ALIAS_LEN = 3
# Symbols are exactly 6 digits — they get a separate \b-bounded check.
SYMBOL_RE = re.compile(r'(?<![0-9])([036]\d{5})(?![0-9])')


def _is_pure_digit(s: str) -> bool:
    return s.isdigit()


def build_matcher(aliases_json: str | Path) -> Tuple[ahocorasick.Automaton, Dict[str, str]]:
    """Build an Aho-Corasick automaton from the alias dict.

    Returns
    -------
    (automaton, sym_to_ts) where:
      automaton  — Aho-Corasick automaton ready for `iter()`
      sym_to_ts  — dict {6-digit symbol → ts_code} for bare-symbol fallback
    """
    with open(aliases_json, 'r', encoding='utf-8') as f:
        d = json.load(f)

    A = ahocorasick.Automaton()
    sym_to_ts: Dict[str, str] = {}
    n_words = 0
    for ts, v in d.items():
        sym_to_ts[v['symbol']] = ts
        for alias in v['aliases']:
            if not alias or len(alias) < MIN_ALIAS_LEN:
                continue
            if _is_pure_digit(alias):
                continue   # handled separately by SYMBOL_RE
            # Storing a SET: a single alias can map to multiple ts_codes
            # (e.g. shared short names). At iter-time we resolve to the union.
            try:
                existing = A.get(alias)
                existing.add(ts)
            except KeyError:
                A.add_word(alias, {ts})
            n_words += 1

    A.make_automaton()
    print(f"[ac] loaded {len(d):,} stocks → {n_words:,} alias entries "
          f"(min_len={MIN_ALIAS_LEN})", flush=True)
    return A, sym_to_ts


def match_article(matcher: Tuple, title: str, content: str) -> List[str]:
    """Return the set of ts_codes mentioned in title+content."""
    A, sym_to_ts = matcher
    text = (title or '') + ' ' + (content or '')
    if not text.strip():
        return []
    hits: Set[str] = set()
    for end_idx, codes in A.iter(text):
        # codes is a set of ts_codes that share this alias
        hits.update(codes)
    # Bare 6-digit symbol fallback (\b-gated)
    for m in SYMBOL_RE.finditer(text):
        sym = m.group(1)
        if sym in sym_to_ts:
            hits.add(sym_to_ts[sym])
    return sorted(hits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aliases', default='stock_data/qa/aliases.json')
    p.add_argument('--self_test', action='store_true')
    args = p.parse_args()

    matcher = build_matcher(args.aliases)

    if args.self_test:
        cases = [
            ("贵州茅台业绩超预期",
             "贵州茅台股份有限公司发布2025年Q4业绩，营业收入超预期。"),
            ("宁德时代签约欧洲客户",
             "宁德时代新能源科技与德国大众签订电池供应协议，覆盖2026-2030。"),
            ("新能源车板块大涨",
             "比亚迪与CATL双双涨停，板块整体走强。CATL = 300750.SZ"),
            ("茅台没出现", "今天市场震荡，沪指收红，沪深300上涨1.2%。"),
            ("纯6位数测试", "标的代码 600519、000001 同时上涨"),
        ]
        for i, (title, content) in enumerate(cases, 1):
            codes = match_article(matcher, title, content)
            print(f"\n[case {i}] title='{title[:30]}...'  → {codes}")


if __name__ == '__main__':
    main()
