"""
Retriever: maps user questions to a small set of relevant artefacts.

For the MVP we DON'T use semantic search — just:
  1. Resolve query to ts_code(s) via the alias dict (reverse lookup)
  2. Filter the linked-news parquet by ts_code
  3. Return the top-N most-recent articles per stock

Phase 2 will add bge-m3 dense retrieval within the ts_code filter so the
"top-N" ranking reflects relevance to the specific question, not just recency.

API
---
    r = Retriever('stock_data/qa/aliases.json',
                   'stock_data/qa/news_linked.parquet')
    hits = r.search("茅台最近有什么新闻", top_k=8)
    # hits = {'ts_codes': ['600519.SH'],
    #         'articles': [{'ts_code', 'date', 'title', 'content_snippet', 'source'}, ...]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SYMBOL_RE = re.compile(r'(?<![0-9])([036]\d{5})(?![0-9])')
FULL_TS_RE = re.compile(r'\b([036]\d{5})\.(SH|SZ)\b', re.I)


class Retriever:
    def __init__(self, aliases_json: str | Path,
                 news_linked_parquet: str | Path):
        with open(aliases_json, 'r', encoding='utf-8') as f:
            self._aliases = json.load(f)        # ts_code → entry
        # Reverse maps for entity resolution
        self._name_to_ts: Dict[str, List[str]] = {}
        self._sym_to_ts:  Dict[str, str]       = {}
        for ts, v in self._aliases.items():
            self._sym_to_ts[v['symbol']] = ts
            for a in v['aliases']:
                if not a or a.isdigit() or len(a) < 2:
                    continue
                self._name_to_ts.setdefault(a, []).append(ts)
        # News
        print(f"[retriever] loading {news_linked_parquet} ...", flush=True)
        df = pd.read_parquet(news_linked_parquet)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime']).sort_values('datetime', ascending=False)
        # Explode list[str] of ts_codes for efficient per-ts lookup
        flat = df.explode('ts_codes_pred').rename(columns={'ts_codes_pred':'ts_code'})
        flat = flat.dropna(subset=['ts_code']).reset_index(drop=True)
        self._news_by_code = flat
        # Per-ts_code index for O(1) lookup
        self._news_by_code_groups = flat.groupby('ts_code', sort=False)
        print(f"[retriever] {len(df):,} linked articles  "
              f"covering {flat['ts_code'].nunique():,} ts_codes",
              flush=True)

    # ─── Entity resolution ─────────────────────────────────────────────────
    def resolve_entities(self, query: str) -> List[str]:
        """Extract ts_code(s) mentioned in the query.

        Order:
          1. Full ts_code form (600519.SH) — exact, highest priority
          2. Bare 6-digit symbol → resolve via _sym_to_ts
          3. Stock name / alias substring match (longest match wins)
        """
        out: List[str] = []
        seen = set()

        # 1. Full ts_codes
        for m in FULL_TS_RE.finditer(query):
            ts = f"{m.group(1)}.{m.group(2).upper()}"
            if ts in self._aliases and ts not in seen:
                seen.add(ts); out.append(ts)

        # 2. Bare 6-digit symbols
        for m in SYMBOL_RE.finditer(query):
            sym = m.group(1)
            ts = self._sym_to_ts.get(sym)
            if ts and ts not in seen:
                seen.add(ts); out.append(ts)

        # 3. Name / alias substring match. Iterate longest-first so that
        #    once we match a long alias we mask out its span — that way we
        #    won't re-match its substrings (e.g. "中国" inside "中国平安").
        #    A query like "比亚迪和宁德时代谁更强" should resolve BOTH 002594.SZ
        #    and 300750.SZ, not stop at the first hit.
        masked = list(query)
        sorted_aliases = sorted(self._name_to_ts.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            current = ''.join(masked)
            idx = current.find(alias)
            if idx >= 0:
                # Heuristic: ignore aliases that look generic — single-stock
                # names and explicit symbols are fine, but multi-mapped short
                # aliases (e.g. "中国" → 30 stocks) cause noise. Cap at 3
                # ts_codes per alias to keep precision.
                ts_list = self._name_to_ts[alias][:3]
                for ts in ts_list:
                    if ts not in seen:
                        seen.add(ts); out.append(ts)
                # Mask out the alias text so substrings don't re-match
                for i in range(idx, idx + len(alias)):
                    masked[i] = '\x00'
        return out

    # ─── News retrieval ────────────────────────────────────────────────────
    def news_for(self, ts_code: str, top_k: int = 8) -> List[dict]:
        """Most recent top_k linked-news articles for a single ts_code."""
        if ts_code not in self._news_by_code_groups.groups:
            return []
        g = self._news_by_code_groups.get_group(ts_code)
        g = g.head(top_k)
        out = []
        for _, r in g.iterrows():
            out.append({
                'ts_code':  ts_code,
                'datetime': r['datetime'],
                'title':    str(r.get('title') or ''),
                'content':  str(r.get('content') or '')[:400],   # snippet
                'source':   str(r.get('source') or ''),
            })
        return out

    # ─── Combined query interface ──────────────────────────────────────────
    def search(self, query: str, top_k: int = 8) -> dict:
        ts_codes = self.resolve_entities(query)
        articles = []
        for ts in ts_codes:
            articles.extend(self.news_for(ts, top_k=top_k))
        return {'ts_codes': ts_codes, 'articles': articles}


def _self_test():
    r = Retriever('stock_data/qa/aliases.json',
                   'stock_data/qa/news_linked.parquet')
    for q in ("贵州茅台最近的新闻是什么？",
              "300750.SZ 业绩怎么样",
              "比亚迪和宁德时代谁的毛利率更高",
              "600519 最近一个月业绩如何"):
        print(f"\n=== {q} ===")
        out = r.search(q, top_k=3)
        print(f"  ts_codes resolved: {out['ts_codes']}")
        for a in out['articles']:
            print(f"  - {a['ts_code']}  {a['datetime'].date()}  "
                  f"[{a['source']}] {a['title'][:50]}")


if __name__ == '__main__':
    _self_test()
