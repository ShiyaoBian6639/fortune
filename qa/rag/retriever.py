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
                 news_linked_parquet: str | Path,
                 entity_index: str | Path | None = None,
                 entity_meta:  str | Path | None = None,
                 semantic_top_k: int = 3,
                 semantic_min_score: float = 0.40):
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
        # Prominence prior: log(news_count) per ts_code. Used by the
        # semantic fallback to break dense ties — 龙头 stocks have orders
        # of magnitude more press than no-name peers in the same sector.
        nc = flat['ts_code'].value_counts()
        import numpy as _np
        self._news_count = nc.to_dict()
        self._news_count_max_log = float(_np.log1p(nc.iloc[0])) if len(nc) else 1.0
        print(f"[retriever] {len(df):,} linked articles  "
              f"covering {flat['ts_code'].nunique():,} ts_codes",
              flush=True)

        # Optional dense semantic fallback (Phase 2)
        self._entity_index = None
        self._entity_meta  = None
        self._embedder     = None
        self._semantic_top_k    = semantic_top_k
        self._semantic_min_score = semantic_min_score
        if entity_index and Path(entity_index).exists() \
           and entity_meta and Path(entity_meta).exists():
            try:
                import faiss
                self._entity_index = faiss.read_index(str(entity_index))
                self._entity_meta  = pd.read_parquet(entity_meta)
                print(f"[retriever] entity index loaded "
                      f"({self._entity_index.ntotal:,} vectors)", flush=True)
            except Exception as e:
                print(f"[retriever] could not load entity index: {e}",
                      flush=True)
                self._entity_index = None
                self._entity_meta  = None

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

    # ─── Semantic fallback (Phase 2) ───────────────────────────────────────
    def _ensure_embedder(self):
        """Lazy-load bge-m3 only when the alias path fails — saves VRAM
        for queries that resolve directly."""
        if self._embedder is None and self._entity_index is not None:
            from qa.rag.embedder import BgeM3Embedder
            self._embedder = BgeM3Embedder(max_length=256)
        return self._embedder

    # Query-intent words that describe what the user wants, not which
    # entity they want. Leaving them in causes bge-m3 to match literal
    # strings ("龙头" → "龙头股份" textile co.) instead of the sector.
    _INTENT_STOP = re.compile(
        r'(板块|龙头股票|龙头股|龙头|代表公司|代表企业|代表股|龙头企业|'
        r'股票|股价|股份|公司|企业|个股|概念股|概念|行业|赛道|领涨|领跌|'
        r'最近|今天|近期|最新|怎么样|如何|是谁|是哪些|有哪些|有什么|'
        r'业绩|新闻|对比|比较|强|弱|好|差)'
    )

    def _normalise_for_embed(self, query: str) -> str:
        s = self._INTENT_STOP.sub(' ', query)
        s = re.sub(r'[ ，,。.？?！!、；;：:\s]+', ' ', s).strip()
        return s or query   # never embed an empty string

    def semantic_resolve(self, query: str) -> List[str]:
        """Dense top-K from the entity index, then rerank by news prominence.

        Pure cosine on entity cards picks up token-level matches (e.g.
        "新能源" in any com_name, "龙头" matching "龙头股份") rather
        than industry leaders. We retrieve a wider top-K, then add a
        log-news-count prior so well-covered names (龙头 stocks have
        10×–1000× more press) bubble up.

        Final score:  cos + 0.10 · (log1p(news_count) / log1p(max_count))

        Cosine still dominates — the prior is a tie-breaker, not a
        replacement.
        """
        if self._entity_index is None or self._entity_meta is None:
            return []
        emb = self._ensure_embedder()
        if emb is None:
            return []
        import numpy as np
        q_text = self._normalise_for_embed(query)
        q = emb.encode([q_text], batch_size=1)
        # Wide initial pool — 龙头 stocks often have weaker raw cosine
        # than literal-name matches and need the prominence prior to
        # surface. 50 candidates ≈ 1 % of universe, cheap to score.
        pool_k = max(self._semantic_top_k * 16, 50)
        scores, idx = self._entity_index.search(q.astype(np.float32),
                                                 k=pool_k)
        cands: List[tuple] = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i < 0 or s < self._semantic_min_score:
                continue
            ts = self._entity_meta.iloc[i]['ts_code']
            news_n = self._news_count.get(ts, 0)
            prior = float(np.log1p(news_n) / max(self._news_count_max_log, 1e-9))
            # Prior weight 0.25: enough to flip a 0.55-cos noname under
            # a 0.45-cos 龙头 (which has 100×–1000× more news coverage).
            final = float(s) + 0.25 * prior
            cands.append((ts, final, float(s), news_n))
        cands.sort(key=lambda x: -x[1])
        return [ts for ts, _, _, _ in cands[: self._semantic_top_k]]

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
        used_semantic = False
        if not ts_codes:
            sem = self.semantic_resolve(query)
            if sem:
                ts_codes = sem
                used_semantic = True
        articles = []
        for ts in ts_codes:
            articles.extend(self.news_for(ts, top_k=top_k))
        return {'ts_codes': ts_codes, 'articles': articles,
                'used_semantic': used_semantic}


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
